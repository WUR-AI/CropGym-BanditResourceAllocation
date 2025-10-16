from typing import Sequence, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy import dtype
from tianshou.data import Batch, to_torch
from tianshou.data.types import RecurrentStateBatch
from tianshou.utils.net.common import NetBase, Recurrent, MLP
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
from torch import nn


class IntrinsicCuriosityModuleMARL(IntrinsicCuriosityModule):
    def forward(
            self,
            s1: np.ndarray | torch.Tensor | Batch,
            act: np.ndarray | torch.Tensor | Batch,
            s2: np.ndarray | torch.Tensor | Batch,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert for our very own cropgymzoo
        if isinstance(s1, Batch):
            s1 = s1.obs
        if isinstance(s2, Batch):
            s2 = s2.obs
        if act.ndim > 1:
            act = act.squeeze(-1)

        r"""Mapping: s1, act, s2 -> mse_loss, act_hat."""
        s1 = to_torch(s1, dtype=torch.float32, device=self.device)
        s2 = to_torch(s2, dtype=torch.float32, device=self.device)
        phi1, phi2 = self.feature_net(s1), self.feature_net(s2)
        act = to_torch(act, dtype=torch.long, device=self.device)
        phi2_hat = self.forward_model(
            torch.cat([phi1, F.one_hot(act, num_classes=self.action_dim)], dim=1),
        )
        mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        return mse_loss, act_hat


class FiLMHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, cond_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.gam1 = nn.Linear(cond_dim, hidden_dim)
        self.bet1 = nn.Linear(cond_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.out  = nn.Linear(hidden_dim, out_dim)
        # identity init for (1+gamma)*h + beta at start
        nn.init.zeros_(self.gam1.weight)
        nn.init.zeros_(self.gam1.bias)
        nn.init.zeros_(self.bet1.weight)
        nn.init.zeros_(self.bet1.bias)

    def forward(self, x, cond):
        h = self.lin1(x)                      # [B,T,H]
        gamma = self.gam1(cond)
        beta = self.bet1(cond)
        h = (1 + gamma) * h + beta   # FiLM on hidden pre-activation
        h = self.act(h)
        return self.out(h)                    # logits


# class BudgetCond(nn.Module):
#     """
#     Creates a conditioning vector from budget information using a proper
#     embedding layer for the discretized budget bin.
#     """
#
#     def __init__(self, n_bins, emb_dim=8):
#         super().__init__()
#         # Use a real embedding layer for the categorical bin information
#         # n_bins + 1 because bucketize can output values from 0 to n_bins inclusive
#         self.bin_embedding = nn.Embedding(n_bins + 1, emb_dim)
#
#         # The output dimension will be the embedding dim + 1 continuous feature
#         self.out_dim = emb_dim + 1
#
#     def forward(self, rem, tot, budget_bin):
#         # 1. Calculate the continuous budget fraction
#         rem_frac = (rem / tot.clamp_min(1e-8)).clamp(0, 1.0).unsqueeze(-1)  # Shape: [B, 1] or [B, T, 1]
#
#         # 2. Get the learned embedding for the budget bin
#         # .long() is required for the embedding layer lookup
#         bin_emb = self.bin_embedding(budget_bin.long())  # Shape: [B, emb_dim] or [B, T, emb_dim]
#
#         # 3. Concatenate the continuous feature and the learned categorical feature
#         # This provides a much richer signal to the FiLM layers
#         cond = torch.cat([rem_frac, bin_emb], dim=-1)
#
#         return cond.float()

class BudgetCond(nn.Module):
    def __init__(self, n_bins, emb_dim=8):
        super().__init__()
        self.n_bins = n_bins
        # self.emb = nn.Embedding(n_bins, emb_dim)
        self.out_dim = 2 + emb_dim   # rem_frac, tot_frac, bin_emb

    def forward(self, rem, tot, budget_bin):
        rem_frac = (rem / tot.clamp_min(1e-8)).clamp(0, 1.0)
        # tot_frac = torch.ones_like(rem_frac)  # or use (rem/tot); keep it simple
        z = torch.stack(
            [
                rem_frac,
                (budget_bin / self.n_bins)
            ], dim=-1).float()  # [B,T,2+emb]
        return z


class ActorFiLM(nn.Module):
    """Condition budget features for actor"""
    def __init__(self, feat_dim, cont_in=2, n_bins=0, emb_dim=16, hidden=32):
        super().__init__()
        self.has_bins = n_bins > 0
        if self.has_bins:
            self.bin_emb = nn.Embedding(n_bins, emb_dim)
            comb_in = emb_dim + cont_in
        else:
            self.bin_emb = None
            comb_in = cont_in

        self.mlp = nn.Sequential(
            nn.Linear(comb_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        # produce FiLM params matching feature width
        self.gamma = nn.Linear(hidden, feat_dim)   # scale
        self.beta  = nn.Linear(hidden, feat_dim)   # shift

    def forward(self, rem_budget, tot_budget, budget_bin=None):
        # rem & tot are tensors with shape [B,T] or [B] → we’ll produce [B,T,H]
        # build continuous block: [rem, frac]
        frac = (rem_budget / (tot_budget.clamp_min(1e-8))).clamp(0, 10)  # safe
        cont = torch.stack([rem_budget, frac], dim=-1)  # [..., 2]

        if self.has_bins:
            z_bin = self.bin_emb(budget_bin)     # [..., emb_dim]
            z = torch.cat([cont, z_bin], dim=-1)
        else:
            z = cont
        z = z.float()
        h = self.mlp(z)
        g = self.gamma(h)
        b = self.beta(h)
        return g, b


class ObsMLP(MLP):
    def __init__(self, *args, **kwargs):
        self.input_dim = kwargs.pop("input_dim")
        self.action_dim = kwargs.pop("action_dim", 0)
        concat_mask = kwargs.pop("concat_mask", False)
        kwargs['input_dim'] = self.input_dim + (self.action_dim if concat_mask else 0)
        super().__init__(*args, **kwargs)

    def forward(
        self,
        obs: Batch,
        state: RecurrentStateBatch | None = None,
        info: dict[str, Any] | None = None,
        detach_state: bool = True,
    ) -> (torch.Tensor, None):
        """Mapping: obs -> flatten (inside MLP)-> logits.

        :param obs:
        :param state: unused and returned as is
        :param info: unused
        """

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x = self.model(obs)

        return x, state


class RecurrentGRU(NetBase[RecurrentStateBatch]):
    """Tianshou-compatible GRU network (same API as common.Recurrent)."""

    def __init__(
            self,
            layer_num: int,
            state_shape: int | Sequence[int],
            action_shape: int | Sequence[int],
            concat_mask: bool = False,
            hidden_layer_size: int = 128,
            device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = state_shape + (action_shape if concat_mask else 0)
        self.hidden_dim = hidden_layer_size
        self.env_num = 1
        self.flag = False

        self.fc1 = nn.Linear(int(np.prod(self.obs_dim)), hidden_layer_size)
        self.gru = nn.GRU(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        # self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))
        self.output_dim = hidden_layer_size

    def forward(  # pylint: disable=arguments-differ
            self,
            obs: Batch,
            state: RecurrentStateBatch | None = None,
            info: dict[str, Any] | Batch | None = None,
            *,
            detach_state: bool = True,
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:

        dones = ~info.Alive
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones).to(self.device)

        # feed-forward
        x = self.fc1(obs)  # [B, H] or [B, T, H]

        if state is None or "hidden" not in state:
            y, h = self.gru(x)  # hidden output: [T, B, H]
        else:
            # input to gru should be [B, T, H]
            h_in = (
                state["hidden"].transpose(0, 1).contiguous()
                if state["hidden"].ndim == 3
                else state["hidden"].contiguous()
            )

            h_in = (
                torch.logical_not(dones).view(1, -1, 1) * h_in
                if h_in.ndim == 3
                else torch.logical_not(dones).view(-1, 1) * h_in
            )

            y, h = self.gru(
                x,
                h_in
            )  # for eval h: [B, H], for train: [T, B, H]

        h = h.transpose(0, 1) if h.ndim == 3 else h

        if detach_state:
            h = h.detach()

        next_hidden = cast(
            RecurrentStateBatch,
            Batch(
                {
                    "hidden": h,
                }
            ),
        )

        return y, next_hidden


class RecurrentLSTM(NetBase[RecurrentStateBatch]):
    def __init__(
            self,
            layer_num: int,
            state_shape: int | Sequence[int],
            action_shape: int | Sequence[int],
            concat_mask: bool = False,
            hidden_layer_size: int = 128,
            device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = state_shape + (action_shape if concat_mask else 0)
        self.hidden_dim = hidden_layer_size
        self.env_num = 1
        self.flag = False

        self.fc1 = nn.Linear(int(np.prod(self.obs_dim)), hidden_layer_size)
        self.lstm = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        # self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))
        self.output_dim = hidden_layer_size

    def forward(  # pylint: disable=arguments-differ
            self,
            obs: Batch,
            state: RecurrentStateBatch | None = None,
            info: dict[str, Any] | Batch | None = None,
            *,
            detach_state: bool = True,
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:

        dones = ~info.Alive
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones).to(self.device)

        # feed-forward + add time dim
        x = self.fc1(obs)  # [B, H] or [B, T, H]

        if state is None or "hidden" not in state:
            y, (h, c) = self.lstm(x)  # hidden output: [T, B, H]
        else:
            # input to lstm should be [B, T, H]
            h_in = (
                state["hidden"].transpose(0, 1).contiguous()
                if state["hidden"].ndim == 3
                else state["hidden"].contiguous(),
                state["cell"].transpose(0, 1).contiguous()
                if state["cell"].ndim == 3
                else state["cell"].contiguous(),
            )

            h_in = (
                torch.logical_not(dones).view(1, -1, 1) * h_in[0]
                if h_in[0].ndim == 3
                else torch.logical_not(dones).view(-1, 1) * h_in[0],
                torch.logical_not(dones).view(1, -1, 1) * h_in[1]
                if h_in[1].ndim == 3
                else torch.logical_not(dones).view(-1, 1) * h_in[1]
            )

            y, (h, c) = self.lstm(
                x,
                h_in
            )  # for eval h: [B, H], for train: [T, B, H]

        h = h.transpose(0, 1) if h.ndim == 3 else h
        c = c.transpose(0, 1) if c.ndim == 3 else c

        if detach_state:
            h = h.detach()
            c = c.detach()

        next_hidden = cast(
            RecurrentStateBatch,
            Batch(
                {
                    "hidden": h,
                    "cell": c,
                }
            ),
        )

        return y, next_hidden


class MaskedActor(Actor):
    """Actor that zeroes logits for illegal actions via provided mask."""

    def __init__(
            self,
            preprocess_net,
            action_dim,
            concat_mask=False,
            last_hidden_dim=None,
            use_film: bool = True,
            device='cpu'
    ):
        super().__init__(preprocess_net=preprocess_net, action_shape=action_dim,
                         softmax_output=False, device=device)  # remember for logits
        self.feature_dim = getattr(self.preprocess, "input_dim", getattr(self.preprocess, "state_shape", None))
        # self.feature_dim = self.feature_dim - action_dim if self.feature_dim is not None else None
        self.concat_mask = concat_mask
        self.last.flatten_input = False

        self.n_bins = 5
        self.edges = torch.linspace(0, 1, self.n_bins + 1, device=self.device)[1:-1]
        self.film = use_film
        if self.film :
            self.build_cond = BudgetCond(self.n_bins, 1)
            self.last = FiLMHead(
                in_dim=last_hidden_dim,
                hidden_dim=32,
                out_dim=action_dim,
                cond_dim=self.build_cond.out_dim,
            )

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None, info: dict | Batch = None):

        # grab vector
        obs.obs = obs.obs.astype(np.float32) if isinstance(obs.obs, np.ndarray) else obs.obs

        if isinstance(obs, Batch):
            x_in = obs.obs  # or dict(obs)   (no copy of scalars)

        # input -> [B, T, H] for training, [B, H] for eval
        if not torch.is_tensor(x_in):
            x_in = torch.from_numpy(x_in).to(self.device)

        if x_in.ndim == 1:  # single env
            x_in = x_in.unsqueeze(0)  # for eval, dim: [B, H]
        # if x_in.ndim == 2:
        #     x_in = x_in.unsqueeze(-2)  # for training, dim: [B, T, H]

        if isinstance(obs, Batch) and "mask" in obs and self.concat_mask:
            mask_t = torch.as_tensor(obs.mask, device=self.device, dtype=torch.float32)
            # Make shapes match: [B,T,A] or [B,A] → add time dim if needed
            if mask_t.ndim == 1 and x_in.ndim > 1:
                mask_t = mask_t.unsqueeze(0)
                if x_in.ndim == 3:
                    mask_t = mask_t.unsqueeze(-2)
            if x_in.ndim == 3 and mask_t.ndim == 2:  # [B,H]
                mask_t = mask_t.unsqueeze(-2) # → [B,1,A]
            # Broadcast along T if needed, then concat on feature dim
            x_in = torch.cat([x_in, mask_t], dim=-1)

        detach_state = True
        if hasattr(obs, "detach_state"):
            detach_state = False

        # preprocess obs (with GRU or anything else)
        features, h = self.preprocess(x_in, state, info, detach_state=detach_state)

        # FiLM section
        if self.film:
            rem_budget = torch.from_numpy(info.BudgetLeft).to(self.device)
            tot_budget = torch.from_numpy(info.BudgetTotal).to(self.device)

            # compute bins
            fraction = (rem_budget / (tot_budget.clamp_min(1e-8))).clamp(0, 1.0)
            budget_bin = torch.bucketize(fraction, self.edges)

            # gamma, beta = self.film(rem_budget, tot_budget, budget_bin)
            # features = (1.0 + gamma) * features + beta
            cond = self.build_cond(rem_budget, tot_budget, budget_bin)
            logits = self.last(features, cond=cond)
        else:
            # generate logits from mlp
            logits = self.last(features)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        # mask
        if isinstance(obs, Batch) and "mask" in obs:
            mask = torch.as_tensor(obs["mask"], device=logits.device).bool()
            # logits = logits.clone()
            # mask = mask.expand_as(logits)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 2 and mask.ndim != logits.ndim:
                mask = mask.unsqueeze(-2)

            if mask.shape[0] == 1 and logits.shape[0] > 1:
                mask = mask.expand(logits.shape[0], -1)  # broadcast along batch

            # logits[mask == False] = -1e10
            # Fill invalid actions with a large negative but finite number
            logits = logits.masked_fill(~mask, -1e10)

        return logits, h


class StackedCritic(Critic):
    def __init__(
            self,
            preprocess_net,
            concat_mask=False,
            last_hidden_dim=None,
            use_film=True,
            device='cpu',
            **kwargs
    ):
        super().__init__(
            preprocess_net=preprocess_net,
            device=device,
            **kwargs
        )
        self.concat_mask = concat_mask

        self.n_bins = 5
        self.edges = torch.linspace(0, 1, self.n_bins + 1, device=self.device)[1:-1]
        self.film = use_film
        if self.film:
            self.build_cond = BudgetCond(self.n_bins, 1)
            self.last = FiLMHead(
                in_dim=last_hidden_dim,
                hidden_dim=32,
                out_dim=1,
                cond_dim=self.build_cond.out_dim,
            )

    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Mapping: s_B -> V(s)_B."""
        # TODO: don't use this mechanism for passing state
        info = kwargs.get("info")

        if isinstance(obs, Batch):
            x_in = obs.obs

        if not torch.is_tensor(x_in):
            x_in = torch.from_numpy(x_in).to(self.device)

        if isinstance(obs, Batch) and "mask" in obs and self.concat_mask:
            mask_t = torch.as_tensor(obs.mask, device=self.device, dtype=torch.float32)
            # Make shapes match: [B,T,A] or [B,A] → add time dim if needed
            if mask_t.ndim == 1 and x_in.ndim > 1:
                mask_t = mask_t.unsqueeze(0)
                if x_in.ndim == 3:
                    mask_t = mask_t.unsqueeze(-2)
            if x_in.ndim == 3 and mask_t.ndim == 2:  # [B,H]
                mask_t = mask_t.unsqueeze(-2) # → [B,1,A]
            # Broadcast along T if needed, then concat on feature dim
            x_in = torch.cat([x_in, mask_t], dim=-1)

        y, _ = self.preprocess(x_in, state=kwargs.get("state", None))

        # FiLM section
        if self.film and info is not None:
            rem_budget = torch.from_numpy(info.BudgetLeft).to(self.device)
            tot_budget = torch.from_numpy(info.BudgetTotal).to(self.device)

            # compute bins
            fraction = (rem_budget / (tot_budget.clamp_min(1e-8))).clamp(0, 1.0)
            budget_bin = torch.bucketize(fraction, self.edges)

            # gamma, beta = self.film(rem_budget, tot_budget, budget_bin)
            # features = (1.0 + gamma) * features + beta
            cond = self.build_cond(rem_budget, tot_budget, budget_bin)
            return self.last(y, cond=cond)
        else:
            # generate logits from mlp
            return self.last(y)


class ConstraintCritic(Critic):
    def __init__(
            self,
            preprocess_net,
            constraint_indices,
            last_hidden_dim=None,
            use_film=True,
            concat_mask=False,
            device='cpu'
    ):
        super().__init__(
            preprocess_net=preprocess_net,
        )
        self.constraint_indices = torch.as_tensor(constraint_indices, device=device)
        self.concat_mask = concat_mask

        self.n_bins = 5
        self.edges = torch.linspace(0, 1, self.n_bins + 1, device=self.device)[1:-1]
        self.film = use_film
        if self.film:
            self.build_cond = BudgetCond(self.n_bins, 1)
            self.last = FiLMHead(
                in_dim=last_hidden_dim,
                hidden_dim=32,
                out_dim=1,
                cond_dim=self.build_cond.out_dim,
            )

    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:

        if isinstance(obs, Batch):
            x_in = obs.obs

        if not torch.is_tensor(x_in):
            x_in = torch.from_numpy(x_in).to(self.device)

        # add checks?
        if x_in.ndim == 1:
            x_in = x_in[self.constraint_indices]
        elif x_in.ndim == 2:
            x_in = x_in[:, self.constraint_indices]
        elif x_in.ndim == 3:
            x_in = x_in[:, :, self.constraint_indices]

        if isinstance(obs, Batch) and "mask" in obs and self.concat_mask:
            mask_t = torch.as_tensor(obs.mask, device=self.device, dtype=torch.float32)
            # Make shapes match: [B,T,A] or [B,A] → add time dim if needed
            if mask_t.ndim == 1 and x_in.ndim > 1:
                mask_t = mask_t.unsqueeze(0)
                if x_in.ndim == 3:
                    mask_t = mask_t.unsqueeze(-2)
            if x_in.ndim == 3 and mask_t.ndim == 2:  # [B,H]
                mask_t = mask_t.unsqueeze(-2) # → [B,1,A]
            # Broadcast along T if needed, then concat on feature dim
            x_in = torch.cat([x_in, mask_t], dim=-1)

        y, _ = self.preprocess(x_in, state=kwargs.get("state", None))

        return self.last(y)


class UCBNetwork(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


