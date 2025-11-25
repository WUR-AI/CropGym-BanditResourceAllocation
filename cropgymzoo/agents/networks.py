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
    def __init__(self, in_dim, hidden_dim, out_dim, cond_dim, device="mps"):
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, device=device)
        self.gam1 = nn.Linear(cond_dim, hidden_dim, device=device)
        self.bet1 = nn.Linear(cond_dim, hidden_dim, device=device)
        self.act  = nn.ReLU()
        self.out  = nn.Linear(hidden_dim, out_dim, device=device)
        # identity init for (1+gamma)*h + beta at start
        nn.init.zeros_(self.gam1.weight)
        nn.init.zeros_(self.gam1.bias)
        nn.init.zeros_(self.bet1.weight)
        nn.init.zeros_(self.bet1.bias)

    def forward(self, x, cond):
        h = self.lin1(x)                      # [B,T,H]
        gamma = self.gam1(cond)
        beta = self.bet1(cond)
        if gamma.ndim == 2:
            gamma = gamma.unsqueeze(1)
        if beta.ndim == 2:
            beta = beta.unsqueeze(1)
        h = (1 + gamma) * h + beta   # FiLM on hidden pre-activation
        h = self.act(h)
        return self.out(h)                    # logits


class BudgetCond(nn.Module):
    """
    Creates a conditioning vector from budget information using a proper
    embedding layer for the discretized budget bin.
    """

    def __init__(self, n_bins, emb_dim=8, device="mps"):
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        super().__init__()
        # Use a real embedding layer for the categorical bin information
        # n_bins + 1 because bucketize can output values from 0 to n_bins inclusive
        self.bin_embedding = nn.Embedding(n_bins + 1, emb_dim, device=device)

        # The output dimension will be the embedding dim + 1 continuous feature
        self.out_dim = emb_dim + 1

    def forward(self, rem, tot, budget_bin):
        # 1. Calculate the continuous budget fraction
        rem_frac = (rem / tot.clamp_min(1e-8)).clamp(0, 1.0).unsqueeze(-1)  # Shape: [B, 1] or [B, T, 1]

        # 2. Get the learned embedding for the budget bin
        # .long() is required for the embedding layer lookup
        bin_emb = self.bin_embedding(budget_bin.long())  # Shape: [B, emb_dim] or [B, T, emb_dim]

        # 3. Concatenate the continuous feature and the learned categorical feature
        # This provides a much richer signal to the FiLM layers
        cond = torch.cat([rem_frac, bin_emb], dim=-1)

        return cond.float()

class ObsMLP(MLP):
    def __init__(self, *args, **kwargs):
        self.input_dim = kwargs.pop("input_dim")
        self.action_dim = kwargs.pop("action_dim", 0)
        concat_mask = kwargs.pop("concat_mask", False)
        self.n_weather_vars = kwargs.pop("n_weather_vars", 4)
        self.n_days = kwargs.pop("n_days", 7)
        self.pool = kwargs.pop("pool", False)
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

        # --- Weather pooling ---
        if self.pool:
            n_weather = self.n_weather_vars * self.n_days
            weather = obs[..., -n_weather:]  # last n_weather_vars * n_days elements
            weather = weather.view(*weather.shape[:-1], self.n_days, self.n_weather_vars)
            weather_avg = weather.mean(dim=-2)  # average over days → shape [..., 4]

            obs_core = obs[..., :-n_weather]  # everything before weather
            obs = torch.cat([obs_core, weather_avg], dim=-1)  # concat averaged weather back

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
            n_weather_vars: int = 4,
            n_days: int = 7,
            pool: bool = False,
            device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = state_shape + (action_shape if concat_mask else 0)
        self.hidden_dim = hidden_layer_size
        self.env_num = 1
        self.flag = False

        # Optional average pooling over the last (n_days * n_weather_vars) obs features
        self.n_weather_vars = n_weather_vars
        self.n_days = n_days
        self.pool = pool
        if self.pool:
            # Pool across the day dimension (length = n_days)
            self.avgpool = nn.AvgPool1d(kernel_size=self.n_days, stride=self.n_days)

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

        # If Alive/dones is [B,T], only the first timestep should mask the initial state
        if isinstance(dones, torch.Tensor) and dones.ndim == 2:
            dones = dones[:, 0]

        # Optional weather pooling on the last n_days * n_weather_vars features
        if self.pool:
            n_weather = self.n_weather_vars * self.n_days
            if obs.shape[-1] >= n_weather:
                # Split core features and weather tail
                obs_core = obs[..., :-n_weather]
                weather = obs[..., -n_weather:]

                # Reshape to [..., n_days, n_weather_vars] then channels-first [..., n_weather_vars, n_days]
                weather = weather.view(*weather.shape[:-1], self.n_days, self.n_weather_vars)
                weather = weather.transpose(-2, -1)

                # Collapse leading dims to N for pooling, pool over days → [N, n_weather_vars, 1]
                flat = weather.reshape(-1, self.n_weather_vars, self.n_days)
                pooled = self.avgpool(flat)
                weather_avg = pooled.squeeze(-1)

                # Restore leading dims and concatenate back
                weather_avg = weather_avg.view(*weather.shape[:-2], self.n_weather_vars)
                obs = torch.cat([obs_core, weather_avg], dim=-1).to(device=self.device)

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
            n_weather_vars: int = 4,
            n_days: int = 7,
            pool: bool = False,
            device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = state_shape + (action_shape if concat_mask else 0)
        self.hidden_dim = hidden_layer_size
        self.env_num = 1
        self.flag = False

        # Optional average pooling over the last (n_days * n_weather_vars) obs features
        self.n_weather_vars = n_weather_vars
        self.n_days = n_days
        self.pool = pool
        if self.pool:
            # Pool across the day dimension (length = n_days)
            self.avgpool = nn.AvgPool1d(kernel_size=self.n_days, stride=self.n_days)


        self.fc1 = nn.Linear(int(np.prod(self.obs_dim)), hidden_layer_size, device=self.device)
        self.lstm = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
            device=self.device
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

        # If Alive/dones is [B,T], only the first timestep should mask the initial state
        if isinstance(dones, torch.Tensor) and dones.ndim == 2:
            dones = dones[:, 0]

        # Optional weather pooling on the last n_days * n_weather_vars features
        if self.pool:
            n_weather = self.n_weather_vars * self.n_days
            if obs.shape[-1] >= n_weather:
                # Split core features and weather tail
                obs_core = obs[..., :-n_weather]
                weather = obs[..., -n_weather:]

                # Reshape to [..., n_days, n_weather_vars] then channels-first [..., n_weather_vars, n_days]
                weather = weather.view(*weather.shape[:-1], self.n_days, self.n_weather_vars)
                weather = weather.transpose(-2, -1)

                # Collapse leading dims to N for pooling, pool over days → [N, n_weather_vars, 1]
                flat = weather.reshape(-1, self.n_weather_vars, self.n_days)
                pooled = self.avgpool(flat)
                weather_avg = pooled.squeeze(-1)

                # Restore leading dims and concatenate back
                weather_avg = weather_avg.view(*weather.shape[:-2], self.n_weather_vars)
                obs = torch.cat([obs_core, weather_avg], dim=-1).to(device=self.device)

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
            prefer_noop=True,  # enable prior
            noop_prior_p=0.6,  # ~90% prior on action 0
            idle_penalty=False,
            device='mps'
    ):
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        super().__init__(preprocess_net=preprocess_net, action_shape=action_dim,
                         softmax_output=False, device=device)  # remember for logits
        self.feature_dim = getattr(self.preprocess, "input_dim", getattr(self.preprocess, "state_shape", None))
        # self.feature_dim = self.feature_dim - action_dim if self.feature_dim is not None else None
        self.concat_mask = concat_mask
        self.last.flatten_input = False
        self.prefer_noop = prefer_noop
        self.noop_prior_p = float(noop_prior_p)
        self.idle_penalty = idle_penalty
        self.idle_penalty_value = 1.0

        self.n_bins = 5
        self.edges = torch.linspace(0, 1, self.n_bins + 1, device=self.device)[1:-1]
        self.film = use_film
        if self.film:
            self.build_cond = BudgetCond(self.n_bins, 1).to(device)
            self.last = FiLMHead(
                in_dim=last_hidden_dim,
                hidden_dim=32,
                out_dim=action_dim,
                cond_dim=self.build_cond.out_dim,
            ).to(device)
        self.last.to(device)
        if self.prefer_noop:
            self._init_noop_skew(action_dim, p0=self.noop_prior_p)


    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None, info: dict | Batch = None):

        # grab vector
        obs.obs = obs.obs.astype(np.float32) if isinstance(obs.obs, np.ndarray) else obs.obs

        if isinstance(obs, Batch):
            x_in = obs.obs  # or dict(obs)   (no copy of scalars)

        # input -> [B, T, H] for training, [B, H] for eval
        if not torch.is_tensor(x_in):
            x_in = torch.from_numpy(x_in).to(self.device)

        if x_in.ndim == 1:  # single env
            x_in = x_in.unsqueeze(0).unsqueeze(0)  # for eval, dim: [B, H]
        elif x_in.ndim == 2:
            x_in = x_in.unsqueeze(-2)  # for training, dim: [B, T, H]

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
            rem_budget = self._to_device_f32(info.BudgetLeft, self.device)
            tot_budget = self._to_device_f32(info.BudgetTotal, self.device)

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

        if self.idle_penalty:
            # push *down* all non-zero actions
            logits[..., 1:] = logits[..., 1:] - self.idle_penalty_value

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

    def _init_noop_skew(self, action_dim: int, p0: float):
        """
        One-time init: set output-layer bias so softmax prefers action 0 with prob ~p0
        (assuming other actions start at 0 logit).
        """
        # clamp p0 to (1/A, 1) to avoid degenerate math
        A = float(action_dim)
        p0 = max(min(p0, 1.0 - 1e-6), 1.0 / A + 1e-6)

        # required offset vs others=0: b0 = log(p0/(1-p0)) + log(A-1)
        b0 = (np.log(p0 / (1.0 - p0)) + np.log(A - 1.0))

        # Find the final nn.Linear that outputs action_dim logits
        lin = None
        for m in self.last.modules():
            if isinstance(m, nn.Linear) and m.out_features == action_dim:
                lin = m
        if lin is None and isinstance(self.last, nn.Linear) and self.last.out_features == action_dim:
            lin = self.last
        if lin is None or lin.bias is None:
            raise RuntimeError("Could not locate final logits bias to set noop skew.")

        with torch.no_grad():
            lin.bias.zero_()
            lin.bias[0] = b0

    @staticmethod
    def _to_device_f32(x, device):
        """Convert numpy / tensor / scalar to torch.float32 on given device."""
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device, dtype=torch.float32)
        else:
            return torch.tensor(x, device=device, dtype=torch.float32)


class StackedCritic(Critic):
    def __init__(
            self,
            preprocess_net,
            concat_mask=False,
            last_hidden_dim=None,
            use_film=True,
            device='mps',
            **kwargs
    ):
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
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
            self.build_cond = BudgetCond(self.n_bins, 1).to(device)
            self.last = FiLMHead(
                in_dim=last_hidden_dim,
                hidden_dim=32,
                out_dim=1,
                cond_dim=self.build_cond.out_dim,
            ).to(device)

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

        if y.ndim == 2:
            y = y.unsqueeze(1)

        # FiLM section
        if self.film and info is not None:
            if isinstance(info.BudgetLeft, np.ndarray):
                rem_budget = torch.from_numpy(info.BudgetLeft).to(device=self.device, dtype=torch.float32)
            else:
                rem_budget = info.BudgetLeft.to(device=self.device, dtype=torch.float32)
            if isinstance(info.BudgetTotal, np.ndarray):
                tot_budget = torch.from_numpy(info.BudgetTotal).to(device=self.device, dtype=torch.float32)
            else:
                tot_budget = info.BudgetTotal.to(device=self.device, dtype=torch.float32)

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


