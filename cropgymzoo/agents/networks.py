from typing import Sequence, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
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


class RecurrentGRU(NetBase[RecurrentStateBatch]):
    """Tianshou-compatible GRU network (same API as common.Recurrent)."""

    def __init__(
        self,
        layer_num: int,
        state_shape: int | Sequence[int],
        action_shape: int | Sequence[int],
        hidden_layer_size: int = 128,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = state_shape + action_shape
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

    def forward(                      # pylint: disable=arguments-differ
        self,
        obs: Batch,
        state: RecurrentStateBatch | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:

        # feed-forward + add time dim
        x = self.fc1(obs) # [B, H] or [B, T, H]

        if state is None or "hidden" not in state:
            y, h = self.gru(x)            # hidden output: [T, B, H]
        else:
            # input to GRU should be [B, T, H]
            h_in = (
                state["hidden"].transpose(0, 1).contiguous()
                if state["hidden"].ndim == 3
                else state["hidden"].contiguous()
            )

            y, h = self.gru(
                x,
                h_in
            )  # for eval h: [B, H], for train: [T, B, H]
        # logits = self.fc2(y)  # [:,-1] if y.ndim == 3 else y)  # [B_alive, A]

        # return to [B, T, H] for storing
        next_hidden = cast(
            RecurrentStateBatch,
            Batch(
                {
                    "hidden": h.transpose(0, 1).detach()
                    if h.ndim == 3
                    else h.detach(),
                }
            )
        )

        return y, next_hidden


class RecurrentLSTM(NetBase[RecurrentStateBatch]):
    def __init__(
            self,
            layer_num: int,
            state_shape: int | Sequence[int],
            action_shape: int | Sequence[int],
            hidden_layer_size: int = 128,
            device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = state_shape + action_shape
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
            info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:

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

            y, (h, c) = self.lstm(
                x,
                h_in
            )  # for eval h: [B, H], for train: [T, B, H]

        next_hidden = cast(
            RecurrentStateBatch,
            Batch(
                {
                    "hidden": h.transpose(0, 1).detach() if h.ndim == 3 else h.detach(),
                    "cell": c.transpose(0, 1).detach() if c.ndim == 3 else c.detach(),
                }
            ),
        )

        return y, next_hidden


class MaskedActor(Actor):
    """Actor that zeroes logits for illegal actions via provided mask."""

    def __init__(self, preprocess_net, action_dim, device='cpu'):
        super().__init__(preprocess_net=preprocess_net, action_shape=action_dim,
                         softmax_output=False, device=device)  # remember for logits
        self.last.flatten_input = False

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
        elif x_in.ndim == 2:
            x_in = x_in.unsqueeze(-2)  # for training, dim: [B, T, H]

        if isinstance(obs, Batch) and "mask" in obs:
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

        # preprocess obs (with GRU or anything else)
        features, h = self.preprocess(x_in, state, info)

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
    def __init__(self, preprocess_net, device='cpu', **kwargs):
        super().__init__(
            preprocess_net=preprocess_net,
            device=device,
            **kwargs
        )

    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Mapping: s_B -> V(s)_B."""
        # TODO: don't use this mechanism for passing state

        if isinstance(obs, Batch):
            x_in = obs.obs

        if isinstance(obs, Batch) and "mask" in obs:
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


class ConstraintCritic(Critic):
    def __init__(self, preprocess_net, constraint_indices, device='cpu'):
        super().__init__(
            preprocess_net=preprocess_net,
        )
        self.constraint_indices = torch.as_tensor(constraint_indices, device=device)

    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:

        if isinstance(obs, Batch):
            obs = obs.obs

        # add checks?
        if obs.ndim == 1:
            obs = obs[self.constraint_indices]
        elif obs.ndim == 2:
            obs = obs[:, self.constraint_indices]
        elif obs.ndim == 3:
            obs = obs[:, :, self.constraint_indices]

        y, _ = self.preprocess(obs, state=kwargs.get("state", None))

        return self.last(y)


class ObsMLP(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        obs: Batch,
        state: RecurrentStateBatch | None = None,
        info: dict[str, Any] | None = None,
    ) -> (torch.Tensor, None):
        """Mapping: obs -> flatten (inside MLP)-> logits.

        :param obs:
        :param state: unused and returned as is
        :param info: unused
        """

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x = self.model(obs)

        return x, state


class UCBNetwork(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


