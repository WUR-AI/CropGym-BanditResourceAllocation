from typing import Sequence, Any

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.data.types import RecurrentStateBatch
from tianshou.utils.net.common import NetBase, Recurrent
from tianshou.utils.net.discrete import Actor, Critic
from torch import nn as nn


class GRUBackbone(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int | Sequence[int] = 128, activation = nn.Tanh):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)
            if len(hidden_dims) == 0:
                raise ValueError("`hidden_dim` sequence must contain at least one element.")

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), activation()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        # 3️⃣ GRU whose input_size **and** hidden_size = last MLP width
        last_dim = hidden_dims[-1]
        self.gru = nn.GRU(input_size=last_dim,
                          hidden_size=last_dim,
                          batch_first=True)

        self._hidden_dim = last_dim  # handy for downstream code

    def forward(self, obs: torch.Tensor,
                state: torch.Tensor | None = None):
        """
        obs   : (batch, obs_dim)
        state : (1, batch, hidden) for a single-layer GRU (or None)
        """
        x = self.mlp(obs)  # (batch, last_dim)
        x = x.unsqueeze(1)  # add time dimension → (batch, 1, last_dim)

        y, h = self.gru(x, state)  # y: (batch, 1, last_dim)
        y = y.squeeze(1)  # remove the time dim  → (batch, last_dim)
        return y, h


class RecurrentGRU(NetBase[RecurrentStateBatch]):
    """Tianshou-compatible GRU network (same API as common.Recurrent)."""

    def __init__(
        self,
        layer_num: int,
        state_shape: int | Sequence[int],
        action_shape: int | Sequence[int],
        hidden_layer_size: int = 128,
        device: str | int | torch.device = "cpu",
        key_order: tuple[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.key_order = key_order
        self.output_dim = int(np.prod(action_shape))

        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.gru = nn.GRU(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))

    def forward(                      # pylint: disable=arguments-differ
        self,
        obs: Batch,
        state: RecurrentStateBatch | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Batch]:

        if isinstance(obs, Batch):
            obs = obs.obs  # or dict(obs)   (no copy of scalars)

        # input -> [bsz, len, dim] for training, [bsz, dim] for eval
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).to(self.device)

        if obs.ndim == 1:  # single env
            obs = obs.unsqueeze(0)  # [1, D]

        # 2. feed-forward + add time dim
        x = torch.tanh(self.fc1(obs)) # [B, H]
        x = x.unsqueeze(1)  # [B, 1, H]
        self.gru.flatten_parameters()

        if state is None or "hidden" not in state:
            y, h_in = self.gru(x)            # hidden: [num_layers, bsz, h]
        else:
            h_in = state["hidden"].transpose(0, 1).contiguous()
        y, hidden = self.gru(x, h_in)

        logits = self.fc2(y.squeeze(1))              # take last time-step

        next_state = Batch({"hidden": hidden.transpose(0, 1).detach()})
        return logits, next_state


class RecurrentLSTM(Recurrent):
    def __init__(
            self,
            layer_num: int,
            state_shape: int | Sequence[int],
            action_shape,
            device: str | int | torch.device = "cpu",
            hidden_layer_size: int = 128,
    ) -> None:
        super().__init__(
            layer_num=layer_num,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            hidden_layer_size=hidden_layer_size,
        )
        self.output_dim = int(np.prod(action_shape))

    def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            state: RecurrentStateBatch | None = None,
            info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:

        if isinstance(obs, Batch):
            obs = obs.obs  # or dict(obs)   (no copy of scalars)

        print(state)
        return super().forward(obs, state, info)


class MaskedActor(Actor):
    """Actor that zeroes logits for illegal actions via provided mask."""

    def __init__(self, preprocess_net, action_dim, device='cpu', key_order = None):
        super().__init__(preprocess_net=preprocess_net, action_shape=action_dim,
                         softmax_output=False, device=device)  # remember for logits
        self.key_order = key_order

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None, info: dict = {}):

        latent, h = self.preprocess(obs, state)
        logits = self.last(latent)
        if isinstance(info, dict) and "action_mask" in info:
            mask = torch.as_tensor(info["action_mask"], device=logits.device)
            logits[mask == 0] = -1e10
        return logits, h


class DictObsCritic(Critic):
    def __init__(self, preprocess_net, device='cpu', key_order = None):
        super().__init__(preprocess_net=preprocess_net)
        self.device = device
        self.key_order = key_order

    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:

        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)
