from typing import Sequence, Any

from copy import deepcopy

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.data.types import RecurrentStateBatch
from tianshou.utils.net.common import NetBase, Recurrent, Net
from tianshou.utils.net.discrete import Actor, Critic
from torch import nn


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
    ) -> None:
        super().__init__()
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        self.obs_dim = state_shape
        self.hidden_dim = hidden_layer_size
        self.env_num = 1
        self.flag = False


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

        if state is None or "hidden" not in state:
            y, h_in = self.gru(x)            # hidden: [num_layers, bsz, h]
        else:
            # state["hidden"] MUST be [B, H] coming from previous step
            h_in = state["hidden"].unsqueeze(0)  # .transpose(0, 1).contiguous()

        y, h = self.gru(x, h_in)
        logits = self.fc2(y.squeeze(1))  # [B_alive, A]

        next_hidden = h

        return logits, Batch(hidden=next_hidden.squeeze(0).detach())

    @staticmethod
    def is_consecutive_and_ordered(arr):
        arr = np.asarray(arr)  # ensure it's a NumPy array
        expected = np.arange(arr[0], arr[0] + len(arr))
        return np.array_equal(arr, expected)

    @staticmethod
    def to_batch_mask(raw_mask, B, idx=None, device=None):
        """Return a bool mask of length B.
        If raw_mask length == B -> pass through.
        If raw_mask length == idx.sum() -> scatter into B using idx (alive mask)."""
        if raw_mask is None:
            return torch.zeros(B, dtype=torch.bool, device=device)
        m = torch.as_tensor(raw_mask, device=device, dtype=torch.bool).flatten()
        if m.numel() == B:
            return m
        if idx is not None:
            idx = torch.as_tensor(idx, device=device, dtype=torch.bool).flatten()
            if m.numel() == int(idx.sum().item()):
                full = torch.zeros(B, dtype=torch.bool, device=device)
                full[idx] = m
                return full
        raise ValueError(f"Mask length {m.numel()} doesn’t match B={B} (and no valid idx).")


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

        return super().forward(obs, state, info)


class MaskedActor(Actor):
    """Actor that zeroes logits for illegal actions via provided mask."""

    def __init__(self, preprocess_net, action_dim, device='cpu'):
        super().__init__(preprocess_net=preprocess_net, action_shape=action_dim,
                         softmax_output=False, device=device)  # remember for logits
        self.previous_env_ids = None
        self.max_env = 1

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None, info: dict | Batch = None):

        env_num = self.preprocess.env_num
        obs_dim = self.preprocess.obs_dim[0]
        out_dim = self.preprocess.output_dim

        initial_shape = obs.shape[0]
        if initial_shape > self.max_env:
            self.max_env = initial_shape

        if state:
            if obs.shape[0] != state.shape[0]:
                dim_to_align = state.shape[0]
                idx = torch.zeros(dim_to_align, dtype=torch.bool)

                try:  # quite hacky here hmmmm is there a better way to do this?
                    idx[info['env_id']] = True
                except IndexError:
                    if set(self.previous_env_ids) != set(info['env_id']):
                        odd_out = [b for b in self.previous_env_ids if b not in info['env_id']]
                        idx = [b not in odd_out for b in self.previous_env_ids]

                # Don't change in-place for back propagation
                obs = deepcopy(obs)

                # Preallocate zeros with matching dtypes
                zero_obs = np.zeros((dim_to_align, obs_dim), dtype=obs.obs.dtype)
                zero_mask = np.zeros((dim_to_align, out_dim), dtype=obs.mask.dtype)

                # Fill only alive slots
                zero_obs[idx] = obs.obs
                zero_mask[idx] = obs.mask

                obs.obs = zero_obs
                obs.mask = zero_mask
                obs.agent_id = np.array([obs.agent_id[-1]] * dim_to_align, dtype=object)


        logits, h = self.preprocess(obs, state, info)
        # logits = self.last(logits)
        if isinstance(info, dict) and "action_mask" in info:
            mask = torch.as_tensor(info["action_mask"], device=logits.device)
            logits[mask == False] = -1e10
        elif isinstance(obs, Batch) and "mask" in obs:
            mask = torch.as_tensor(obs["mask"], device=logits.device)
            logits[mask == False] = -1e10

        if initial_shape != logits.shape[0]:
            logits = logits[info['env_id']]

        # to save for later
        if hasattr(info, "env_id"):
            self.previous_env_ids = info['env_id']

        return logits, h


class DictObsCritic(Critic):
    def __init__(self, preprocess_net, device='cpu', key_order = None):
        super().__init__(preprocess_net=preprocess_net)
        self.device = device
        self.key_order = key_order

    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:

        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)

class NetObs(Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits.

        :param obs:
        :param state: unused and returned as is
        :param info: unused
        """
        if isinstance(obs, Batch):
            obs = obs.obs

        return super().forward(obs, state, info)
