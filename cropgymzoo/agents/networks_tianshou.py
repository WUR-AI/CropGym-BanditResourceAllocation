from typing import Sequence, Any, cast

from copy import deepcopy

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.data.types import RecurrentStateBatch
from tianshou.utils.net.common import NetBase, Recurrent, Net
from tianshou.utils.net.discrete import Actor, Critic
from torch import nn


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
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:

        if isinstance(obs, Batch):
            obs = obs.obs  # or dict(obs)   (no copy of scalars)

        # input -> [B, T, H] for training, [B, H] for eval
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).to(self.device)

        if obs.ndim == 1:  # single env
            obs = obs.unsqueeze(0)  # for eval, dim: [B, H]
        elif obs.ndim == 2:
            obs = obs.unsqueeze(-2)  # for training, dim: [B, T, H]

        # 2. feed-forward + add time dim
        x = self.fc1(obs) # [B, H] or [B, T, H]

        if state is None or "hidden" not in state:
            y, h_in = self.gru(x)            # hidden output: [T, B, H]
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
        logits = self.fc2(y)  # [B_alive, A]

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

        return logits, next_hidden

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

        obs_dim = self.preprocess.obs_dim[0]
        out_dim = self.preprocess.output_dim

        initial_shape = obs.shape[0]
        if initial_shape > self.max_env:
            self.max_env = initial_shape

        # if state:
        #     if obs.shape[0] != state.shape[0] and len(obs.obs.shape) != 1:
        #         dim_to_align = state.shape[0]
        #         idx = torch.zeros(dim_to_align, dtype=torch.bool)
        #
        #         try:  # quite hacky here hmmmm is there a better way to do this?
        #             idx[info['env_id']] = True
        #         except IndexError:
        #             if set(self.previous_env_ids) != set(info['env_id']):
        #                 odd_out = [b for b in self.previous_env_ids if b not in info['env_id']]
        #                 idx = [b not in odd_out for b in self.previous_env_ids]
        #
        #         # Don't change in-place for back propagation
        #         obs = deepcopy(obs)
        #
        #         # Preallocate zeros with matching dtypes
        #         zero_obs = np.zeros((dim_to_align, obs_dim), dtype=obs.obs.dtype)
        #         zero_mask = np.zeros((dim_to_align, out_dim), dtype=obs.mask.dtype)
        #
        #         # Fill only alive slots
        #         zero_obs[idx] = obs.obs
        #         zero_mask[idx] = obs.mask
        #
        #         obs.obs = zero_obs
        #         obs.mask = zero_mask
        #         obs.agent_id = np.array([obs.agent_id[-1]] * dim_to_align, dtype=object)

        obs.obs = obs.obs.astype(np.float32)
        logits, h = self.preprocess(obs, state, info)
        # logits = self.last(logits)
        if isinstance(obs, Batch) and "mask" in obs:
            mask = torch.as_tensor(obs["mask"], device=logits.device)
            logits = logits.clone()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 2:
                mask = mask.unsqueeze(-2)
            elif mask.ndim == 3:
                mask = mask.transpose(0, 1)

            logits[mask == False] = -1e10

            if logits.ndim == 3:
                logits = logits.squeeze(1)


        if hasattr(info, 'env_id') and initial_shape != logits.shape[0]:
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
        self.obs_dim = (kwargs.get("state_shape", None),)

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
