from typing import Sequence, Any, cast

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tianshou.data import Batch, to_torch
from tianshou.data.types import RecurrentStateBatch
from tianshou.utils.net.common import NetBase, Recurrent, Net
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
        self.gru.flatten_parameters()
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
        logits = self.fc2(y[:,-1] if y.ndim == 3 else y)  # [B_alive, A]

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

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None, info: dict | Batch = None):

        # grab vector
        obs.obs = obs.obs.astype(np.float32)

        # preprocess obs (with GRU or anything else)
        x, h = self.preprocess(obs, state, info)

        # generate logits from mlp
        logits = self.last(x)

        # mask
        if isinstance(obs, Batch) and "mask" in obs:
            mask = torch.as_tensor(obs["mask"], device=logits.device).bool()
            # logits = logits.clone()
            mask = mask.expand_as(logits)
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
