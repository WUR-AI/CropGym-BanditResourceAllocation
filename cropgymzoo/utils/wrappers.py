import copy
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
import pettingzoo
import torch
from tianshou.env import PettingZooEnv, VectorEnvNormObs, BaseVectorEnv, VectorEnvWrapper
from tianshou.utils import RunningMeanStd


def flatten_dict_space(
    dict_space: gym.spaces.Dict,
    key_order: tuple[str, ...] | None = None
) -> tuple[gym.spaces.Box, tuple[str, ...]]:
    """
    Turn a Dict(...) gym space into a single Box that stacks all leaves.

    Returns
    -------
    flat_space : Box
    key_order  : the deterministic key order used
    slices     : slices[i] tells you where key_order[i] lives in the flat vector
    """
    if key_order is None:
        # Gym's Dict preserves insertion order, but be explicit
        key_order = tuple(dict_space.spaces.keys())

    lows, highs, slices = [], [], []
    start = 0
    for k in key_order:
        space_k = dict_space[k]
        if not isinstance(space_k, gym.spaces.Box):
            raise TypeError(
                f"Only Box leaves supported, got {type(space_k)} for key '{k}'"
            )
        flat_low  = np.asarray(space_k.low , dtype=np.float32).reshape(-1)
        flat_high = np.asarray(space_k.high, dtype=np.float32).reshape(-1)
        lows.append(flat_low)
        highs.append(flat_high)

        end = start + flat_low.size
        slices.append(slice(start, end))
        start = end

    low  = np.concatenate(lows)
    high = np.concatenate(highs)

    flat_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    return flat_space, key_order


def flatten_dict_obs(
    obs_dict: dict[str, np.ndarray | float | int],
    key_order: tuple[str, ...],
) -> np.ndarray:
    """Flatten a single observation dict using the same key order."""

    first = np.asarray(obs_dict[key_order[0]], dtype=np.float32)
    n_envs = first.shape[0]

    flats = []

    for k in key_order:
        # v = obs_dict[k]
        # flats.append(np.asarray(v, dtype=np.float32).reshape(-1))
        v = np.asarray(obs_dict[k], dtype=np.float32)  # (n_envs, …)
        flats.append(v.reshape(n_envs, -1))  # keep env axis
    return np.concatenate(flats, axis=-1)


class MultiAgentDictObsFlatten:
    """Flatten nested dict observations for **every agent** and carry the mask.

    After wrapping, the env returns:
        obs  : Dict[str, 1‑D np.ndarray]
        info : Dict[str, Dict], where each dict has "action_mask": ndarray

    This matches what `MultiAgentPolicyManager` expects – each policy sees its
    own flattened vector plus `info[agent]["action_mask"]`.
    """

    def __init__(self, env: gym.Env | pettingzoo.AECEnv | PettingZooEnv):
        sample_obs, _ = env.reset(seed=0)
        self.agent_ids = env.unwrapped.possible_agents
        self.current_masks: dict[str, np.ndarray] = {k: None for k in self.agent_ids}  # type: ignore

        # Build obs space dict with flattened dimensions per agent
        spaces = {}
        for agent_id in self.agent_ids:
            spaces[agent_id] = sample_obs[agent_id]
        self.observation_spaces = gym.spaces.Dict(spaces)

    # ------------------------------------------------------------------
    @staticmethod
    def _dict_to_flat(d: dict) -> np.ndarray:
        parts: list[np.ndarray] = []
        for v in d.values():
            if isinstance(v, dict):
                parts.append(MultiAgentDictObsFlatten._dict_to_flat(v))
            else:
                parts.append(np.asarray(v, dtype=np.float32).flatten())
        return np.concatenate(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _split_and_flatten(obs: dict) -> np.ndarray:
        """Return (flat, mask) from one *agent*’s raw observation dict."""
        obs = obs.copy()  # avoid side‑effects
        parts: list[np.ndarray] = []
        for v in obs.values():
            if isinstance(v, dict):
                parts.append(MultiAgentDictObsFlatten._dict_to_flat(v))
            else:
                parts.append(np.asarray(v, dtype=np.float32).flatten())
        flat_obs = np.concatenate(parts, dtype=np.float32)
        return flat_obs

    # ------------------------------------------------------------------
    def _process_obs_dict(self, obs_dict: dict[str, dict]) -> dict[str, np.ndarray]:
        flat_dict = {}
        for agent, ob in obs_dict.items():
            flat, mask = self._split_and_flatten(ob)
            flat_dict[agent] = flat
            self.current_masks[agent] = mask
        return flat_dict

    # ------------------------------------------------------------------
    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        flat_obs = self._process_obs_dict(obs)
        info = info or {}
        for a in self.agent_ids:
            info.setdefault(a, {})["action_mask"] = self.current_masks[a]
        return flat_obs, info

    def step(self, action):  # type: ignore[override]
        obs, rew, term, trunc, info = self.env.step(action)
        flat_obs = self._process_obs_dict(obs)
        for a in self.agent_ids:
            info.setdefault(a, {})["action_mask"] = self.current_masks[a]
        return flat_obs, rew, term, trunc, info


class VecNormObs(VectorEnvNormObs):
    """
    Normalises a Dict observation by first flattening it (stable key order),
    then applying a single RunningMeanStd.
    """
    def __init__(self,
                 venv: BaseVectorEnv,
                 obs_dim: int | tuple[int] | None = None,
                 update_obs_rms: bool = True,
                 device="cpu"):
        super().__init__(venv, update_obs_rms)
        self.obs_dim = obs_dim
        self.device = device

    def collapse_info_dict(self, info: dict[str, dict[str, list | float]]) -> dict[str, dict[str, float]]:
        """
        Replace any list-valued items in the info dict with their last element.

        Args:
            info: Dict[agent_id, Dict[info_key, value]]

        Returns:
            A cleaned-up info dict with all values scalar (no growing lists).
        """
        collapsed = {}
        for k, v in info.items():
            if isinstance(v, list) and v:
                collapsed[k] = v[-1]
            elif isinstance(v, list) and not v:
                collapsed[k] = np.nan
            else:
                collapsed[k] = v
        return collapsed

    # ---------------- overrides ------------------------- #
    def reset(
        self,
        env_id: int | list[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs, info = self.venv.reset(env_id, **kwargs)

        if isinstance(obs, tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )

        obs_extracted = obs.copy()
        if isinstance(obs_extracted, (list, np.ndarray)):  # the common case
            obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs_extracted])

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_extracted)
        obs_extracted = self._norm_obs(obs_extracted)


        for i, venv_obs in enumerate(obs_extracted):
            obs[i]["obs"] = obs_extracted[i]

        for i, _i in enumerate(info):
            info[i] = self.collapse_info_dict(_i)

        return obs, info

    def step(
            self,
            action: np.ndarray | torch.Tensor | None,
            id: int | list[int] | np.ndarray | None = None,
    ):
        step_results = self.venv.step(action, id)

        obs = step_results[0].copy()
        if isinstance(obs, (list, np.ndarray)):  # the common case
            obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs])

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_extracted)
        obs_extracted = self._norm_obs(obs_extracted)

        for i, venv_obs in enumerate(obs_extracted):
            obs[i]["obs"] = obs_extracted[i]

        info = []
        for i, _i in enumerate(step_results[-1]):
            info.append(self.collapse_info_dict(_i))
        info = np.stack(info)

        return (obs, *step_results[1:-1], info)


def _new_rms():
    # from tianshou.utils.statistics import RunningMeanStd
    return RunningMeanStd()


class DictVectorEnvNormObs(VectorEnvWrapper):
    """
    Normalise every scalar in  ``obs_item["obs"]``  *independently*
    (zero mean, unit variance).

    Supports both batching styles a vector-env can emit:

    ❶  **list of per-env dicts**              ← what you actually get
        `[{"obs": {...}, "mask": ...},                 # env 0
          {"obs": {...}, "mask": ...}]                 # env 1 …]`

    ❷  **dict of arrays** (rare, but still valid)
        `{"obs": {"TEMP": np.ndarray[n_envs], …}, "mask": …}`
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(venv)
        self.update_obs_rms = update_obs_rms
        self.eps = eps
        # one RunningMeanStd per scalar key (mean / std are 0-D arrays)
        self.obs_rms: defaultdict[str, RunningMeanStd] = defaultdict(_new_rms)

    # ------------------------------------------------------------------ #
    # ── helpers -------------------------------------------------------- #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scalar_dicts_from(
        obs_container: Any,
    ) -> list[dict[str, float]]:
        """
        Return a **list** whose i-th element is the scalar dict
        of env *i*, *AND* keep references so modifications happen in place.
        """
        if isinstance(obs_container, (list, np.ndarray)):           # the common case
            return [d["observation"] if "observation" in d else d["obs"] for d in obs_container]

        if isinstance(obs_container, tuple):          # rarely tuples
            return [d["observation"] if "observation" in d else d["obs"] for d in list(obs_container)]

        raise TypeError(
            f"Unsupported obs container type: {type(obs_container)}"
        )

    # ------------------------------------------------------------------ #
    def _normalise_in_place(self, obs_container: Any) -> None:
        """
        Update running statistics and overwrite the original scalars
        with their normalised values.
        """
        per_env_dicts = self._scalar_dicts_from(obs_container)

        # 1. stack   key → np.ndarray[n_env]
        stacked: dict[str, np.ndarray] = {}
        for k in per_env_dicts[0]:
            stacked[k] = np.asarray(
                [d[k] for d in per_env_dicts], dtype=np.float32
            )
        print(stacked)
        print(self.obs_rms)
        # 2. update rms & normalise
        for k, vec in stacked.items():
            rms = self.obs_rms[k]
            if self.update_obs_rms:
                rms.update(vec)
            stacked[k] = (vec - rms.mean) / (rms.var + self.eps)

        # 3. write back to the original objects
        for i, d in enumerate(per_env_dicts):
            for k in d:
                d[k] = stacked[k][i]

    # ------------------------------------------------------------------ #
    # ── VectorEnv API -------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def reset(self, env_id=None, **kwargs):
        obs, info = self.venv.reset(env_id, **kwargs)
        self._normalise_in_place(obs)
        return obs, info

    def step(self, action, id=None):
        obs, rew, term, trunc, info = self.venv.step(action, id)
        self._normalise_in_place(obs)
        return obs, rew, term, trunc, info

    # ------------------------------------------------------------------ #
    # ── share statistics between train / test envs -------------------- #
    # ------------------------------------------------------------------ #
    def get_obs_rms(self):
        return {k: copy.deepcopy(v) for k, v in self.obs_rms.items()}

    def set_obs_rms(self, other: dict[str, RunningMeanStd]):
        self.obs_rms = defaultdict(
            lambda: RunningMeanStd(),
            {k: copy.deepcopy(v) for k, v in other.items()},
        )
