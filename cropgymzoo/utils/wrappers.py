import copy
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
import pettingzoo
import torch

from gymnasium import spaces
from tianshou.env import PettingZooEnv, VectorEnvNormObs, BaseVectorEnv, VectorEnvWrapper
from tianshou.utils import RunningMeanStd



class VecNormObs(VectorEnvNormObs):
    """
    Normalises observation and runs several Batch pre-processing for Tianshou training.
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
                collapsed[k] = 0.0
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
        obs_extracted = obs_extracted.astype(np.float32)


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

        # Process obs

        obs = step_results[0].copy()
        if isinstance(obs, (list, np.ndarray)):  # the common case
            obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs])

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_extracted)
        obs_extracted = self._norm_obs(obs_extracted)
        obs_extracted = obs_extracted.astype(np.float32)

        for i, venv_obs in enumerate(obs_extracted):
            obs[i]["obs"] = obs_extracted[i]

        # Process info

        info = []
        for i, _info in enumerate(step_results[-1]):
            info.append(self.collapse_info_dict(_info))
        info = np.stack(info)

        # Process terminates

        terms = []
        deads = []
        envs = self.venv.workers
        for env in envs:
            env = _get_env(env) # risky risky here
            term_signal = getattr(env, "terminations", {})
            dead_step = getattr(env, "dead_step", {})
            terms.append(term_signal)
            deads.append(dead_step)

        terminateds = []
        for i, (term, dead) in enumerate(zip(terms, deads)):
            termed = all(term.values())
            # deaded = bool(dead) and all(dead.values()) # list(dead.values()).count(False) == 1  # hacky hacky wacky
            terminated = termed
            terminateds.append(terminated)
        terminateds = np.array(terminateds, dtype=bool)

        return obs, step_results[1], terminateds, step_results[-2], info

def _get_env(env):
    if hasattr(env, "env"):
        return _get_env(env.env)
    return env


def _new_rms():
    # from tianshou.utils.statistics import RunningMeanStd
    return RunningMeanStd()