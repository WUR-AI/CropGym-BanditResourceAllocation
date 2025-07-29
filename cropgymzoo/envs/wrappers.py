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
                 update_obs_rms: bool = True,
                 shared: bool = False,
                 device="cpu",):
        super().__init__(venv, update_obs_rms)
        self.device = device

        # hacky here; do we need just an input?
        self.agents = _get_env(venv.workers[0].env).agents

        self.shared = shared
        self.num_agents = len(self.agents)
        self.obs_rms: RunningMeanStd | dict = (
            RunningMeanStd() if shared else {
                agent_id: RunningMeanStd()
                for agent_id in self.agents
            }
        )

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
            agent_ids = np.array([d["agent_id"] for d in obs_extracted])
            obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs_extracted])

        if self.shared:
            if self.obs_rms and self.update_obs_rms:
                self.obs_rms.update(obs_extracted)
            obs_extracted = self._norm_obs(obs_extracted)
            obs_extracted = obs_extracted.astype(np.float32)
        else:
            for i, agent_id in enumerate(agent_ids):
                if self.obs_rms[agent_id] and self.update_obs_rms:
                    self.obs_rms[agent_id].update(obs_extracted)
                obs_extracted[i] = self._norm_obs(obs_extracted[i], agent_id)
                obs_extracted[i] = obs_extracted[i].astype(np.float32)

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
            agent_ids = np.array([d["agent_id"] for d in obs])
            obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs])

        if self.shared:
            if self.obs_rms and self.update_obs_rms:
                self.obs_rms.update(obs_extracted)
            obs_extracted = self._norm_obs(obs_extracted)
            obs_extracted = obs_extracted.astype(np.float32)
        else:
            for i, agent_id in enumerate(agent_ids):
                if self.obs_rms[agent_id] and self.update_obs_rms:
                    self.obs_rms[agent_id].update(obs_extracted)
                obs_extracted[i] = self._norm_obs(obs_extracted[i], agent_id)
                obs_extracted[i] = obs_extracted[i].astype(np.float32)

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

    def _norm_obs(self, obs: np.ndarray, agent_id=None) -> np.ndarray:
        if self.shared:
            if self.obs_rms:
                return self.obs_rms.norm(obs)  # type: ignore
        else:
            if self.obs_rms[agent_id]:
                return self.obs_rms[agent_id].norm(obs)
        return obs

    def get_obs_rms(self) -> RunningMeanStd | dict[str, RunningMeanStd]:
        return self.obs_rms

    @staticmethod
    def collapse_info_dict(info: dict[str, dict[str, list | float]]) -> dict[str, dict[str, float]]:
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

def _get_env(env):
    if hasattr(env, "env"):
        return _get_env(env.env)
    return env


def _new_rms():
    # from tianshou.utils.statistics import RunningMeanStd
    return RunningMeanStd()