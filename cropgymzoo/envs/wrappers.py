from copy import deepcopy
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
import pettingzoo
import torch

from gymnasium import spaces
from tianshou.env import PettingZooEnv, VectorEnvNormObs, BaseVectorEnv, VectorEnvWrapper, SubprocVectorEnv
from tianshou.utils import RunningMeanStd



class MultiAgentVecNormObs(VectorEnvNormObs):
    """
    Normalises observation and runs several Batch pre-processing for Tianshou training.
    """
    def __init__(self,
                 venv: BaseVectorEnv,
                 agents: list[str],
                 update_obs_rms: bool = True,
                 shared: bool = True,
                 device: str = "cpu",
                 reward_scale: float = 1e5,
                 reward_clip: tuple[float, float] | None = None,
                 log_true_reward: bool = True,
     ):
        super().__init__(venv, update_obs_rms)
        self.device = device

        # hacky here; do we need just an input?
        # OK this doesn't work with subprocvecenv...
        # self.agents = _get_env(venv.workers[0].env).agents

        self.agents = agents
        self.shared = shared
        self.num_agents = len(self.agents)

        # reward scaling controls
        self.old_rew = None
        self.reward_scale: float = float(reward_scale)
        self.reward_clip: tuple[float, float] | None = reward_clip
        self.log_true_reward: bool = bool(log_true_reward)

        # observations
        self.obs_rms: RunningMeanStd | dict = (
            RunningMeanStd() if shared else {
                agent_id: RunningMeanStd()
                for agent_id in self.agents
            }
        )

        # terminateds
        self._terminateds = None
        self._vec_ids = None
        self.subproc = False
        if isinstance(self.venv, SubprocVectorEnv):
            self.subproc = True
            self._vec_ids = list(range(self.venv.env_num))
            self._terminateds = {
                id: {
                    ag: False
                    for ag in self.agents
                }
                for id in self._vec_ids
            }

        self.old_obs = None

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
        self.old_obs = obs
        agent_ids = np.array([d["agent_id"] for d in obs_extracted])
        obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs_extracted])

        if self.shared:
            if self.obs_rms and self.update_obs_rms:
                self.obs_rms.update(obs_extracted)
            obs_extracted = self._norm_obs(obs_extracted)
            obs_extracted = obs_extracted.astype(np.float32)
        else:
            # for i, agent_id in enumerate(agent_ids):
            #     if self.obs_rms[agent_id] and self.update_obs_rms:
            #         self.obs_rms[agent_id].update(obs_extracted[i])
            if env_id is not None:
                for i, (e, agent_id) in enumerate(zip(env_id, agent_ids)):
                    if self.obs_rms[agent_id] and self.update_obs_rms:
                        self.obs_rms[agent_id].update(obs_extracted[i])
                    obs_extracted[i] = self.obs_rms[agent_id].norm(obs_extracted[i])  # notice wrapped obs
            else:
                for i, agent_id in enumerate(agent_ids):
                    if self.obs_rms[agent_id] and self.update_obs_rms:
                        self.obs_rms[agent_id].update(obs_extracted[i])
                    obs_extracted[i] = self.obs_rms[agent_id].norm(obs_extracted[i])
            obs_extracted = obs_extracted.astype(np.float32)

        for i, venv_obs in enumerate(obs_extracted):
            obs[i]["obs"] = obs_extracted[i]

        for i, _info in enumerate(info):
            info[i] = self.collapse_info_dict(_info)

        # Reset terminations
        if self.subproc and env_id is not None:
            for idx in env_id:
                self._terminateds[idx] = {
                    agent_id: False for agent_id in self.agents
                }

        # initialize last raw rewards after reset
        try:
            env_num = getattr(self.venv, "env_num", None)
            if env_num is None:
                # fall back to len(obs)
                env_num = len(obs)
            self.old_rew = np.zeros(int(env_num), dtype=np.float32)
        except Exception:
            self.old_rew = None


        return obs, info

    def step(
            self,
            action: np.ndarray | torch.Tensor | None,
            env_id: int | list[int] | np.ndarray | None = None,
    ):
        step_results = self.venv.step(action, env_id)

        # Reward scaling: preserve raw reward, scale a sanitized copy
        raw_rew = step_results[1]
        if isinstance(raw_rew, torch.Tensor):
            raw_rew = raw_rew.detach().cpu().numpy()
        self.old_rew = np.array(raw_rew, copy=True)

        rew = np.array(raw_rew, dtype=np.float64, copy=True)
        rew = np.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        # fixed scale towards O(1-10) magnitudes for stability
        if self.reward_scale and self.reward_scale != 0.0:
            rew = rew / float(self.reward_scale)
        if self.reward_clip is not None:
            lo, hi = self.reward_clip
            rew = np.clip(rew, float(lo), float(hi))
        rew = rew.astype(np.float32)


        # Process obs

        obs = step_results[0].copy()

        self.old_obs = obs

        agent_ids = np.array([d["agent_id"] for d in obs])
        obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs])

        if self.shared:
            if self.obs_rms and self.update_obs_rms:
                self.obs_rms.update(obs_extracted)
            obs_extracted = self._norm_obs(obs_extracted)
            obs_extracted = obs_extracted.astype(np.float32)
        else:
            # for i, agent_id in enumerate(agent_ids):
            #     if self.obs_rms[agent_id] and self.update_obs_rms:
            #         self.obs_rms[agent_id].update(obs_extracted)
            # for i, (e, agent_id) in enumerate(zip(env_id, agent_ids)):
            #     if self.obs_rms[agent_id]:
            #         normed_obs = self.obs_rms[agent_id].norm([obs_extracted[e]])  # notice wrapped obs
            #         obs_extracted[e] = normed_obs[0]
            # obs_extracted = obs_extracted.astype(np.float32)
            if env_id is not None:
                for i, (e, agent_id) in enumerate(zip(env_id, agent_ids)):
                    if self.obs_rms[agent_id] and self.update_obs_rms:
                        self.obs_rms[agent_id].update(obs_extracted[i])
                    obs_extracted[i] = self.obs_rms[agent_id].norm(obs_extracted[i])  # notice wrapped obs
            else:
                for i, agent_id in enumerate(agent_ids):
                    if self.obs_rms[agent_id] and self.update_obs_rms:
                        self.obs_rms[agent_id].update(obs_extracted[i])
                    obs_extracted[i] = self.obs_rms[agent_id].norm(obs_extracted[i])
            obs_extracted = obs_extracted.astype(np.float32)

        for i, venv_obs in enumerate(obs_extracted):
            obs[i]["obs"] = obs_extracted[i]

        # Process info

        info = []
        for i, _info in enumerate(step_results[-1]):
            collapsed = self.collapse_info_dict(_info)
            # annotate rewards for logging/analysis
            if self.log_true_reward:
                try:
                    collapsed["TrueReward"] = float(self.old_rew[i])
                    collapsed["ScaledReward"] = float(rew[i])
                except Exception:
                    pass
            # Add resets flag in info
            collapsed["ResetMask"] = False
            info.append(collapsed)
        info = np.stack(info)

        # Process terminates
        terminateds = self._get_terminateds(obs, step_results[2], env_id)

        return obs, rew, terminateds, step_results[-2], info

    def _get_terminateds(self, obs, terminated_ids, env_id) -> np.array:
        if not self.subproc:
            terms = []
            envs = self.venv.workers
            for env in envs:
                env = _get_env(env)  # risky risky here
                term_signal = getattr(env, "terminations", {})
                terms.append(term_signal)

            terminateds = []
            for i, (term) in enumerate(terms):
                termed = all(term.values())
                terminated = termed
                terminateds.append(terminated)
            terminateds = np.array(terminateds, dtype=bool)
        else:
            agent_ids = [o['agent_id'] for o in obs]
            for agent_id, idx, term_signal in zip(agent_ids, env_id, terminated_ids):
                self._terminateds[idx][agent_id] = term_signal

            terms = [
                all(self._terminateds[idx].values())
                for idx in env_id
            ]

            terminateds = np.array(terms, dtype=bool)

        return terminateds

    def _norm_obs(self, obs: np.ndarray, agent_id=None) -> np.ndarray:
        if self.obs_rms:
            return self.obs_rms.norm(obs)  # type: ignore
        return obs

    def get_original_obs(self):
        return deepcopy(self.old_obs)

    def get_original_reward(self) -> np.ndarray:
        if getattr(self, "old_rew", None) is None:
            return np.array([], dtype=np.float32)
        return self.old_rew.copy()


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