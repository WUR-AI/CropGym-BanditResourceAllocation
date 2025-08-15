from abc import ABC, abstractmethod
from typing import Any

import torch

import gymnasium as gym
import pettingzoo

from tianshou.env.venv_wrappers import BaseVectorEnv
from tianshou.env import PettingZooEnv
from tianshou.data import Batch
from tianshou.policy import MultiAgentPolicyManager

from cropgymzoo.train_tianshou import make_vec_env, grab_spaces, make_ppo_policy, load_model


class BaseAgent(ABC):
    def __init__(
        self,
        env: gym.Env | pettingzoo.AECEnv | BaseVectorEnv,
        render: bool = False,
        **kwargs
    ):
        self.env = env
        self.render = render

    def run(self, years: list) -> dict:

        info_dict = {}
        for year in years:
            info_dict[year] = {}

            self.env.reset(options={'year': year})

            for agent in self.env.agent_iter():
                obs, rew, term, trunc, info = self.env.last()

                action = self.get_action(agent)

                if self.env.terminations[agent]:
                    info_dict[year][agent] = info
                    self.env.step(None)
                else:
                    self.env.step(action)

        return info_dict


    @abstractmethod
    def get_action(self, agent: str) -> torch.Tensor:
        raise NotImplementedError


class MultiRLAgent(BaseAgent):
    def __init__(
            self,
            env: pettingzoo.AECEnv | BaseVectorEnv,
            saved_model: dict,
            seed: int = 107,
            render: bool = False,
    ):
        super().__init__(env, render)

        self.env = env
        self.agents = self.env.possible_agents

        # dummy_env, agents, obs_dim, act_dim = grab_spaces(seed)

        obs_dim = self.env.sample_observation_space_agent().shape
        act_dim = self.env.action_spaces[self.agents[0]].n

        policies = {
            a: make_ppo_policy(
                obs_dim=obs_dim,
                act_dim=act_dim,
            ) for a in self.agents
        }

        self.policy_manager = MultiAgentPolicyManager(
            policies=list(policies.values()),
            env=PettingZooEnv(self.env),
        )

        # load models and rms
        for agent, policy in self.policy_manager.policies.items():
            policy.load_state_dict(saved_model['model'][agent], strict=True)

        self.obs_rms = saved_model["obs_rms"]

    def run(self, years: list) -> dict:
        info_dict = {}
        for i, year in enumerate(years):
            info_dict[year] = {}

            next_states = {
                ag: None for ag in self.agents
            }

            self.env.reset(options={'year': year})

            for agent in self.env.agent_iter():
                obs, rew, term, trunc, info = self.env.last()

                # get appropriate info shape for policy
                processed_info = Batch({k: [i[-1]] for k, i in info.items()})
                processed_info['env_id'] = [0]

                with torch.no_grad():
                    out = self.policy_manager.policies[agent](
                        batch=Batch(
                            {
                                'obs': {
                                    'obs': self.obs_rms.norm(obs['observation']),
                                    'mask': self.env._get_mask(agent),
                                },
                                'info': processed_info,
                            }
                        ),
                        state=Batch(next_states[agent]),
                    )

                action = out.act.item()
                state = None if not hasattr(out, 'state') else out.state

                next_states[agent] = state

                if self.env.terminations[agent]:
                    info_dict[year][agent] = info  # grab info before agent dies
                    self.env.step(None)
                else:
                    self.env.step(action)
        return info_dict


def continue_training():
    ...


def run_inference(model) -> None:

    env = make_vec_env(
        False,
        True,
        1,
        True,
        False,
    )

    dummy_env, agents, obs_dim, act_dim = grab_spaces(107)

    policies = {a: make_ppo_policy(obs_dim, act_dim, recurrent=True) for a in agents}

