from abc import ABC, abstractmethod
from typing import Any

import torch
import numpy as np

import gymnasium as gym
import pettingzoo

from tianshou.env.venv_wrappers import BaseVectorEnv
from tianshou.env import PettingZooEnv
from tianshou.data import Batch
from tianshou.policy import MultiAgentPolicyManager

from cropgymzoo.train_policy import (
    initialize_policy,
    load_model
)
from cropgymzoo.agents.lagppo import IPPOPolicy
from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.plotters import plot_year, plot_results


def load_policy(
        env: pettingzoo.AECEnv,
        saved_model: Any,
):
    args = saved_model['args']

    agent = env.agents[0]
    obs_dim = env.sample_observation_space_agent().shape
    act_dim = env.action_spaces[agent].n

    policies = {
        a: initialize_policy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=args.hidden_layers,
            use_icm=args.use_icm,
            args=args,
            skew_prior_action=False,
        ) for a in env.agents
    }
    print(f"Using {'LagrangianIPPO' if args.lagrangian_ppo else 'IPPO'} policy!")

    policy_manager = MultiAgentPolicyManager(
        policies=list(policies.values()),
        env=PettingZooEnv(env),
    )

    # load models and rms
    for agent, policy in policy_manager.policies.items():
        policy.load_state_dict(saved_model['models'][agent], strict=True)
        policy.eval()
        policy.deterministic_eval = True

    obs_rms = saved_model["obs_rms"]

    return policy_manager, obs_rms


def predict_policy(
        obs,
        agent,
        mask,
        policy,
        obs_rms,
        info,
        next_states = None,
) -> tuple[int, Batch | None]:
    out = policy.policies[agent](
        batch=Batch(
            {
                'obs': {
                    'obs': obs_rms.norm(obs['observation']),
                    'mask': mask,
                },
                'info': info,
            }
        ),
        state=Batch(next_states[agent]) if next_states is not None else None,
    )
    return out.act.item(), None if next_states is None else out.state


class BaseAgent(ABC):
    def __init__(
        self,
        env: gym.Env | pettingzoo.AECEnv | BaseVectorEnv,
        render: bool = False,
        **kwargs
    ):
        self.env = env
        self.render = render

        self.agents = self.env.possible_agents

    def run(self, years: list, year_key=True, scenario: str = 'max') -> dict:

        info_dict = {}
        for year in years:
            if year_key:
                info_dict[year] = {}

            self.env.reset(options={'year': year})

            for agent in self.env.agent_iter():
                obs, rew, term, trunc, info = self.env.last()

                action = self.get_action(agent, env=self.env, scenario=scenario)

                if self.env.terminations[agent]:
                    if year_key:
                        info_dict[year][agent] = info
                    else:
                        info_dict[agent] = info
                    self.env.step(None)
                else:
                    self.env.step(action)
            if self.render:
                self.env.render()
        return info_dict


    @abstractmethod
    def get_action(self, agent: str, env = None, scenario = None):
        raise NotImplementedError


class RoTAgent(BaseAgent):
    def __init__(
        self,
        env: gym.Env | pettingzoo.AECEnv | BaseVectorEnv,
        render: bool = True,
    ):
        super().__init__(env, render)

    def get_action(
            self,
            agent: str,
            env: MultiFieldEnv = None,
            scenario: str = 'max',
    ) -> np.ndarray:
        return env.rule_of_thumb(agent, scenario=scenario)

class RandomAgent(BaseAgent):
    def __init__(
        self,
        env: gym.Env | pettingzoo.AECEnv | BaseVectorEnv,
        render: bool = True,
    ):
        super().__init__(env, render)

    def get_action(
            self,
            agent: str,
            env: MultiFieldEnv = None,
            scenario: str = 'max',
    ) -> np.ndarray:
        return env.random_fertilization(agent)



class MultiRLAgent(BaseAgent):
    def __init__(
            self,
            env: pettingzoo.AECEnv | BaseVectorEnv,
            saved_model: dict,
            render: bool = False,
    ):
        super().__init__(env, render)

        # dummy_env, agents, obs_dim, act_dim = grab_spaces(seed)

        args = saved_model['args']

        obs_dim = self.env.sample_observation_space_agent().shape
        act_dim = self.env.action_spaces[self.agents[0]].n

        policies = {
            a: initialize_policy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden=args.hidden_layers,
                use_icm=args.use_icm,
                args=args,
                skew_prior_action=False,
            ) for a in self.agents
        }
        print(f"Using {str(args.alg).upper()} policy!")

        self.policy_manager = MultiAgentPolicyManager(
            policies=list(policies.values()),
            env=PettingZooEnv(self.env),
        )

        # load models and rms
        for agent, policy in self.policy_manager.policies.items():
            policy.load_state_dict(saved_model['models'][agent], strict=True)
            policy.eval()
            policy.deterministic_eval = True

        self.obs_rms = saved_model["obs_rms"]

    def get_action(
            self,
            agent: str,
            obs: Batch = None,
            next_states: Batch | dict = None,
            info: Batch = None,
    ) -> Batch:
        out = self.policy_manager.policies[agent](
            batch=Batch(
                {
                    'obs': {
                        'obs': self.obs_rms.norm(obs['observation'])
                        if not isinstance(self.obs_rms, dict)
                        else self.obs_rms[agent].norm(obs['observation']),
                        'mask': self.env._get_mask(agent),
                    },
                    'info': info,
                }
            ),
            state=Batch(next_states[agent]),
        )

        return out

    def run(self, years: list, year_key=True, scenario=None) -> dict:
        info_dict = {}
        for i, year in enumerate(years):
            if year_key:
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
                    out = self.get_action(
                        agent,
                        obs=obs,
                        next_states=next_states,
                        info=processed_info,
                    )

                action = out.act.item()
                state = None if not hasattr(out, 'state') else out.state

                next_states[agent] = state

                if self.env.terminations[agent]:
                    if year_key:
                        info_dict[year][agent] = info
                    else:
                        info_dict[agent] = info
                    self.env.step(None)
                else:
                    self.env.step(action)

            if self.render:
                self.env.render()
        return info_dict


def continue_training():
    ...


def run_episodes(args) -> None:

    # make eval env
    env = MultiFieldEnv(
        warm_up=0,
        training=False,
    )

    model = load_model(args)

    years_list = list(range(1990, 2020))

    runner = MultiRLAgent(
        env=env,
        saved_model=model,
        seed=args.seed,
        render=True,
        use_icm=args.use_icm,
    )

    results_dict = runner.run(years_list)

    for year in years_list:
        plot_results(
            results_dict[year]
        )

