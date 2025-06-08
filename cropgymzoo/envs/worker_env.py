import os
import yaml

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv

from cropgymzoo import _FIELDS_CONFIG

class ParallelRLWorkers(ParallelEnv):
    metadata = {
        "name": "CropGymZooEnv",
    }

    def __init__(self,
                 seed: int = 107,
                 allocator: str = 'random',
                 global_budget: int = 400,
                 warm_up: int = 100):

        self.seed = seed

        with open(_FIELDS_CONFIG) as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        self.n_agents = len(dict_fields)
        self.agents = [i for i in dict_fields.keys()]
        self.possible_agents = self.agents.copy()

        # either 'random' or 'bandit'
        self.allocator = allocator

        self.global_budget = global_budget

        self.warm_up = warm_up

        self._init_fields()
        self._init_spaces()
        self._init_farm_variables()



    def reset(self, seed=None, options=None):

        # get decision for bandit here?
        # but I need first observations to then assign budget with the options dict...

        locals_, infos = {}, {}
        for ag, env in self.fields.items():
            o, i = env.reset(seed=seed, options=options)
            locals_[ag], infos[ag] = o, i

        obs = {ag: {"local": locals_[ag],
                    "shared": self.shared_space,
                    "action_mask": self._get_mask(ag)}
               for ag in self.agents}

        # TODO change this!
        self.global_budget = {ag: options['global_budget'] for ag in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):

        # init dict for each variable
        locals_, rews, terms, truncs, infos = {}, {}, {}, {}, {}

        # loop through agent steps
        for ag, env in self.fields.items():
            o, r, t, tr, i = env.step(actions[ag])
            # self._apply_dose(ag, actions[ag])          # update budget
            locals_[ag], rews[ag] = o, r
            terms[ag], truncs[ag], infos[ag] = t, tr, i

        obs = {ag: {"local": locals_[ag],
                    "shared": self.shared_space,
                    "action_mask": self._get_mask(ag)}
               for ag in self.agents}

        self.agents = [ag for ag in self.agents if not (terms[ag] or truncs[ag])]
        return obs, rews, terms, truncs, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_mask(self, agent):
        return self.fields[agent].unwrapped.action_mask()

    def get_field_env(self, n: int):
        return self.fields[self.agents[n]]

    def _init_farm_variables(self):
        self.global_budget_left = self.global_budget

    def _init_fields(self):
        self.fields = {}
        # create each gymnasium env
        for n in self.agents:
            env = gym.make(n, seed=self.seed)  # set same seed for each field. Change?
            self.fields[n] = env

    def _init_spaces(self):
        # TODO
        shared_dim = ...

        self.shared_space = self._build_shared()

        self.observation_spaces = {
            ag: gym.spaces.Dict({
                "local": env.observation_space,
                "shared": self.shared_space,
                "action_mask": env.unwrapped.action_mask(),
            }) for ag, env in self.fields.items()
        }
        self.action_spaces = {agent: env.action_space
                              for agent, env in self.fields.items()}

    def _build_shared(self) -> dict[str, np.ndarray | list]:
        """
        Selected transformed features for shared observation.
        Change features in self._get_shared_obs_keys()
        """
        # change this functionality
        shared_obs = {}
        for feature in self._get_shared_obs_keys():
            # now a list. Maybe a dict?
            # Aggregate somehow?
            shared_obs[feature] = [env.unwrapped.get_latest_info(feature) for env in self.fields.values()]
        return shared_obs

    def _warm_up(self):
        ...

    def _get_shared_obs_keys(self):
        return ["NO3", "NH4", "Yield", "BudgetLeft", "Naction", "NamountSO", "FertilizerPrice", "CropCode"]
