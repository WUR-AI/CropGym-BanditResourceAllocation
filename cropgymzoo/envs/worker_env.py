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
                 seed: int = 107,):

        self.seed = seed

        with open(_FIELDS_CONFIG) as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        self.n_agents = len(dict_fields)
        self.agents = [i for i in dict_fields.keys()]
        self.possible_agents = self.agents.copy()

        self._init_fields()
        self._init_spaces()



    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _init_fields(self):
        self.fields = {}
        # create each gymnasium env
        for n in self.agents:
            env = gym.make(n, seed=self.seed)
            self.fields[n] = env

    def _init_spaces(self):
        # TODO
        shared_dim = ...

        self.shared_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(shared_dim,), dtype=np.float32)

        self.observation_spaces = {
            ag: gym.spaces.Dict({
                "local": env.observation_space,
                "shared": self.shared_space  # same for all
            }) for ag, env in self.fields.items()
        }
        self.action_spaces = {agent: env.action_space
                              for agent, env in self.fields.items()}

    def _build_shared(self, locals_: dict[str, np.ndarray]) -> np.ndarray:
        """Example: simply concatenate the local vectors in a fixed order."""
        # change this functionality
        return np.concatenate([locals_[ag].ravel() for ag in self.agents])
