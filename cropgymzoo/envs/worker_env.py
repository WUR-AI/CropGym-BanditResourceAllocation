import os
import gymnasium as gym
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv

class ParallelRLWorkers(ParallelEnv):
    metadata = {
        "name": "CropGymZooEnv",
    }

    def __init__(self,
                 n_fields: int = 6):
        self.n_agents = n_fields
        self.agents = [f"field{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents.copy()

        self._init_fields()

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
        for n in self.agents:
            ...

