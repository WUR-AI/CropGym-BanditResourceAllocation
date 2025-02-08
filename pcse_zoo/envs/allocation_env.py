from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import gymnasium as gym

import numpy as np

from pcse_gym.envs.winterwheat import WinterWheat as WorkerEnv

class AllocationEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_envs):
        super().__init__()
        self.num_envs = num_envs

        self.agents = ['coordinator'] + [f'farmer_{i}' for i in range(num_envs)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {agent: idx for idx, agent in enumerate(self.agents)}

        # To manage agent turn order
        self._agent_selector = agent_selector(self.agents)

        # Create worker environments and set up their observation/action spaces.
        self.worker_envs = []
        for i in range(num_envs):
            env = WorkerEnv()  # instantiate your Gymnasium worker env
            self.worker_envs.append(env)
            self.observation_spaces[f'worker_{i}'] = env.observation_space
            self.action_spaces[f'worker_{i}'] = env.action_space

        self.reset()

    def reset(self, seed=None, options=None):
        # Reset all workers.
        self.worker_obs = []
        for env in self.worker_envs:
            obs, _ = env.reset(seed=seed)
            self.worker_obs.append(obs)

        # Build the coordinator's observation (e.g., a summary of worker states).
        # This is just a dummy example; you can design your observation as needed.
        self.coordinator_obs = np.array([np.mean(obs) for obs in self.worker_obs], dtype=np.float32)

        # Assemble observations for all agents.
        self.observations = {
            'coordinator': self.coordinator_obs
        }
        for i in range(self.num_workers):
            self.observations[f'worker_{i}'] = self.worker_obs[i]

        # Set up rewards, dones, and infos.
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Reset the agent selector.
        self.agent_selection = self._agent_selector.reset()