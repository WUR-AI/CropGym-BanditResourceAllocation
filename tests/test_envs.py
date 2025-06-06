import unittest

import gymnasium as gym
from cropgymzoo.envs.singular_env import ParcelEnv

class TestCreationSingularEnv(unittest.TestCase):
    def test_create_singular_env(self):

        env = gym.make('field-1')

        self.assertEqual(isinstance(env.unwrapped, ParcelEnv), True)  # add assertion here

if __name__ == '__main__':
    unittest.main()
