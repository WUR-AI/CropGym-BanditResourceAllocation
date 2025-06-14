import unittest

import numpy as np

import cropgymzoo  # for gym make
import gymnasium as gym

from cropgymzoo.envs.worker_env import ParallelRLWorkers


class TestSingularRewardFunctions(unittest.TestCase):
    def setUp(self):
        self.env_nue = gym.make('field-1', reward='NUE')
        self.env_pny = gym.make('field-1', reward='PNY')

    def test_nue(self):
        # crop sugarbeets
        _, info = self.env_nue.reset(options={'year': 2010})
        self.env_nue.unwrapped.set_budget(200)

        for _ in range(5):
            _, reward, terminated, _, info = self.env_nue.step(0)
        _, reward, terminated, _, info = self.env_nue.step(8)
        _, reward, terminated, _, info = self.env_nue.step(3)
        while not terminated:
            _, reward, terminated, _, info = self.env_nue.step(0)

        self.assertGreaterEqual(reward, 1)

    def test_pny(self):
        # crop sugarbeets
        _, info = self.env_pny.reset(options={'year': 2010})
        self.env_pny.unwrapped.set_budget(200)

        rewards = []

        for _ in range(5):
            _, reward, terminated, _, info = self.env_pny.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny.step(8)
        rewards.append(reward)
        print(f"reward in step {self.env_pny.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny.step(3)
        rewards.append(reward)
        print(f"reward in step {self.env_pny.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        while not terminated:
            _, reward, terminated, _, info = self.env_pny.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")

        lo, hi = self.env_pny.unwrapped.reward_class.reward_bounds()

        self.assertTrue(lo <= np.sum(rewards) <= hi)


if __name__ == '__main__':
    unittest.main()
