import unittest

import numpy as np

import cropgymzoo  # for gym make
import gymnasium as gym

from cropgymzoo.envs.worker_env import ParallelRLWorkers


class TestSingularRewardFunctions(unittest.TestCase):
    def setUp(self):
        self.env_nue = gym.make('field-1', reward='NUE')
        self.env_pny_1 = gym.make('field-1', reward='PNY')
        self.env_pny_2 = gym.make('field-4', reward='PNY')

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

    def test_pny_beets1(self):
        # crop sugarbeets
        _, info = self.env_pny_1.reset(options={'year': 2010})
        self.env_pny_1.unwrapped.set_budget(200)

        rewards = []

        for _ in range(5):
            _, reward, terminated, _, info = self.env_pny_1.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_1.step(8)
        rewards.append(reward)
        print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_1.step(3)
        rewards.append(reward)
        print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        while not terminated:
            _, reward, terminated, _, info = self.env_pny_1.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")

        lo, hi = self.env_pny_1.unwrapped.reward_class.reward_bounds()

        print(lo, hi)

        self.assertTrue(lo <= np.sum(rewards) <= hi)

    def test_pny_beets2(self):
        # crop sugarbeets
        _, info = self.env_pny_2.reset(options={'year': 2010})
        self.env_pny_1.unwrapped.set_budget(200)

        rewards = []

        for _ in range(5):
            _, reward, terminated, _, info = self.env_pny_2.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_2.step(8)
        rewards.append(reward)
        print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_2.step(3)
        rewards.append(reward)
        print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        while not terminated:
            _, reward, terminated, _, info = self.env_pny_2.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")

        lo, hi = self.env_pny_2.unwrapped.reward_class.reward_bounds()

        print(lo, hi)

        self.assertTrue(lo <= np.sum(rewards) <= hi)


if __name__ == '__main__':
    unittest.main()
