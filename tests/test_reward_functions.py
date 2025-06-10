import unittest

import cropgymzoo  # for gym make
import gymnasium as gym

from cropgymzoo.envs.worker_env import ParallelRLWorkers


class TestSingularRewardFunctions(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('field-1')

    def test_nue(self):
        # crop sugarbeets
        _, info = self.env.reset(options={'year': 2010})
        self.env.unwrapped.set_budget(200)

        for _ in range(5):
            _, reward, terminated, _, info = self.env.step(0)
        _, reward, terminated, _, info = self.env.step(8)
        _, reward, terminated, _, info = self.env.step(3)
        while not terminated:
            _, reward, terminated, _, info = self.env.step(0)

        self.assertGreaterEqual(reward, 1)


if __name__ == '__main__':
    unittest.main()
