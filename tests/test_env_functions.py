import unittest

import cropgymzoo  # for gym make
import gymnasium as gym


class TestEnvFunctions(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('field-1')

    def test_reset_singular(self):
        obs, info = self.env.reset(options={'year': 2010, 'budget_n': 200})

        # check obs and info type
        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

        # check reset options
        self.assertEqual(self.env.unwrapped.year, 2010)
        self.assertEqual(self.env.unwrapped.budget_n, 200)

    def test_step_singular(self):
        obs, info = self.env.reset(options={'year': 2010, 'budget_n': 200})
        obs, rew, term, trunc, info = self.env.step(1)


        self.assertEqual(isinstance(obs, dict), True)

        # check NUE reward
        self.assertEqual(rew, 0)

        # check whether obs and info align
        self.assertEqual(obs['NO3'], info['NO3'][-1])

    def test_terminate_singular(self):
        obs, info = self.env.reset(options={'year': 2010, 'budget_n': 200})

        rew = 0
        term = False
        while not term:
            obs, rew, term, trunc, info = self.env.step(1)
            print(self.env.unwrapped.n_steps)

        # again
        self.assertEqual(obs['NO3'], info['NO3'][-1])



if __name__ == '__main__':
    unittest.main()
