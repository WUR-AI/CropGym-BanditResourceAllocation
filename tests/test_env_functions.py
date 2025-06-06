import unittest

import cropgymzoo  # for gym make
import gymnasium as gym

class TestEnvFunctions(unittest.TestCase):
    def test_functions_singular(self):

        env = gym.make('field-1')

        env = self.test_reset_singular(env)
        env = self.test_step_singular(env)
        env = self.test_terminate_singular(env)

        del env


    def test_reset_singular(self, env):
        obs, info = env.reset(options={'year': 2010, 'budget_n': 200})

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)
        self.assertEqual(env.unwrapped.year, 2010)
        self.assertEqual(env.unwrapped.budget_n, 200)
        # self.assertIn(env.unwrapped.crop_features, list(obs.keys()))
        # self.assertIn(env.unwrapped.action_features, list(obs.keys()))
        return env

    def test_step_singular(self, env):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())

        print(obs)
        print(rew)
        print(info)

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(rew, 0)
        # self.assertIn(env.unwrapped.crop_features, list(obs.keys()))
        # self.assertIn(env.unwrapped.action_features, list(obs.keys()))

        return env

    def test_terminate_singular(self, env):

        rew = 0
        term = False
        while not term:
            obs, rew, term, trunc, info = env.step(env.action_space.sample())

        self.assertNotEqual(rew, 0)

        return env


if __name__ == '__main__':
    unittest.main()
