import unittest

import cropgymzoo  # for gym make
import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt

from cropgymzoo.envs.worker_env import ParallelRLWorkers


class TestSingularEnvFunctions(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('field-1')

    def test_reset_singular(self):
        obs, info = self.env.reset(options={'year': 2010})
        self.env.unwrapped.set_budget(200)

        # check obs and info type
        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

        print(obs)

        # check reset options
        self.assertEqual(self.env.unwrapped.year, 2010)
        self.assertEqual(self.env.unwrapped.budget_n, 200)

    def test_step_singular(self):
        obs, info = self.env.reset(options={'year': 2010})
        obs, rew, term, trunc, info = self.env.step(1)


        self.assertEqual(isinstance(obs, dict), True)

        # check PNY reward
        # self.assertEqual(rew, 0)

        # check whether obs and info align
        self.assertEqual(obs['NO3'], info['NO3'][-1])

    def test_terminate_singular(self):
        obs, info = self.env.reset(options={'year': 2010})

        rew = 0
        term = False
        while not term:
            obs, rew, term, trunc, info = self.env.step(1)

        print(info)

        # again
        self.assertEqual(obs['NO3'], info['NO3'][-1])

    def test_replace_year_every_terminate_singular(self):
        obs, info = self.env.reset(options={'year': 2010})

        term = False
        while not term:
            obs, rew, term, trunc, info = self.env.step(0)

        self.assertEqual(self.env.unwrapped.year, 2010)

        print(info["CO2"][-1])

        obs, info = self.env.reset(options={'year': 1985})

        term = False
        while not term:
            obs, rew, term, trunc, info = self.env.step(0)

        print(info["CO2"][-1])

        self.assertEqual(self.env.unwrapped.year, 1985)

    def test_infos_shape_singular(self):
        obs, info = self.env.reset(options={'year': 2010})

        term = False
        while not term:
            obs, rew, term, trunc, info = self.env.step(0)

        for feature in info:
            self.assertEqual(isinstance(info[feature], list), True)

class TestSingularTrainingEnvFunctions(unittest.TestCase):
    def setUp(self):
        self.env_training = gym.make('field-1', training=True)
        self.env_testing = gym.make('field-1')

    def test_reset_singular(self):
        obs, info = self.env_training.reset(options={'year': 2010})
        self.env_training.unwrapped.set_budget(200)

        self.assertEqual(self.env_training.unwrapped.training, True)

        # check obs and info type
        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

        print(obs)

        # check reset options
        self.assertEqual(self.env_training.unwrapped.year, 2010)
        self.assertEqual(self.env_training.unwrapped.budget_n, 200)

    def test_reset_compare(self):
        obs_train, info_train = self.env_training.reset(options={'year': 2010})
        obs_test, info_test = self.env_testing.reset(options={'year': 2010})

        print(obs_train)
        print(obs_test)

        self.assertNotEqual(obs_train, obs_test)

    def test_episode_compare(self):
        year = np.random.choice(range(1951, 2025))
        obs_train, info_train = self.env_training.reset(options={'year': year})
        obs_test, info_test = self.env_testing.reset(options={'year': year})

        term_train = False
        while not term_train:
            obs_train, _, term_train, _, info_train = self.env_training.step(0)

        term_test = False
        while not term_test:
            obs_test, _, term_test, _, info_test = self.env_testing.step(0)

        plt.plot(info_train['Yield'], label='train yield')
        plt.plot(info_test['Yield'], label='test yield')

        plt.plot(info_train['TAGP'], label='train TAGP')
        plt.plot(info_test['TAGP'], label='test TAGP')

        plt.legend()

        plt.show()

        self.assertNotEqual(obs_train, obs_test)


class TestMultiEnvFunctions(unittest.TestCase):
    def setUp(self):
        self.env = ParallelRLWorkers(
            warm_up=0,
            global_budget=500,
        )

    def test_reset_multi(self):
        obs, info = self.env.reset(options={'year': 2010})

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

    def test_step_multi(self):
        obs, info = self.env.reset(options={'year': 2010})
        obs, rew, term, trunc, info = self.env.step({
            agent: 0 for agent in self.env.unwrapped.agents
        })

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

    def test_render_multi(self):
        obs, info = self.env.reset(options={'year': 2010})
        obs, rew, term, trunc, info = self.env.step({
            agent: 0 for agent in self.env.unwrapped.agents
        })
        self.env.render()

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

    def test_terminate_multi(self):
        obs, info = self.env.reset(options={'year': 2010})

        terms = {agent: False for agent in self.env.unwrapped.agents}
        while not all(terms.values()):
            obs, rew, terms, trunc, info = self.env.step({
                agent: 0 for agent in self.env.unwrapped.agents
            })
        print(info)

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

    def test_print_multi(self):
        obs, info = self.env.reset(options={'year': 2010})

        terms = {agent: False for agent in self.env.unwrapped.agents}
        while not all(terms.values()):
            obs, rew, terms, trunc, info = self.env.step({
                agent: 0 for agent in self.env.unwrapped.agents
            })

        print(self.env)

        self.assertIn("Farm", self.env.__str__())

    def test_multi_years(self):
        obs, info = self.env.reset(options={'year': 2010})

        terms = {agent: False for agent in self.env.unwrapped.agents}
        while not all(terms.values()):
            obs, rew, terms, trunc, info = self.env.step({
                agent: 0 for agent in self.env.unwrapped.agents
            })

        self.assertEqual(isinstance(info, dict), True)

        obs, info = self.env.reset(options={'year': 1999})

        terms = {agent: False for agent in self.env.unwrapped.agents}
        while not all(terms.values()):
            obs, rew, terms, trunc, info = self.env.step({
                agent: 0 for agent in self.env.unwrapped.agents
            })

        print(self.env)

        self.assertEqual(isinstance(info, dict), True)

class TestMultiEnvWarmUp(unittest.TestCase):
    def setUp(self):
        self.env = ParallelRLWorkers(
            warm_up=10,
            global_budget=500,
        )

    def test_warm_multi(self):
        obs, info = self.env.reset(options={'year': 2010})

        print(next(iter(self.env.warm_up_infos.values()))['field-2']['Naction'])

        self.assertNotEqual(self.env.warm_up_infos, None)

if __name__ == '__main__':
    unittest.main()
