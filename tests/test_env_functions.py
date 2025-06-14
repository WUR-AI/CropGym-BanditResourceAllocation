import unittest

import cropgymzoo  # for gym make
import gymnasium as gym

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

        # check NUE reward
        self.assertEqual(rew, 0)

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
