import unittest
from typing import cast

import numpy as np

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.envs.multi_field_env import MultiFieldEnv

import gymnasium as gym

class TestBudgetDynamics(unittest.TestCase):
    def setUp(self):
        self.env = cast(ParcelEnv, gym.make('field-1'))

        self.multi_env = MultiFieldEnv(
            warm_up=0,
            training=True,
            random_budget=True,
        )

    def test_budget_subtractions(self):
        _, _ = self.env.reset(options={'year': 2010})
        self.env.unwrapped.set_budget(200)

        _, _, _, _, info = self.env.step(1)

        self.assertEqual(self.env.unwrapped.budget_left, 190)  # add assertion here

        _, _, _, _, info = self.env.step(8)

        _, _, _, _, info = self.env.step(8)

        self.assertEqual(self.env.unwrapped.budget_left, 30)

        _, _, _, _, info = self.env.step(3)

        self.assertEqual(self.env.unwrapped.budget_left, 0)

        # reset again with different budget!
        _, _ = self.env.reset(options={'year': 2010})
        self.env.unwrapped.set_budget(150)

        self.assertEqual(self.env.unwrapped.budget_left, 150)

        _, _, _, _, info = self.env.step(3)

        self.assertEqual(self.env.unwrapped.budget_left, 120)

    def test_action_mask(self):
        obs, info = self.env.reset(options={'year': 2010})
        self.env.unwrapped.set_budget(200)

        _, _, _, _, info = self.env.step(8)

        print(np.array([True for _ in range(self.env.action_space.n)]))
        print(self.env.unwrapped.action_mask())
        self.assertEqual(
            np.array_equal(
                np.array([True for _ in range(self.env.action_space.n)]),
                self.env.unwrapped.action_mask(),
            ), True
        )

        _, _, _, _, info = self.env.step(8)

        self.assertEqual(
            np.array_equal(
                np.array([True, True, True, True, True, False, False, False, False]),
                self.env.unwrapped.action_mask(),
            ), True
        )

        _, _, _, _, info = self.env.step(1)

        self.assertEqual(
            np.array_equal(
                np.array([True, True, True, True, False, False, False, False, False]),
                self.env.unwrapped.action_mask(),
            ), True
        )

        _, _, _, _, info = self.env.step(4)

        self.assertEqual(
            np.array_equal(
                np.array([True, False, False, False, False, False, False, False, False]),
                self.env.unwrapped.action_mask(),
            ), True
        )

    def test_multi_random_budget(self):
        for i in range(10):
            self.multi_env.reset(options={'year': 2010})
            print(f"Global max budget: {self.multi_env._get_global_max_budget()}")
            print(f"Global budget: {self.multi_env._get_global_budget()}")

            budget_sum = []
            for agent in self.multi_env.possible_agents:
                budget = self.multi_env.get_per_parcel_budget(agent)
                budget_max = self.multi_env.get_per_parcel_max_budget(agent)
                print(f"Agent {agent} max budget: {budget_max}")
                print(f"Agent {agent} budget: {budget}")
                budget_sum.append(budget)
                self.assertLessEqual(40., budget)

            self.assertEqual(np.sum(budget_sum), self.multi_env._get_global_budget())

if __name__ == '__main__':
    unittest.main()
