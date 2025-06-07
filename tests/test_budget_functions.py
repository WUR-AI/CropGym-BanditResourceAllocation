import unittest

import numpy as np

import cropgymzoo
import gymnasium as gym

class TestBudgetDynamics(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('field-1')

    def test_budget_subtractions(self):
        _, _ = self.env.reset(options={'year': 2010, 'budget_n': 200})

        _, _, _, _, info = self.env.step(1)

        self.assertEqual(self.env.unwrapped.budget_left, 190)  # add assertion here

        _, _, _, _, info = self.env.step(8)

        _, _, _, _, info = self.env.step(8)

        self.assertEqual(self.env.unwrapped.budget_left, 30)

        _, _, _, _, info = self.env.step(3)

        self.assertEqual(self.env.unwrapped.budget_left, 0)

        # reset again with different budget!
        _, _ = self.env.reset(options={'year': 2010, 'budget_n': 150})

        self.assertEqual(self.env.unwrapped.budget_left, 150)

        _, _, _, _, info = self.env.step(3)

        self.assertEqual(self.env.unwrapped.budget_left, 120)

    def test_action_mask(self):
        obs, info = self.env.reset(options={'year': 2010, 'budget_n': 200})

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


if __name__ == '__main__':
    unittest.main()
