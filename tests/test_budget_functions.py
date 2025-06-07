import unittest

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


if __name__ == '__main__':
    unittest.main()
