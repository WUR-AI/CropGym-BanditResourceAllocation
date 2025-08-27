import unittest
import os
import gymnasium as gym

import numpy as np

from cropgymzoo import _DEFAULT_PLOTDIR
from cropgymzoo.envs.allocation_env import AllocationBandit
from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.plotters import plot_results

try:
    from tianshou.env import PettingZooEnv, SubprocVectorEnv, DummyVectorEnv
except ImportError:
    tianshou = None

class TestCreationSingularEnv(unittest.TestCase):
    def setUp(self):
        self.env_1 = gym.make('field-1', training=True)
        self.env_2 = gym.make('field-2', training=True)
        self.env_3 = gym.make('field-3', training=True)
        self.env_4 = gym.make('field-4', training=True)
        self.env_5 = gym.make('field-5', training=True)
        self.env_6 = gym.make('field-6', training=True)

    def test_create_singular_env(self):

        env = gym.make('field-1')

        self.assertEqual(isinstance(env.unwrapped, ParcelEnv), True)  # add assertion here

    def test_running_several_episodes(self):

        envs = [self.env_1, self.env_2, self.env_3, self.env_4, self.env_5, self.env_6]

        for env in envs:
            for _ in range(4):
                year = np.random.choice(range(1951, 2024))
                env.reset(options={'year': year})

                print(f"Crop is {env.unwrapped.crop}")
                print(f"Training mode is {env.unwrapped.training}")

                start_year = year

                print(f"Date start: {env.unwrapped.date}")

                terminated = False
                while not terminated:
                    _, _, terminated, _, _ = env.step(0)

                date_end = env.unwrapped.date
                print(f"Date end: {date_end}")

                end_year = date_end.year

                checker = (start_year == end_year) or (start_year == (end_year - 1))

                print(f"Simulation year {start_year} is {checker}")

                end_yield = env.unwrapped.infos["Yield"][-1]
                nue = env.unwrapped.infos["Nue"][-1]

                print(f"End yield is {end_yield}")
                print(f"NUE is {nue}")

                self.assertTrue(checker)
                self.assertNotEquals(0, end_yield)
                self.assertNotEquals(0.0, nue)


class TestCreationMultiFieldEnv(unittest.TestCase):
    def setUp(self):
        self.env = MultiFieldEnv(
            warm_up=0,
            training=True
        )

    def test_running_several_episodes(self):
        for _ in range(4):
            year = np.random.choice(range(1951, 2024))
            self.env.reset(options={'year': year})

            for agent in self.env.possible_agents:
                print(f"{agent}'s crop is {self.env.fields[agent].unwrapped.crop} and start date is {self.env.fields[agent].unwrapped.date}")

            for agent in self.env.agent_iter():
                obs, rew, term, trunc, info = self.env.last()

                action = self.env.sample_masked_action(agent)
                if self.env.terminations[agent]:
                    self.env.step(None)
                else:
                    self.env.step(action)

            for agent in self.env.possible_agents:
                print(f"{agent}'s crop is {self.env.fields[agent].unwrapped.crop} and start date is {self.env.fields[agent].unwrapped.date}")
                print(f"{agent}'s NUE is {self.env.fields[agent].unwrapped.infos['Nue'][-1]}")
                print(f"{agent}'s Yield is {self.env.fields[agent].unwrapped.infos['Yield'][-1]}")
                print(f"{agent}'s NamountSO is {self.env.fields[agent].unwrapped.infos['NamountSO'][-1]}")

                self.assertNotEquals(self.env.fields[agent].unwrapped.infos['Nue'][-1], 0.0)


class TestEnvPlotter(unittest.TestCase):
    def setUp(self):
        self.env = MultiFieldEnv(
                    warm_up=0,
                    training=True
                )

    def test_episode(self):

        self.env.reset(options={'year': np.random.choice(range(1951, 2024))})

        infos = {}
        for agent in self.env.agent_iter():
            _, _, _, _, info = self.env.last()
            action = self.env.sample_masked_action(agent)
            if self.env.terminations[agent]:
                infos[agent] = info
                self.env.step(None)
            else:
                self.env.step(action)

        plot_results(
            infos,
            save_path=os.path.join(_DEFAULT_PLOTDIR, 'test_episode.png'),
        )

        self.env.render()

        self.assertTrue(True)

    def test_episode_farmers_practice(self):

        self.env.reset(options={'year': np.random.choice(range(1951, 2024))})

        infos = {}
        for agent in self.env.agent_iter():
            _, _, _, _, info = self.env.last()
            action = self.env.farmers_practice(agent, info)
            if self.env.terminations[agent]:
                infos[agent] = info
            if self.env.terminations[agent]:
                infos[agent] = info
                self.env.step(None)
            else:
                self.env.step(action)

        plot_results(
            infos,
            save_path=os.path.join(_DEFAULT_PLOTDIR, 'test_episode.png'),
        )

        self.env.render()

        self.assertTrue(True)

class TestAgentOrder(unittest.TestCase):
    def setUp(self):
        self.env = MultiFieldEnv(
                warm_up=0,
        )

    @staticmethod
    def is_in_canonical_order(seq, order) -> bool:
        """Return True if `seq` is a subsequence of `order` (strictly increasing positions)."""
        pos = {v: i for i, v in enumerate(order)}
        # All elements must exist in the canonical order
        try:
            idxs = [pos[x] for x in seq]
        except KeyError:
            return False
        # Must be strictly increasing (no reordering or repeats)
        return all(a < b for a, b in zip(idxs, idxs[1:]))

    def test_episode(self):
        agent_order = self.env.possible_agents

        self.env.reset(options={'year': np.random.choice(range(1951, 2024))})

        for agent in self.env.agent_iter():

            obs, rew, term, trunc, info = self.env.last()

            action = self.env.sample_masked_action(agent)
            if self.env.terminations[agent]:
                self.env.step(None)
            else:
                self.env.step(action)

            print(self.env._agent_selector.agent_order)

            self.assertTrue(self.is_in_canonical_order(self.env._agent_selector.agent_order, agent_order))

class TestAllocationEnv(unittest.TestCase):
    def setUp(self):
        self.bandit_env = AllocationBandit(
            warm_up_eps=2,
        )

    def test_init(self):
        context, info = self.bandit_env.reset()

        print(f"Context: {context}")
        print(f"Info: {info}")

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
