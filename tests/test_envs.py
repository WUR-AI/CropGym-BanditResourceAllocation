import unittest

import gymnasium as gym
import pettingzoo

import numpy as np
from numba.core.typing.old_builtins import Print

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.envs.multi_field_env import MultiFieldEnv

from cropgymzoo.utils.wrappers import VecNormObs, PettingZooEnvChecker

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

                self.env.step(0)

            for agent in self.env.possible_agents:
                print(f"{agent}'s crop is {self.env.fields[agent].unwrapped.crop} and start date is {self.env.fields[agent].unwrapped.date}")
                print(f"{agent}'s NUE is {self.env.fields[agent].unwrapped.infos['Nue'][-1]}")
                print(f"{agent}'s Yield is {self.env.fields[agent].unwrapped.infos['Yield'][-1]}")
                print(f"{agent}'s NamountSO is {self.env.fields[agent].unwrapped.infos['NamountSO'][-1]}")

                self.assertNotEquals(self.env.fields[agent].unwrapped.infos['Nue'][-1], 0.0)


class TestEnvWrappers(unittest.TestCase):
    def setUp(self):
        self.env = PettingZooEnvChecker(
                MultiFieldEnv(
                    warm_up=0,
                    training=True
                )
            )

        self.venv_subproc_train = SubprocVectorEnv(
            [lambda: PettingZooEnvChecker(
                MultiFieldEnv(
                    warm_up=0,
                    training=True
                )
            ) for _ in range(2)]
        )

        self.venv_subproc_test = SubprocVectorEnv(
            [lambda: PettingZooEnvChecker(
                MultiFieldEnv(
                    warm_up=0,
                    training=False
                )
            ) for _ in range(1)]
        )

        self.venv_train = DummyVectorEnv(
            [lambda: PettingZooEnvChecker(
                MultiFieldEnv(
                    warm_up=0,
                    training=True
                )
            )]
        )

        self.venv_test = DummyVectorEnv(
            [lambda: PettingZooEnvChecker(
                MultiFieldEnv(
                    warm_up=0,
                    training=False
                )
            )]
        )

    def test_norm_wrapper(self):
        train_env = VecNormObs(
            self.venv_train,
            update_obs_rms=True,
        )
        test_env = VecNormObs(
            self.venv_test,
            update_obs_rms=False,
        )

        train_env.reset(options={'year': np.random.choice(range(1951, 2024))})

        print(train_env)

        self.assertTrue(train_env is not None)

    def test_pettingzoo_tianshou_wrapper(self):
        obs, info = self.env.reset(options={'year': np.random.choice(range(1951, 2024))})

        print(obs)

        self.assertTrue(isinstance(info, dict))

        obs, rew, term, trunc, info = self.env.step(1)

        print(info)

        self.assertTrue(isinstance(info, dict))


    def test_episode(self):

        self.venv_train.reset(options={'year': np.random.choice(range(1951, 2024))})

        for agent in self.venv_train.workers[0].env.env.agent_iter():
            obs, rew, term, trunc, info = self.venv_train.last()

            if term:
                action = None
            else:
                action = self.venv_train.action_space(agent).sample()

            self.venv_train.step(action)

        self.assertTrue(True)




if __name__ == '__main__':
    unittest.main()
