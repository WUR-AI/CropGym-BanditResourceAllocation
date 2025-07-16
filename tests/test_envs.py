import unittest

import gymnasium as gym
import pettingzoo

import numpy as np

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.envs.worker_env import ParallelRLWorkers

from cropgymzoo.utils.wrappers import VecNormObs

try:
    from tianshou.env import PettingZooEnv, SubprocVectorEnv
except ImportError:
    tianshou = None

class TestCreationSingularEnv(unittest.TestCase):
    def test_create_singular_env(self):

        env = gym.make('field-1')

        self.assertEqual(isinstance(env.unwrapped, ParcelEnv), True)  # add assertion here

class TestEnvWrappers(unittest.TestCase):
    def setUp(self):
        self.env = PettingZooEnv(
                ParallelRLWorkers(
                    warm_up=0,
                    training=True
                )
            )

        self.venv_train = SubprocVectorEnv(
            [lambda: PettingZooEnv(
                ParallelRLWorkers(
                    warm_up=0,
                    training=True
                )
            ) for _ in range(2)]
        )

        self.venv_test = SubprocVectorEnv(
            [lambda: PettingZooEnv(
                ParallelRLWorkers(
                    warm_up=0,
                    training=False
                )
            ) for _ in range(1)]
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




if __name__ == '__main__':
    unittest.main()
