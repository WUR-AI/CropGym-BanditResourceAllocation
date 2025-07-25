import unittest
import datetime
from copy import deepcopy

import cropgymzoo  # for gym make
import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt

import pcse

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.helper_for_unit_tests import run_aec_till_terminate
from cropgymzoo.utils.domain_randomizer import NoisyOpenMeteo


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

    def test_action_mask_singular(self):
        obs, info = self.env.reset(options={'year': np.random.choice(range(1951, 2025))})
        self.env.unwrapped.set_budget(200)

        self.env.step(18)

        action = self.env.unwrapped.sample_masked_action()

        self.assertIn(action, [0, 1, 2])

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

        a = 0.6

        plt.plot(info_train['Date'], info_train['Yield'], label='train yield', alpha=a)
        plt.plot(info_test['Date'], info_test['Yield'], label='test yield', alpha=a)

        plt.plot(info_train['Date'], info_train['TAGP'], label='train TAGP', alpha=a)
        plt.plot(info_test['Date'], info_test['TAGP'], label='test TAGP', alpha=a)

        plt.legend()

        plt.show()

        self.assertNotEqual(obs_train, obs_test)


class TestMultiEnvFunctions(unittest.TestCase):
    def setUp(self):
        self.env = MultiFieldEnv(
            warm_up=0,
        )

    def test_reset_multi(self):
        self.env.reset(options={'year': 2010})
        obs, rew, term, trunc, info = self.env.last()

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

    def test_step_multi(self):
        self.env.reset(options={'year': 2010})

        obs, rew, term, trunc, info = self.env.last()

        self.env.step(None) \
            if (term or trunc) \
            else self.env.step(np.random.choice(range(7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

    def test_render_multi(self):
        self.env.reset(options={'year': 2010})
        obs, rew, term, trunc, info = self.env.last()

        self.env.step(None) \
            if (term or trunc) \
            else self.env.step(np.random.choice(range(7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))

        self.env.render()

        self.assertEqual(isinstance(obs, dict), True)
        self.assertEqual(isinstance(info, dict), True)

    def test_render_multi_end(self):
        self.env.reset(options={'year': 2010})

        self.env, cumulative_step_rewards, running_sum = run_aec_till_terminate(self.env)

        self.env.render()

        self.assertIn("Farm", self.env.__str__())

class TestMultiEnvTraining(unittest.TestCase):
    def setUp(self):
        self.env = MultiFieldEnv(
            warm_up=0,
            random_budget=True,
        )

    def test_multi_action_mask(self):
        self.env.reset(options={'year': np.random.choice(self.env.years)})

        print(f"Budgets: {[self.env.get_per_parcel_budget(a) for a in self.env.unwrapped.possible_agents]}")
        print(f"Total: {self.env._get_global_budget()}")

        terms = {agent: False for agent in self.env.unwrapped.possible_agents}

        for agent in self.env.agent_iter():
            obs, rew, term, trunc, info = self.env.last()
            is_last = (agent == self.env.agents[-1])

            action = np.random.choice(range(7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

            self.env.step(None) \
                if (term or trunc) \
                else self.env.step(action)

            budget_left = self.env.get_per_parcel_budget(agent)
            action_mask = self.env._get_mask(agent)

            if budget_left <= 0:
                self.assertTrue(
                    np.array_equal(action_mask,
                    np.array([True, True, True, True, True, False, False, False, False])))


class TestMultiEnvWarmUp(unittest.TestCase):
    def setUp(self):
        self.env = MultiFieldEnv(
            warm_up=10,
        )

    def test_warm_multi(self):
        self.env.reset(options={'year': 2010})

        print(next(iter(self.env.warm_up_infos.values()))['field-2']['Naction'])

        self.assertNotEqual(self.env.warm_up_infos, None)


class TestWeatherFunctions(unittest.TestCase):
    def setUp(self):
        self.noisy_wdp = NoisyOpenMeteo(*(52.512, 5.545))
        self.normal_wdp = NoisyOpenMeteo(*(52.512, 5.545))

    def test_weather(self):
        start_date = datetime.date(1951, 1, 1)


        for i in range(500):
            all_noisy_tmin = []
            all_normal_tmin = []
            all_noisy_vap = []
            all_normal_vap = []
            print(f"Iteration {i}")
            for current in range(7):
                current_date = start_date + datetime.timedelta(days=current)

                noisy = self.noisy_wdp(current_date)
                all_noisy_tmin.append(noisy.TMIN)
                all_noisy_vap.append(noisy.VAP)
                normal = self.normal_wdp(current_date)
                all_normal_tmin.append(normal.TMIN)
                all_normal_vap.append(normal.VAP)
            print(f"TMIN Noisy: {all_noisy_tmin}")
            print(f"TMIN Normal: {all_normal_tmin}\n")

            print(f"VAP Noisy: {all_noisy_vap}")
            print(f"VAP Normal: {all_normal_vap}")

            self.assertNotEquals(all_noisy_tmin, all_normal_tmin)
            self.assertNotEquals(all_noisy_vap, all_normal_vap)

class TestParameterPerturber(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("field-1", training=False)

    def test_consistency(self):
        env = self.env

        env = self.run_env_episode(env, perturb=False)

        dvs_normal_1 = env.unwrapped.infos['DVS']
        dvs_changed = []
        dvs_original = []

        for i in range(40):
            print(f"Iteration {i}")
            env = self.run_env_episode(env, perturb=False if i != 20 else True)

            dates_normal = env.unwrapped.infos['Date']
            dvs_normal = env.unwrapped.infos['DVS']
            plt.plot(dates_normal, dvs_normal, label="Normal")
            plt.show()

            if i == 19:
                dvs_original = dvs_normal
            if i == 20:
                print('stop here')
            if i == 21:
                dvs_changed = dvs_normal
            self.assertEqual(dvs_normal, dvs_changed if i > 20 else dvs_normal_1) if i != 20 else self.assertNotEqual(dvs_normal, dvs_normal_1)
            if i > 21:
                self.assertEqual(dvs_original, dvs_normal)

            dvs_normal_1 = dvs_normal

    def test_parameter_perturber(self):
        env = self.env

        for i in range(30):
            print(f"iteration {i}")
            env = self.run_env_episode(env, perturb=False)

            infos_normal = env.unwrapped.infos
            dvs_normal = infos_normal['DVS']
            dates_normal = infos_normal['Date']

            env = self.run_env_episode(env, perturb=True)

            infos_perturbed = env.unwrapped.infos
            dvs_perturbed = infos_perturbed['DVS']
            dates_perturbed = infos_perturbed['Date']

            dvs_normal, dvs_perturbed = self.align_length(dvs_normal, dvs_perturbed)
            dates_normal, dates_perturbed = self.align_length(dates_normal, dates_perturbed, dates=True)

            plt.plot(dates_normal, dvs_normal, label="Normal")
            plt.plot(dates_normal, dvs_perturbed, label="Perturbed")
            plt.legend()
            plt.show()

            self.assertNotEquals(dvs_perturbed, dvs_normal)

    def align_length(self, a: list, b: list, dates=False) -> tuple:
        if not len(a) == len(b):
            if len(a) > len(b):
                b = b + [b[-1]] if not dates else b + [b[-1] + datetime.timedelta(days=1)]
                a, b = self.align_length(a, b, dates=dates)
            else:
                a = a + [a[-1]] if not dates else a + [a[-1] + datetime.timedelta(days=1)]
                a, b = self.align_length(a, b, dates=dates)
        return a, b

    @staticmethod
    def run_env_episode(env: gym.Env, perturb: bool = False):
        if perturb:
            env.unwrapped.training = True
        else:
            env.unwrapped.training = False
        env.reset(options={'year': 2010})

        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(0)

        return env

if __name__ == '__main__':
    unittest.main()
