import unittest

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

import datetime

import cropgymzoo  # for gym make
import gymnasium as gym

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils import rewards

from cropgymzoo.utils.helper_for_unit_tests import run_aec_till_terminate, run_aec_step
from cropgymzoo.utils.plotters import plot_results


class TestSingularRewardFunctions(unittest.TestCase):
    def setUp(self):
        self.env_nue = gym.make('field-sb-s', reward='NUE')
        self.env_pny_1 = gym.make('field-sb-s', reward='PNY')
        self.env_pny_2 = gym.make('field-pt-s', reward='PNY')
        self.env_pny_3 = gym.make('field-ww-s', reward='PNY')

        self.env_pnb_1 = gym.make('field-sb-s', reward='PNB')
        self.env_pnb_2 = gym.make('field-pt-s', reward='PNB')
        self.env_pnb_3 = gym.make('field-ww-s', reward='PNB')

        self.env_pnr_1 = gym.make('field-sb-s', reward='PNR')
        self.env_pnr_train = gym.make('field-sb-s', reward='PNR', training=True)

        self.year_dict = {'year': 2000}

    def test_pnr(self):
        _, info = self.env_pnr_1.reset(options=self.year_dict)
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env_pnr_1.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

        _, info = self.env_pnr_1.reset(options={'year': 2010})
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env_pnr_1.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

    def test_pnr_training(self):
        _, info = self.env_pnr_train.reset(options=self.year_dict)
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env_pnr_train.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

        # once again for domain repeat
        _, info = self.env_pnr_train.reset(options=self.year_dict)
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env_pnr_train.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

        # try curriculum advance
        self.env_pnr_train.unwrapped.random_manager.stage = 3
        self.env_pnr_train.unwrapped.domain_repeat_left = 0

        _, info = self.env_pnr_train.reset(options={'year': 1990})
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env_pnr_train.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

        self.env_pnr_train.unwrapped.random_manager.stage = 4
        self.env_pnr_train.unwrapped.domain_repeat_left = 0

        _, info = self.env_pnr_train.reset(options={'year': 1995})
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env_pnr_train.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

        _, info = self.env_pnr_train.reset(options={'year': 1995})
        terminated = False
        rewards = []
        for _ in range(3):
            _ = self.env_pnr_train.step(0)
        _, self.env_pnr_train.step(10)
        while not terminated:
            _, reward, terminated, _, info = self.env_pnr_train.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(True)





    def test_nue(self):
        # crop sugarbeets
        _, info = self.env_nue.reset(options=self.year_dict)
        self.env_nue.unwrapped.set_budget(200)

        for _ in range(5):
            _, reward, terminated, _, info = self.env_nue.step(0)
        _, reward, terminated, _, info = self.env_nue.step(8)
        _, reward, terminated, _, info = self.env_nue.step(3)
        while not terminated:
            _, reward, terminated, _, info = self.env_nue.step(0)

        self.assertTrue(0 <= reward <= 1)

    def test_pny_beets1(self):
        # crop sugarbeets
        _, info = self.env_pny_1.reset(options=self.year_dict)
        self.env_pny_1.unwrapped.set_budget(200)

        rewards = []

        for _ in range(5):
            _, reward, terminated, _, info = self.env_pny_1.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_1.step(8)
        rewards.append(reward)
        print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_1.step(3)
        rewards.append(reward)
        print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        while not terminated:
            _, reward, terminated, _, info = self.env_pny_1.step(0)
            rewards.append(reward)
            print(f"reward in step {self.env_pny_1.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")

        lo, hi = self.env_pny_1.unwrapped.reward_class.reward_bounds()

        print(lo, hi)

        self.assertTrue(lo <= np.sum(rewards) <= hi)

    def test_pny_beets2(self):
        # crop sugarbeets
        _, info = self.env_pny_2.reset(options=self.year_dict)
        self.env_pny_1.unwrapped.set_budget(200)

        rewards = []
        y = []

        for _ in range(5):
            _, reward, terminated, _, info = self.env_pny_2.step(0)
            rewards.append(reward)
            y.append(self.env_pny_2.unwrapped.get_latest_info("Yield"))
            print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_2.step(8)
        rewards.append(reward)
        y.append(self.env_pny_2.unwrapped.get_latest_info("Yield"))
        print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        _, reward, terminated, _, info = self.env_pny_2.step(3)
        y.append(self.env_pny_2.unwrapped.get_latest_info("Yield"))
        rewards.append(reward)
        print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        while not terminated:
            _, reward, terminated, _, info = self.env_pny_2.step(0)
            y.append(self.env_pny_2.unwrapped.get_latest_info("Yield"))
            rewards.append(reward)
            print(f"reward in step {self.env_pny_2.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")

        lo, hi = self.env_pny_2.unwrapped.reward_class.reward_bounds()

        print(lo, hi)
        plt.plot(y)
        plt.show()

        self.assertTrue(lo <= np.sum(rewards) <= hi)

    def test_pny_wheat(self):
        # crop sugarbeets
        _, info = self.env_pny_3.reset(options=self.year_dict)
        self.env_pny_1.unwrapped.set_budget(200)

        rewards = []
        y = []

        for _ in range(20):
            _, reward, terminated, _, info = self.env_pny_3.step(0)
            rewards.append(reward)
            y.append(self.env_pny_3.unwrapped.get_latest_info("Yield"))
            print(f"reward in step {self.env_pny_3.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        for _ in range(1):
            _, reward, terminated, _, info = self.env_pny_3.step(24)
            rewards.append(reward)
            y.append(self.env_pny_3.unwrapped.get_latest_info("Yield"))
            print(f"reward in step {self.env_pny_3.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")
        while not terminated:
            _, reward, terminated, _, info = self.env_pny_3.step(0)
            y.append(self.env_pny_3.unwrapped.get_latest_info("Yield"))
            rewards.append(reward)
            print(f"reward in step {self.env_pny_3.unwrapped.n_steps} is {reward} and cumulative is {np.sum(rewards)}")

        print(f"NUE: {self.env_pny_3.unwrapped.get_latest_info('Nue')}, Nsurp: {self.env_pny_3.unwrapped.get_latest_info('Nsurp')}")

        lo, hi = self.env_pny_3.unwrapped.reward_class.reward_bounds()

        plt.plot(np.cumsum(rewards))

        plt.show()

        print(f"End yield: {self.env_pny_3.unwrapped.get_latest_info('Yield')}")
        print(f"Total fertilized: {self.env_pny_3.unwrapped.get_latest_info('Naction')}")
        print(f"Profit: {self.env_pny_3.unwrapped.get_latest_info('Profit')}")

        print(lo, hi)
        plt.plot(y)
        plt.show()

        self.assertTrue(lo <= np.sum(rewards) <= hi)

    def test_pnb_sugarbeet(self):
        _, info = self.env_pnb_1.reset(options=self.year_dict)

        term = False
        while not term:
            _, reward, term, _, info = self.env_pnb_1.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(True) #sum_reward <= 0)

        _, info = self.env_pnb_1.reset(options=self.year_dict)

        for _ in range(8):
            _, reward, term, _, info = self.env_pnb_1.step(0)
        _, reward, term, _, info = self.env_pnb_1.step(6)
        _, reward, term, _, info = self.env_pnb_1.step(0)
        _, reward, term, _, info = self.env_pnb_1.step(8)
        while not term:
            _, reward, term, _, info = self.env_pnb_1.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(sum_reward > 0)

        _, info = self.env_pnb_1.reset(options=self.year_dict)

        for _ in range(8):
            _, reward, term, _, info = self.env_pnb_1.step(0)
        _, reward, term, _, info = self.env_pnb_1.step(6)
        _, reward, term, _, info = self.env_pnb_1.step(0)
        _, reward, term, _, info = self.env_pnb_1.step(10)
        while not term:
            _, reward, term, _, info = self.env_pnb_1.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(sum_reward > 0)

    def test_pnb_wheat(self):
        _, info = self.env_pnb_2.reset(options=self.year_dict)

        term = False
        while not term:
            _, reward, term, _, info = self.env_pnb_2.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")
        print(f"Budget Left: {info['BudgetLeft'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(True)#sum_reward <= 0)

        _, info = self.env_pnb_2.reset(options=self.year_dict)

        for _ in range(8):
            _, reward, term, _, info = self.env_pnb_2.step(0)
        _, reward, term, _, info = self.env_pnb_2.step(6)
        _, reward, term, _, info = self.env_pnb_2.step(0)
        _, reward, term, _, info = self.env_pnb_2.step(8)
        while not term:
            _, reward, term, _, info = self.env_pnb_2.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")
        print(f"Budget Left: {info['BudgetLeft'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(sum_reward > 0)

        _, info = self.env_pnb_2.reset(options=self.year_dict)

        for _ in range(8):
            _, reward, term, _, info = self.env_pnb_2.step(0)
        _, reward, term, _, info = self.env_pnb_2.step(6)
        _, reward, term, _, info = self.env_pnb_2.step(0)
        _, reward, term, _, info = self.env_pnb_2.step(16)
        while not term:
            _, reward, term, _, info = self.env_pnb_2.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")
        print(f"Budget Left: {info['BudgetLeft'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(sum_reward > 0)

    def test_pnb_potato(self):
        _, info = self.env_pnb_3.reset(options=self.year_dict)

        term = False
        while not term:
            _, reward, term, _, info = self.env_pnb_3.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")
        print(f"Budget Left: {info['BudgetLeft'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(True) #sum_reward <= 0)

        _, info = self.env_pnb_3.reset(options=self.year_dict)

        for _ in range(8):
            _, reward, term, _, info = self.env_pnb_3.step(0)
        _, reward, term, _, info = self.env_pnb_3.step(6)
        _, reward, term, _, info = self.env_pnb_3.step(0)
        _, reward, term, _, info = self.env_pnb_3.step(8)
        while not term:
            _, reward, term, _, info = self.env_pnb_3.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")
        print(f"Budget Left: {info['BudgetLeft'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(sum_reward > 0)

        _, info = self.env_pnb_3.reset(options=self.year_dict)

        for _ in range(8):
            _, reward, term, _, info = self.env_pnb_3.step(0)
        _, reward, term, _, info = self.env_pnb_3.step(6)
        _, reward, term, _, info = self.env_pnb_3.step(2)
        _, reward, term, _, info = self.env_pnb_3.step(0)
        while not term:
            _, reward, term, _, info = self.env_pnb_3.step(0)

        reward = info["Reward"]

        print(f"Fertilizer price: {info['FertilizerPrice'][-1]}, Crop price: {info['CropPrice'][-1]}")
        print(f"Reward: {reward}")
        print(f"NUE: {info['Nue'][-1]}, Nsurp {info['Nsurp'][-1]} ")
        print(f"Profit: {info['Profit'][-1]}")
        print(f"Budget Left: {info['BudgetLeft'][-1]}")

        plt.plot(info["Date"], np.cumsum(info["Reward"]))
        plt.show()

        plt.plot(info["Date"], info["NAVAIL"])
        plt.show()


        sum_reward = sum(info["Reward"])
        print(f"Sum reward: {sum_reward}")

        self.assertTrue(sum_reward > 0)


class TestSingularRewardFunctionsMPN(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('field-sb-s', reward='MPN', training=True)


    def test_reward_mpn(self):
        _, info = self.env.reset(options={'year': 2000})
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

        _, info = self.env.reset(options={'year': 2010})
        terminated = False
        rewards = []
        while not terminated:
            _, reward, terminated, _, info = self.env.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(np.sum(rewards) == 0)

        _, info = self.env.reset(options={'year': 2010})
        terminated = False
        rewards = []
        for _ in range(3):
            _ = self.env.step(0)
        _, self.env.step(10)
        while not terminated:
            _, reward, terminated, _, info = self.env.step(0)
            rewards.append(reward)

        print(f"Total rew: {np.sum(rewards)}")

        self.assertTrue(True)


class TestMultiRewardFunctionNSU(unittest.TestCase):
    def setUp(self):
        self.env_full = MultiFieldEnv(
            training=True,
            reward='NSU',
        )

    def test_reward_multi_full(self):
        self.env_full.reset(options={'year': 2020})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                print(rew)
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(0 <= np.sum(rewards[agent]) <= 1)


class TestMultiRewardFunctionPNR(unittest.TestCase):
    def setUp(self):
        self.env_full = MultiFieldEnv(
            training=True,
        )

        self.env_eval = MultiFieldEnv()

    def test_reward_multi_full(self):
        self.env_full.reset(options={'year': 2020})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)

        self.env_full.reset(options={'year': 2010})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)

        self.env_full._domain_repeat_left = 0
        self.env_full.reset(options={'year': 2010})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)


        self.env_full.reset(options={'year': 2010})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)

        self.env_full._domain_repeat_left = 0
        self.env_full.set_curriculum_stage(3)
        self.env_full.reset(options={'year': 2010})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)

        self.env_full.reset(options={'year': 2010})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)

    def test_reward_multi_eval(self):
        self.env_full.reset(options={'year': 2000})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)

        self.env_full.reset(options={'year': 2010})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)

        self.env_full.reset(options={'year': 2020})

        infos = {}
        rewards = {}
        for agent in self.env_full.unwrapped.possible_agents:
            rewards[agent] = []
        for agent in self.env_full.agent_iter():
            obs, rew, term, trunc, info = self.env_full.last()
            rewards[agent].append(rew)
            if self.env_full.terminations[agent]:
                infos[agent] = info
                self.env_full.step(None)
            else:
                self.env_full.step(0)

        for agent in self.env_full.unwrapped.possible_agents:
            print(f"Agent {agent} has cumulative reward {np.sum(rewards[agent])}")
            self.assertTrue(np.sum(rewards[agent]) == 0)


class TestMultiRewardFunction(unittest.TestCase):
    def setUp(self):
        from tianshou.env import PettingZooEnv
        self.env = MultiFieldEnv(
            warm_up=0,
        )
        self.env_training = MultiFieldEnv(
            warm_up=0,
            training=True
        )

        self.env_wrapped = PettingZooEnv(
            self.env_training,
        )

    def test_reward_area_multi(self):
        year = np.random.choice(range(1951, 2025))
        self.env.reset(options={'year': year})

        traces = defaultdict(lambda: {"Date": [], "Reward": [], "Action": []})

        env, cumulative_step_rewards, cumulative_global_rewards = run_aec_till_terminate(self.env)

        agents = self.env.unwrapped.possible_agents

        print(np.sum(cumulative_step_rewards))
        print(np.sum(cumulative_global_rewards))

        self.assertTrue(0 <= np.sum(cumulative_step_rewards) <= 1.5)

    def test_test_reward_area_multi_training(self):
        year = np.random.choice(range(1951, 2025))
        self.env_training.reset(options={'year': year})

        self.env_training, cumulative_rewards, cumulative_global_rewards = run_aec_till_terminate(self.env_training)

        agents = self.env_training.unwrapped.possible_agents

        infos = self.env_training.infos

        plot_results(infos)

        # print(infos[agents[2]]['Yield'])
        # print(cumulative_rewards)
        # print(cumulative_global_rewards)
        #
        # for i, agent in enumerate(agents):
        #     color = plt.get_cmap('tab10')
        #     plt.plot(infos[agent]['Date'], np.cumsum(infos[agent]['Reward']),
        #              label=f"{self.env_training.unwrapped.fields[agent].unwrapped.name}, "
        #                    f"{self.env_training.unwrapped.fields[agent].unwrapped.crop}",
        #              color=color(i))
        #     min_date = min(infos[agent]['Date'][0] for agent in agents)
        #     plt.vlines(infos[agent]['Date'], np.zeros(len(infos[agent]['Action'])),
        #                [i / 10 for i in infos[agent]['Action']], color=color(i), alpha=0.3)
        #
        # date_range = [min_date + datetime.timedelta(days=i) for i in range(len(cumulative_rewards))]
        # # plt.plot(date_range, np.cumsum(cumulative_global_rewards))
        #
        # plt.legend()
        # plt.show()

        print(np.sum(cumulative_rewards))
        print(infos)

        self.assertTrue(0 <= np.sum(cumulative_rewards) <= 100_000)

    def test_obs_and_reward_wrapped(self):
        obs, info = self.env_wrapped.reset(options={'year': np.random.choice(range(1951, 2023)),})

        term = False
        while not term:
            obs, rew, term, trun, info = self.env_wrapped.step(np.random.choice(range(7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
            print(info)

        self.assertTrue(np.sum(info["Reward"]) != 0.0)

if __name__ == '__main__':
    unittest.main()
