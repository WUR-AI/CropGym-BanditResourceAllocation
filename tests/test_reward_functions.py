import unittest

import numpy as np

import matplotlib.pyplot as plt

import cropgymzoo  # for gym make
import gymnasium as gym

from cropgymzoo.envs.worker_env import ParallelRLWorkers


class TestSingularRewardFunctions(unittest.TestCase):
    def setUp(self):
        self.env_nue = gym.make('field-1', reward='NUE')
        self.env_pny_1 = gym.make('field-1', reward='PNY')
        self.env_pny_2 = gym.make('field-4', reward='PNY')
        self.env_pny_3 = gym.make('field-2', reward='PNY')

    def test_nue(self):
        # crop sugarbeets
        _, info = self.env_nue.reset(options={'year': 2010})
        self.env_nue.unwrapped.set_budget(200)

        for _ in range(5):
            _, reward, terminated, _, info = self.env_nue.step(0)
        _, reward, terminated, _, info = self.env_nue.step(8)
        _, reward, terminated, _, info = self.env_nue.step(3)
        while not terminated:
            _, reward, terminated, _, info = self.env_nue.step(0)

        self.assertGreaterEqual(reward, 1)

    def test_pny_beets1(self):
        # crop sugarbeets
        _, info = self.env_pny_1.reset(options={'year': 2010})
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
        _, info = self.env_pny_2.reset(options={'year': 2010})
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
        _, info = self.env_pny_3.reset(options={'year': 2015})
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

class TestMultiRewardFunction(unittest.TestCase):
    def setUp(self):
        self.env = ParallelRLWorkers(
            warm_up=0,
        )
        self.env_training = ParallelRLWorkers(
            warm_up=0,
            training=True
        )

    def test_reward_area_multi(self):
        year = np.random.choice(range(1951, 2025))
        obs, info = self.env.reset(options={'year': year})

        cumulative_rewards = []
        terminateds = {agent: False for agent in self.env.unwrapped.agents}
        while not all(terminateds.values()):
            _, rewards, terminateds, _, infos = self.env.step({
                agent: np.random.choice(range(0, 7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,])
                for agent in self.env.unwrapped.agents
            })
            # print(f"Scalar: {rewards}")
            cumulative_rewards.append(rewards)
        agents = self.env.unwrapped.possible_agents

        print(infos[agents[2]]['Yield'])

        for i, agent in enumerate(agents):
            color = plt.get_cmap('tab10')
            plt.plot(infos[agent]['Date'], np.cumsum(infos[agent]['Reward']),
                     label=f"{self.env.unwrapped.fields[agent].unwrapped.name}, "
                           f"{self.env.unwrapped.fields[agent].unwrapped.crop}",
                     color=color(i))
            plt.vlines(infos[agent]['Date'], np.zeros(len(infos[agent]['Action'])),
                       [i/10 for i in infos[agent]['Action']], color=color(i), alpha=0.3)

        plt.legend()
        plt.show()

        print(np.sum(cumulative_rewards))

        self.assertTrue(0 <= np.sum(cumulative_rewards) <= 1)

    def test_test_reward_area_multi_training(self):
        year = np.random.choice(range(1951, 2025))
        obs, info = self.env_training.reset(options={'year': year})

        cumulative_rewards = []
        terminateds = {agent: False for agent in self.env_training.unwrapped.agents}
        while not all(terminateds.values()):
            _, rewards, terminateds, _, infos = self.env_training.step({
                agent: np.random.choice(range(0, 7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, ])
                for agent in self.env_training.unwrapped.agents
            })
            # print(f"Scalar: {rewards}")
            cumulative_rewards.append(rewards)
        agents = self.env_training.unwrapped.possible_agents

        print(infos[agents[2]]['Yield'])

        for i, agent in enumerate(agents):
            color = plt.get_cmap('tab10')
            plt.plot(infos[agent]['Date'], np.cumsum(infos[agent]['Reward']),
                     label=f"{self.env_training.unwrapped.fields[agent].unwrapped.name}, "
                           f"{self.env_training.unwrapped.fields[agent].unwrapped.crop}",
                     color=color(i))
            plt.vlines(infos[agent]['Date'], np.zeros(len(infos[agent]['Action'])),
                       [i / 10 for i in infos[agent]['Action']], color=color(i), alpha=0.3)

        plt.legend()
        plt.show()

        print(np.sum(cumulative_rewards))

        self.assertTrue(0 <= np.sum(cumulative_rewards) <= 1)

if __name__ == '__main__':
    unittest.main()
