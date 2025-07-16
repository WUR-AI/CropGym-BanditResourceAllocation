import os
import numpy as np
import matplotlib.pyplot as plt

from cropgymzoo.envs.worker_env import ParallelRLWorkers

def plot_year(infos: dict, var: str = "Reward"):
    agents = list(infos.keys())

    for i, agent in enumerate(agents):
        color = plt.get_cmap('tab10')
        plt.plot(infos[agent]['Date'], np.cumsum(infos[agent]['Reward']),
                 label=f"{agents.name}, "
                       f"{agents.crop}",
                 color=color(i))
        min_date = min(infos[agent]['Date'][0] for agent in agents)
        plt.vlines(infos[agent]['Date'], np.zeros(len(infos[agent]['Action'])),
                   [i / 10 for i in infos[agent]['Action']], color=color(i), alpha=0.3)

    # date_range = [min_date + datetime.timedelta(days=i) for i in range(len(cumulative_rewards))]
    # plt.plot(date_range, np.cumsum(cumulative_rewards))

    plt.legend()
    plt.show()