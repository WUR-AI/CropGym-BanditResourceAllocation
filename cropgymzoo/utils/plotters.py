import os
import numpy as np
import matplotlib.pyplot as plt

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.defaults import get_default_plot_vars
from cropgymzoo.envs.multi_field_env import MultiFieldEnv

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

def plot_results(
        infos: dict,
        variable_list: list = get_default_plot_vars(),
        cmap_str: str = "tab10",
        save_path: str = None,
        dpi: int = 300,
        show: bool = True,
):
    agents = list(infos.keys())
    agent_crops = [
        infos[agent]['CropName'][-1]
        for agent in agents
    ]

    # fig = plt.figure(figsize=(12, 10))

    # subfig = fig.subfigures(1, 2, width_ratios=[2, 1])

    fig, axes = plt.subplots(
        nrows=len(variable_list),
        ncols=1,
        figsize=(12, 10),
        constrained_layout=True,
    )

    for i, (agent, crop) in enumerate(zip(agents, agent_crops)):
        for j, variable in enumerate(variable_list):
            color = plt.get_cmap(cmap_str)
            axes[j].text(
                0.015, 0.85, variable,
                transform=axes[j].transAxes,
                va="top", ha="left",
                bbox=dict(facecolor="white",
                          edgecolor="lightgrey",
                          boxstyle="round,pad=0.25")
            )
            if variable in ['RAIN']:
                plot_function = axes[j].bar
            if variable in ['Action']:
                # [val if val != 0.0 else np.nan for val in infos[agent][variable]]
                infos[agent][variable] = [val * 10 for val in infos[agent][variable]]
                plot_function = axes[j].scatter
                axes[j].set_ylim(10, 110)  # <-- set Y limit for Action
                axes[j].set_yticks(range(10, 100, 20))
            else:
                plot_function = axes[j].plot
            plot_function(
                infos[agent]['Date'],
                np.array(infos[agent][variable]),
                label=f"{agent} - {crop}",
                color=color(i),
            )

    # axes.legend()
    # Add legends to each subplot
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # keep only the first occurrence of each label
    by_label = {k: v for k, v in sorted(dict(zip(labels, handles)).items())}

    fig.legend(
        by_label.values(), by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),  # y < 0 ⇒ place *below* the figure
        ncol=min(3, len(by_label)),  # wrap into rows if many entries
        frameon=False,
        # bbox_transform=fig.transFigure
    )

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    if save_path is not None:
        if not save_path.endswith(".png"):
            save_path += ".png"
        plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()

    return fig