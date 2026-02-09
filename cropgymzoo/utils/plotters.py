import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.defaults import get_default_plot_vars
from cropgymzoo.envs.multi_field_env import MultiFieldEnv

def plot_year(infos: dict, var: str = "Reward"):
    agents = list(infos.keys())

    for i, agent in enumerate(agents):
        color = plt.get_cmap('tab10')
        plt.plot(
            infos[agent]['Date'],
            np.cumsum(infos[agent]['Reward']),
            label=f"{agent}",
            color=color(i)
        )
        min_date = min(infos[agent]['Date'][0] for agent in agents)
        plt.vlines(
            infos[agent]['Date'],
            np.zeros(len(infos[agent]['Action'])),
            [i / 10 for i in infos[agent]['Action']],
            color=color(i), alpha=0.3
        )

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
            elif variable in ['Action']:
                y = np.array(infos[agent][variable], dtype=float) * 10.0
                plot_function = axes[j].scatter
                axes[j].set_ylim(10, 110)  # <-- set Y limit for Action
                axes[j].set_yticks(range(10, 100, 20))
            elif variable in ['Reward']:
                y = np.cumsum(np.array(infos[agent][variable], dtype=float))
                plot_function = axes[j].step
            else:
                plot_function = axes[j].plot
            plot_function(
                infos[agent]['Date'],
                y if 'y' in locals() else np.array(infos[agent][variable]),
                label=f"{agent} - {crop}",
                color=color(i),
            )
            if 'y' in locals():
                del y

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


# --- Daisy-chained multi-season helpers and plotting ---

def _season_concat_series(
        info_by_year: dict,
        agent: str,
        key: str,
        *,
        reward_cumsum_per_season: bool = True,
        action_scale_10: bool = True,
):
    """Concatenate a per-agent time series over multiple season-years.

    - Inserts NaN gaps between seasons so matplotlib breaks lines.
    - For 'Reward', optionally computes cumulative sum *within each season*.
    - For 'Action', optionally scales by 10 to match kg/ha convention.

    Parameters
    ----------
    info_by_year: dict[int -> dict]
        Mapping season_year -> AgentInfos, where AgentInfos is dict[agent -> info_dict].
    agent: str
        Agent key, e.g. 'field-1'.
    key: str
        Info key, e.g. 'DVS', 'Profit', 'Reward', 'Action', ...

    Returns
    -------
    x_dates: list[datetime.date]
    y_vals: np.ndarray
    """

    years_sorted = sorted([int(y) for y in info_by_year.keys()])

    x_dates: list = []
    y_vals: list = []

    for yi, y in enumerate(years_sorted):
        agent_infos = info_by_year[y].get(agent, {})
        dates = agent_infos.get('Date', [])
        seq = agent_infos.get(key, [])

        if not dates or seq is None:
            continue

        # Ensure same length
        n = min(len(dates), len(seq))
        dates = list(dates)[:n]
        seq = list(seq)[:n]

        # Transform
        if key == 'Reward' and reward_cumsum_per_season:
            arr = np.cumsum(np.asarray(seq, dtype=float))
        else:
            # allow mixed types; coerce where possible
            try:
                arr = np.asarray(seq, dtype=float)
            except Exception:
                arr = np.asarray(seq, dtype=object)

        if key == 'Action' and action_scale_10:
            try:
                arr = np.asarray(arr, dtype=float) * 10.0
            except Exception:
                pass

        # Append season chunk
        x_dates.extend(dates)
        y_vals.extend(list(arr))

        # Add a gap between seasons to break the line
        if yi < len(years_sorted) - 1:
            # Insert one-day gap marker (date itself doesn't matter much)
            try:
                last_d = dates[-1]
                gap_d = last_d + datetime.timedelta(days=1)
            except Exception:
                gap_d = dates[-1]
            x_dates.append(gap_d)
            y_vals.append(np.nan)

    return x_dates, np.asarray(y_vals)


def plot_results_daisy_chained(
        infos_by_year: dict,
        variable_list: list = None,
        cmap_str: str = "tab10",
        save_path: str = None,
        dpi: int = 300,
        show: bool = True,
        *,
        reward_cumsum_per_season: bool = True,
        action_scale_10: bool = True,
        season_markers: bool = True,
        season_marker_label: bool = True,
        season_marker_every_axis: bool = True,
        season_crop_labels: bool = True,
        season_crop_label_with_year: bool = True,
        season_crop_label_every_axis: bool = False,
):
    """Plot daisy-chained multi-season evaluation in ONE figure.

    Expected input structure:
        infos_by_year[season_year] = AgentInfos
        AgentInfos[agent] = { 'Date': [...], 'DVS': [...], ... }

    This function concatenates the time series per agent across years and inserts NaN
    gaps between seasons to visually separate them.
    """

    if variable_list is None:
        variable_list = get_default_plot_vars()

    years_sorted = sorted([int(y) for y in infos_by_year.keys()])
    if not years_sorted:
        raise ValueError("infos_by_year is empty")

    # Determine agent set from the first year
    first_year = years_sorted[0]
    agents = list(infos_by_year[first_year].keys())

    # Use agent label only (stable legend), and build per-season crop lookup
    agent_label = {ag: str(ag) for ag in agents}
    crop_by_year_agent = {}
    for y in years_sorted:
        crop_by_year_agent[int(y)] = {}
        for ag in agents:
            cseq = infos_by_year.get(int(y), {}).get(ag, {}).get('CropName', None)
            crop = None
            if cseq:
                # prefer last non-empty crop string
                for v in reversed(cseq):
                    if v is not None and str(v) not in ("", "nan", "None"):
                        crop = str(v)
                        break
            crop_by_year_agent[int(y)][ag] = crop if crop is not None else "–"

    fig, axes = plt.subplots(
        nrows=len(variable_list),
        ncols=1,
        figsize=(12, 10),
        constrained_layout=True,
    )

    if len(variable_list) == 1:
        axes = [axes]

    def _get_season_bounds(season_year: int):
        """Return (start_date, end_date) for a season across all agents, or (None, None)."""
        start_d = None
        end_d = None
        try:
            per_year = infos_by_year.get(int(season_year), {})
        except Exception:
            per_year = {}

        for ag2 in agents:
            dates = per_year.get(ag2, {}).get("Date", [])
            if not dates:
                continue
            # guard against non-date entries
            try:
                d0 = dates[0]
                d1 = dates[-1]
            except Exception:
                continue
            if d0 is not None:
                if start_d is None or d0 < start_d:
                    start_d = d0
            if d1 is not None:
                if end_d is None or d1 > end_d:
                    end_d = d1
        return start_d, end_d

    for i, ag in enumerate(agents):
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

            x_dates, y = _season_concat_series(
                infos_by_year,
                ag,
                variable,
                reward_cumsum_per_season=reward_cumsum_per_season,
                action_scale_10=action_scale_10,
            )

            if variable in ['RAIN']:
                plot_function = axes[j].bar
            elif variable in ['Action']:
                plot_function = axes[j].scatter
                axes[j].set_ylim(10, 110)
                axes[j].set_yticks(range(10, 100, 20))
            elif variable in ['Reward']:
                plot_function = axes[j].step
            else:
                plot_function = axes[j].plot

            plot_function(
                x_dates,
                y,
                label=f"{agent_label.get(ag, ag)}",
                color=color(i),
            )

    # Optional: draw season boundary markers (start/end) for each season-year
    if season_markers:
        for y in years_sorted:
            s0, s1 = _get_season_bounds(int(y))
            if s0 is None and s1 is None:
                continue

            # choose which axes to mark
            axes_to_mark = axes if season_marker_every_axis else [axes[0]]

            for ax in axes_to_mark:
                if s0 is not None:
                    ax.axvline(s0, linestyle="--", linewidth=0.8, alpha=0.25)
                if s1 is not None:
                    ax.axvline(s1, linestyle=":", linewidth=0.8, alpha=0.25)

            # Optional label only on the first axis to avoid clutter
            if season_marker_label and s0 is not None:
                try:
                    axes[0].annotate(
                        str(int(y)),
                        xy=(s0, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(2, -2),
                        textcoords="offset points",
                        va="top",
                        ha="left",
                        rotation=90,
                        fontsize=8,
                        alpha=0.7,
                    )
                except Exception:
                    pass

    # Optional: annotate crop name per season segment (helps interpret rotations)
    def _get_agent_season_start(season_year: int, agent: str):
        try:
            dates = infos_by_year.get(int(season_year), {}).get(agent, {}).get("Date", [])
            return dates[0] if dates else None
        except Exception:
            return None
    if season_crop_labels:
        axes_to_mark = axes if season_crop_label_every_axis else [axes[0]]
        for y in years_sorted:
            for i, ag in enumerate(agents):
                s0 = _get_agent_season_start(int(y), ag)
                if s0 is None:
                    continue
                crop = crop_by_year_agent.get(int(y), {}).get(ag, "–")
                txt = f"{y}:{crop}" if season_crop_label_with_year else f"{crop}"
                for ax in axes_to_mark:
                    try:
                        ax.annotate(
                            txt,
                            xy=(s0, 0.02),
                            xycoords=("data", "axes fraction"),
                            xytext=(2, 2),
                            textcoords="offset points",
                            ha="left",
                            va="bottom",
                            fontsize=7,
                            alpha=0.75,
                        )
                    except Exception:
                        pass

    # Build a single legend under the figure
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    by_label = {k: v for k, v in sorted(dict(zip(labels, handles)).items())}

    fig.legend(
        by_label.values(), by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=min(3, len(by_label)),
        frameon=False,
    )

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    if save_path is not None:
        if not save_path.endswith(".png"):
            save_path += ".png"
        plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()

    return fig


def plot_nue_template(ax=None, label=True, size=(4, 4), max=300, get_return=True) -> plt:
    l = max

    n_in = np.linspace(0, l, l)

    # For NUE = 50%, the output is the same as the input
    n_output_50 = n_in * 0.5

    # For NUE = 90%, the output is 90% of the input
    n_output_90 = n_in * 0.9

    max_surplus_line = n_in - 40

    # max_surplus_line[max_surplus_line < 0] = 0

    if not ax:
        plt.figure(figsize=size)

        plt.plot(n_in, max_surplus_line, 'k--', linewidth=0.5, zorder=1)  #, label='N surplus = 40 kg/ha/yr')

        plt.fill_between(n_in, n_output_50, max_surplus_line, color='grey',
                         alpha=0.5)  #, label='N surplus > 40 kg/ha/yr')

        plt.plot(n_in, n_output_50, 'k-', linewidth=0.5)  #, label='NUE = 50%')

        plt.plot(n_in, n_output_90, 'k-', linewidth=0.5)  #, label='NUE = 90%')

        # plt.plot(n_in, min_productivity_output, 'k-.', label='Desired minimum productivity (N output > 80 kg/ha/yr)')

        plt.fill_between(n_in, 0, n_output_50, color='lightcoral', zorder=2)
        # label='NUE very low (NUE < 50%): Risk of inefficient N use')

        plt.fill_between(n_in, n_output_90, l, color='lightcoral', zorder=2)
        # label='NUE very high (NUE > 90%): Risk of soil mning')
        if label:
            plt.xlabel('Input', size=8)
            plt.ylabel('Output', size=8)
        plt.tick_params(axis='both', which='major', labelsize=7)
        plt.ylim(0, 300)
        plt.xlim(0, 300)
        plt.tight_layout()
        if get_return:
            return plt
