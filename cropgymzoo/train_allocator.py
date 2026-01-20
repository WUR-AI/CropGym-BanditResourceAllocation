import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

import numpy as np
import os
import matplotlib.pyplot as plt

import datetime
import pickle

from cropgymzoo.eval_allocator import run_eval_allocator
from cropgymzoo.agents.nn_agp import NNAGPBandit
from cropgymzoo.utils.agent_helpers import min_max_normalize
from cropgymzoo.utils.callbacks import _setup_bandit_comet, log_selection_info, log_model_histograms, fig_to_chw_uint8
from cropgymzoo.envs.allocation_env import AllocationBandit
from cropgymzoo import _DEFAULT_LOGDIR
from tianshou.utils.statistics import RunningMeanStd
from cropgymzoo.utils.plotters import plot_results


def farm_int_mapper(x: int):
    """
    Maps a global farm index (0–52) to (region, farmer_id).

    Regions:
        Gelderland: 0–11  (12 farms)
        Groningen: 12–24  (13 farms)
        Zeeland:   25–51  (27 farms)

    Returns:
        (region_name: str, farmer_id: int)
    """

    if not (0 <= x <= 52):
        raise ValueError(f"farm index must be between 0 and 52, got {x}")

    # Gelderland (0–11)
    if x < 12:
        return "gelderland", x

    # Groningen (12–24)
    if x < 12 + 13:  # up to index 24
        return "groningen", x - 12

    # Zeeland (25–52)
    return "zeeland", x - (12 + 13)


def train_allocator(args):
    log_folder_name = f"Bandit_{args.method}_{datetime.datetime.now():%m%d-%H%M}"
    # initialize comet if using
    comet_experiment = None
    if args.use_comet:
        comet_experiment = _setup_bandit_comet(args)

    years = list(range(2020 - args.years, 2020))

    # initialize env
    env = AllocationBandit(
        warm_up_eps=2,
        args=args,
        years=years,
        seed=args.seed,
        flat_context=True,
    )

    # context and action dims
    d_theta, d_x = env.observation_space.shape[0], env.action_space.shape[0]

    # the paper suggested this
    m = d_theta//10 + d_x//3 + 3
    print(f"d_theta: {d_theta}, d_x: {d_x}. So, m: {m}")

    # put the bandit algorithm here
    bandit = NNAGPBandit(
        d_theta=d_theta,
        d_x=d_x,
        m=m,  # vector len of multi output GP
        Q=args.q,  # number of shared GP outputs
        lr=args.bandit_lr,
        device=torch.device("cpu")
    )

    training_loop(env, bandit, args, comet_experiment, log_folder_name)

    print("Training Complete!")


def train_allocator_for_farm(args):
    farm_int = args.farm
    render = args.render

    region, farm_id = farm_int_mapper(farm_int)

    print(f"Training Farm {region}_{farm_id}!")
    print(f"Using {args.model_dir} policy!")

    # training_years = list(range(2000, 2020))

    years = list(range(2020 - args.years, 2020))

    log_folder_name = f"Bandit_{args.model_dir}_{region}_{farm_id}_{datetime.datetime.now():%m%d}"
    # initialize comet if using
    comet_experiment = None
    if args.use_comet:
        comet_experiment = _setup_bandit_comet(args, region=region, farm_id=farm_id)
        comet_experiment.add_tags([f"{region}_{farm_id}"])
        comet_experiment.add_tag(f"{args.method}")
        comet_experiment.add_tag(f"{args.model_dir}")

    env = AllocationBandit(
        warm_up_eps=len(years),
        args=args,
        seed=args.seed,
        flat_context=True,
        field_reward='NSU',
        years=years,
        region=region,
        farm_id=farm_id,
        render=render,
    )

    # context and action dims
    d_theta, d_x = env.observation_space.shape[0], env.action_space.shape[0]

    # the paper suggested this
    m = d_theta // 10 + d_x // 3 + 3
    print(f"d_theta: {d_theta}, d_x: {d_x}. So, m: {m}")

    # put the bandit algorithm here
    bandit = NNAGPBandit(
        d_theta=d_theta,
        d_x=d_x,
        m=m,  # vector len of multi output GP
        Q=args.q,  # number of shared GP outputs
        lr=args.bandit_lr,
        device=torch.device("cpu"),
        posterior_type=args.bandit_posterior,
        coreset_size=args.coreset_size,
        coreset_mode=args.coreset_mode,
    )

    training_loop(env, bandit, args, comet_experiment, log_folder_name, region=region, farm_id=farm_id)

    print(f"Training for Farm {region}_{farm_id} Complete!")



def training_loop(env: AllocationBandit, bandit: NNAGPBandit, args, comet_experiment = None,
                log_folder_name: str = None, region: str | None = None, farm_id: int | None = None):

    # misc
    rng = np.random.RandomState(args.seed)
    torch.set_default_dtype(torch.float32)

    # action candidates for each round
    num_candidates = args.action_candidate_length

    # initialize running mean
    rms = RunningMeanStd()

    test_step = 0

    method = args.method

    # ---------------- Elite candidate memory ----------------
    # Keep the best action seen so far (by *raw* env reward) and its neighbors,
    # and always inject them into later candidate sets.
    elite_enabled = getattr(args, "elite_enabled", False)
    elite_neighbors = int(getattr(args, "elite_neighbors", 64))
    elite_keep_max = int(getattr(args, "elite_keep_max", 512))
    elite_top_k = int(getattr(args, "elite_top_k", 10))

    # For each scenario, keep up to top-K elite centers.
    # Each elite stores: reward, key, and its local candidate cloud (center + neighbors).
    elite = {
        "full": {"items": [], "cands": None},
        "reduced": {"items": [], "cands": None},
    }

    def _unique_rows_torch(X: torch.Tensor) -> torch.Tensor:
        # torch.unique(dim=0) exists in modern torch; keep it simple
        if X is None or X.numel() == 0:
            return X
        return torch.unique(X, dim=0)

    def _center_key(x: torch.Tensor) -> tuple:
        # Stable key for comparing centers across steps
        # (round to avoid tiny float differences)
        arr = x.detach().cpu().numpy().astype(np.float64)
        return tuple(np.round(arr, 6).tolist())

    def _rebuild_elite_cands(scenario: str):
        """Rebuild the aggregated elite candidate tensor for fast injection."""
        if not elite_enabled:
            elite[scenario]["cands"] = None
            return
        items = elite[scenario]["items"]
        if not items:
            elite[scenario]["cands"] = None
            return
        all_c = torch.vstack([it["cands"] for it in items])
        all_c = _unique_rows_torch(all_c)
        if all_c.shape[0] > elite_keep_max:
            all_c = all_c[:elite_keep_max]
        elite[scenario]["cands"] = all_c

    def _update_elite(env, scenario: str, x_t: torch.Tensor, reward_raw: float):
        """Maintain a top-K set of elite centers for the scenario."""
        if not elite_enabled:
            return

        key = _center_key(x_t)
        items = elite[scenario]["items"]

        # If this exact center already exists, only update if reward improved
        for it in items:
            if it["key"] == key:
                if reward_raw > it["reward"]:
                    it["reward"] = float(reward_raw)

                    # refresh its neighbor cloud
                    center = x_t.detach().cpu().numpy().reshape(1, -1)
                    neigh_np = env.sample_neighbors(
                        center=center.squeeze(0),
                        n_neighbors=elite_neighbors,
                        reduced=(scenario == "reduced"),
                    )
                    cand_np = np.vstack([center, neigh_np]).astype(np.float32)
                    it["cands"] = _unique_rows_torch(torch.from_numpy(cand_np))

                    _rebuild_elite_cands(scenario)
                return

        # Decide whether to insert new elite center
        if len(items) < elite_top_k:
            should_insert = True
        else:
            worst = min(items, key=lambda z: z["reward"])
            should_insert = reward_raw > worst["reward"]

        if not should_insert:
            return

        # Build candidate cloud for this new elite center
        center = x_t.detach().cpu().numpy().reshape(1, -1)
        neigh_np = env.sample_neighbors(
            center=center.squeeze(0),
            n_neighbors=elite_neighbors,
            reduced=(scenario == "reduced"),
        )
        cand_np = np.vstack([center, neigh_np]).astype(np.float32)
        cand = _unique_rows_torch(torch.from_numpy(cand_np))

        items.append({
            "reward": float(reward_raw),
            "key": key,
            "cands": cand,
        })

        # Keep only top-K by reward
        items.sort(key=lambda z: z["reward"], reverse=True)
        del items[elite_top_k:]

        # Rebuild aggregated candidates used for injection
        _rebuild_elite_cands(scenario)

    def _inject_elite(scenario: str, Xc: torch.Tensor) -> torch.Tensor:
        if not elite_enabled:
            return Xc
        elite_c = elite[scenario]["cands"]
        if elite_c is None:
            return Xc
        elite_c = elite_c.to(dtype=Xc.dtype)
        X = torch.vstack([elite_c, Xc])
        return _unique_rows_torch(X)

    # --------------------------------------------------------

    # put the training loop here
    for t in range(1, args.rounds + 1):
        if region is not None or farm_id is not None:
            env.get_rotation_year(rng.choice([2020, 2021, 2022, 2023, 2024]))
        theta_t, env_info = env.reset(
            options={
                'year': rng.choice(env.years),
            },
            seed=args.seed
        )
        if comet_experiment:
            comet_experiment.log_metric("episode/year", int(env_info['year']))

        # normalize
        # rms.update(theta_t)
        # theta_t = rms.norm(theta_t)

        # convert to numpy
        theta_t = torch.from_numpy(theta_t)

        x_cand = env.sample_super_arms(
            n_candidates=num_candidates,
            reduced=False,
            rng=rng,
        )
        x_cand = torch.from_numpy(x_cand.astype(np.float32))

        # Always inject persistent elite candidates for training scenario="full"
        x_cand = _inject_elite("full", x_cand)

        # train the surrogate a bit on accumulated data
        loss_val = bandit.train_step(steps=args.bandit_epochs, lr=args.bandit_lr)
        n = len(bandit.y_hist)
        loss_per_sample = loss_val / max(n, 1)
        print(f"round {t}, loss: {loss_per_sample}")
        if comet_experiment:
            comet_experiment.log_metric("loss", loss_per_sample, step=t)

            log_model_histograms(
                comet_experiment,
                bandit.model,
                step=t,
                prefix="nn-agp/",
                log_grads=True,
            )

        if not args.streaming:
            # pick by UCB (or switch to bandit.select_ts(...))
            if method == "ucb":
                x_t, selection_info = bandit.select_ucb(theta_t, x_cand, delta=0.1)
            else:
                x_t, selection_info = bandit.select_ts(theta_t, x_cand)
            if isinstance(x_t, np.ndarray):
                x_t = torch.from_numpy(x_t)
            if comet_experiment:
                log_selection_info(comet_experiment, selection_info, t)
        else:
            x_t, best = bandit.select_ucb_streaming(theta_t, torch.from_numpy(x_cand), delta=0.1)
            if isinstance(x_t, np.ndarray):
                x_t = torch.from_numpy(x_t)
            if comet_experiment:
                comet_experiment.log_metrics(best, step=t) if best is not None else None

        # run env and normalize reward
        _, reward_env, _, _, step_info = env.step(x_t)
        normalized_reward = min_max_normalize(float(reward_env))
        print(f"reward: {reward_env}")

        # Update elite memory using *raw* reward signal
        _update_elite(env, "full", x_t, float(reward_env))

        if comet_experiment:
            comet_experiment.log_metrics(
                {
                    "reward/train/normalized": float(normalized_reward),
                    "reward/train/reward": float(reward_env),
                },
                step=t
            )

        # update rolling historical average
        # env.add_stats_to_context(step_info['AgentInfos'])

        # observe noisy reward
        y_t = float(normalized_reward + 0.05 * torch.randn(()))
        if comet_experiment:
            comet_experiment.log_metric(
                "reward/noisy",
                float(y_t),
                step=t
            )
        bandit.update(theta_t, x_t, y_t)

        test_per_round = args.eval_steps

        # eval the allocator after rounds
        if t % test_per_round == 0:
            # test bandit
            bandit.model.eval()
            # edit?
            years: list = [2020, 2021, 2022, 2023, 2024]

            rewards = []

            scenarios = ['full', 'reduced']
            for scenario in scenarios:
                info_dict = {}
                for year in years:
                    raw_reward, normalized_reward, infos, x_eval = run_eval_allocator(
                        env=env,
                        bandit=bandit,
                        year=year,
                        rms=rms,
                        experiment=comet_experiment,
                        step=t,
                        method=method,
                        candidate_size=100_000,
                        scenario=scenario,
                        elite_candidates_np=(
                            None if elite[scenario]["cands"] is None
                            else elite[scenario]["cands"].detach().cpu().numpy()
                        )
                    )
                    # Update elite from eval too (per scenario)
                    _update_elite(env, scenario, x_eval, float(raw_reward))
                    info_dict[year] = infos
                    print(f"test year: {year}, reward: {raw_reward}")
                    rewards.append(raw_reward)
                    if comet_experiment:
                        comet_experiment.log_metrics(
                            {
                                f"reward/scenario_{scenario}/test_year:{year}/raw": float(raw_reward),
                                f"reward/scenario_{scenario}/test_year:{year}/normalized": float(normalized_reward),
                            },
                            step=test_step,
                        )
                        fig = plot_results(
                                step_info['AgentInfos'],
                                variable_list=['DVS', 'Profit', 'Reward', 'Action', 'Yield', 'BudgetLeft'],
                                show=False,
                            )
                        comet_experiment.log_figure(
                            figure_name=f"image/{scenario}/plot_year:{year}",
                            figure=fig,
                            step=test_step,
                        )
                        plt.close(fig)
                pickle_path = os.path.join(_DEFAULT_LOGDIR, f"Bandit_{args.model_dir}_{scenario}",
                                           f"bandit_{region}_{farm_id}_info_{test_step}.pkl")
                os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
                with open(pickle_path, "wb") as f:
                    pickle.dump(info_dict, f)
                if comet_experiment:
                    comet_experiment.log_metrics(
                        {
                            f"reward/mean/raw": float(np.sum(rewards)),
                        },
                        step=test_step,
                    )
                    comet_experiment.log_asset(
                        file_data=pickle_path,
                        file_name=f"bandit_{region}_{farm_id}_{scenario}_info.pkl",
                        step=test_step,
                    )

            test_step += test_per_round

            bandit.model.train()

        farm_name = f"{region}_{farm_id}" if region is not None else None

        # save the model iteratively
        if t % test_per_round == 0:
            file_dir = bandit.save(
                seed=args.seed,
                t=t,
                name=log_folder_name,
                args=args,
                rms=rms,
                farm_id=farm_name,
            )

            if comet_experiment:
                comet_experiment.log_asset(
                    file_dir,
                    file_name=f"s{args.seed}_{args.method}_bandit_{t}",
                    step=t,
                )
