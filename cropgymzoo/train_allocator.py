import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import yaml
import shutil
from copy import deepcopy

import numpy as np
import os
import matplotlib.pyplot as plt

import datetime
import pickle

from collections import deque

from cropgymzoo.eval_allocator import run_eval_allocator
from cropgymzoo.agents.nn_agp import NNAGPBandit
from cropgymzoo.utils.agent_helpers import last_before_nan
from cropgymzoo.utils.callbacks import _setup_bandit_comet, log_selection_info, log_model_histograms, fig_to_chw_uint8
from cropgymzoo.envs.allocation_env import AllocationBandit
from cropgymzoo import _DEFAULT_LOGDIR, _DEFAULT_RESULTSDIR, _SCENARIO_PATH
from cropgymzoo.utils.plotters import plot_results, plot_results_daisy_chained


class BanditNormalizer:
    def __init__(
        self,
        env,
        rng,
        seed: int | None = None,
        clip: float = 999.0,
        n_calib: int = 20,
        cache_dir: str = "theta_norm_cache",
        cache_tag: str | None = None,
        use_cache: bool = True,
    ):
        self.theta_clip = float(clip)

        years_len = int(len(env.years))
        d_theta = int(env.observation_space.shape[0])

        # cache file name
        tag = "" if cache_tag is None else f"_{cache_tag}"
        os.makedirs(os.path.join(_DEFAULT_RESULTSDIR, cache_dir), exist_ok=True)
        cache_path = os.path.join(
            _DEFAULT_RESULTSDIR,
            cache_dir,
            f"theta_norm_n_years_{years_len}{tag}.pkl"
        )

        # -------- Try loading cache --------
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    blob = pickle.load(f)

                ok = True
                ok &= int(blob.get("years_len", -1)) == years_len
                ok &= ("theta_mean" in blob) and ("theta_std" in blob)

                if ok:
                    self.theta_mean = np.asarray(blob["theta_mean"], dtype=np.float32)
                    self.theta_std = np.asarray(blob["theta_std"], dtype=np.float32)
                    if "theta_clip" in blob:
                        self.theta_clip = float(blob["theta_clip"])
                    assert len(self.theta_mean) == d_theta, "Theta is not the same length as saved run; rerunning the Normalizer!"
                    print(f"[BanditNormalizer] Loaded cache: {cache_path}")
                    return
                else:
                    print(f"[BanditNormalizer] Cache incompatible -> recomputing: {cache_path}")

            except Exception as e:
                print(f"[BanditNormalizer] Failed to load cache -> recomputing. Reason: {e}")

        # -------- Compute fresh stats --------
        print("[BanditNormalizer] Computing fixed theta normalization stats...")

        theta_buf = []
        n_calib = int(n_calib)

        for i in range(n_calib):
            # keep this cheap + representative
            print(f"reset number {i+1}")
            years = [2020, 2021, 2022, 2023, 2024]
            # env.get_rotation_year(rng.choice(years))
            farm_dict_by_year = {}
            for y in years:
                with open(
                        os.path.join(_SCENARIO_PATH, f"{env.region}", f"{y}", f"farmer_{env.farm_id}.yaml"),
                        'r') as f:
                    farm_dict_by_year[int(y)] = yaml.safe_load(f)
            theta_np, _ = env.reset(
                options={
                    "year": int(years[0]),
                    "eval_horizon_years": [int(y) for y in years],
                    "farm_dict_by_year": farm_dict_by_year,
                    "preseason_allocation": True,
                    # simpler decision point: 7 days before sowing
                    "days_before_sowing": 7,
                },
                seed=seed or 0,
            )
            # theta_buf.append(theta_np.astype(np.float32))

            done = False
            while not done:
                theta_buf.append(theta_np.astype(np.float32))
                theta_np, raw_env, done, _, _ = env.step(torch.randint(low=0, high=10, size=(len(env.agents_order),)))


        theta_stack = np.stack(theta_buf, axis=0)
        self.theta_mean = theta_stack.mean(axis=0).astype(np.float32)
        theta_std = theta_stack.std(axis=0).astype(np.float32)
        self.theta_std = np.maximum(theta_std, 1e-6).astype(np.float32)

        print("[BanditNormalizer] Done computing stats.")

        # -------- Save cache --------
        if use_cache:
            blob = {
                "years_len": years_len,
                "theta_mean": self.theta_mean,
                "theta_std": self.theta_std,
                "theta_clip": self.theta_clip,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(blob, f)
            print(f"[BanditNormalizer] Saved cache: {cache_path}")

    def norm_theta(self, theta_np: np.ndarray) -> np.ndarray:
        z = (theta_np.astype(np.float32) - self.theta_mean) / self.theta_std
        return np.clip(z, -self.theta_clip, self.theta_clip)


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

    # training_loop(env, bandit, args, comet_experiment, log_folder_name)
    #
    # print("Training Complete!")


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

    bandit_action_mode = getattr(args, "bandit_action_mode", "factored")  # 'joint' or 'factored'

    d_theta_per_field = len(env._get_context_keys())
    m_field = d_theta_per_field // 10 + 1 // 3 + 3

    bandit = NNAGPBandit(
        d_theta=d_theta,
        d_x=d_x,
        m=m,  # kept for compatibility; sub-bandits use m_field
        Q=args.q,
        lr=args.bandit_lr,
        device=torch.device("cpu"),
        posterior_type=args.bandit_posterior,
        coreset_size=args.coreset_size,
        coreset_mode=args.coreset_mode,
        action_mode="factored",
        n_fields=int(env.n_fields),
        d_theta_per_field=int(d_theta_per_field),
        m_sub=int(m_field),
        use_farm_budget=True,
        use_lstm=getattr(args, "lstm", False),
    )

    training_loop_factored(
        env, bandit, args, comet_experiment, log_folder_name, region=region, farm_id=farm_id
    )

    print(f"Training for Farm {region}_{farm_id} Complete!")



def training_loop_factored(
    env,
    bandit,
    args,
    comet_experiment=None,
    log_folder_name: str | None = None,
    region: str | None = None,
    farm_id: int | None = None,
):
    """
    Clean factored loop:
      - Each field i has a small 1D candidate set (e.g., 0..27 step 0.5 => 55 candidates).
      - We score ALL candidates per field each round (no candidate sampling tricks).
    """
    if getattr(bandit, "action_mode", "joint") != "factored":
        raise ValueError("training_loop_factored_clean requires bandit.action_mode == 'factored'")

    rng = np.random.RandomState(args.seed)
    torch.set_default_dtype(torch.float32)

    rms = BanditNormalizer(env, rng, cache_tag=f"{str(region)}_{str(farm_id)}")
    method = args.method
    test_step = 0

    test_years = [2020, 2021, 2022, 2023, 2024]

    # Persistent best-eval trackers
    # best_eval_sum = {seed: {"full": float("-inf"), "reduced": float("-inf")} for seed in
    #                  range(args.seed, args.seed + seed_range)}
    # best_eval_step = {seed: {"full": None, "reduced": None} for seed in range(args.seed, args.seed + seed_range)}
    # best_eval_pickle = {seed: {"full": None, "reduced": None} for seed in range(args.seed, args.seed + seed_range)}

    best_eval_sum = {"full": float("-inf"), "reduced": float("-inf")}
    best_eval_step = {"full": None, "reduced": None}
    best_eval_pickle = {"full": None, "reduced": None}

    concurrent_rolling = bool(getattr(args, "concurrent_rolling", False))
    if concurrent_rolling:
        bandits = {
            k: deepcopy(bandit) for k in test_years
        }

        for k, b in bandits.items():
            length_stride = len(list(range(2015, k)))
            b.seq_len = length_stride
            b.seq_stride = length_stride
            b._theta_seq = deque(maxlen=length_stride)

    online_update_mode = bool(getattr(args, "online_update_mode", False))


    def set_subbandits_train_mode(train: bool):
        # Your factored NNAGPBandit stores sub-bandits in bandit.sub_bandits
        for sb in getattr(bandit, "sub_bandits", []):
            sb.model.train(train)

    def hist_size_total() -> int:
        return int(sum(len(sb.y_hist) for sb in getattr(bandit, "sub_bandits", [])))

    for t in range(1, args.rounds + 1):

        if not getattr(args, "train_multi_campaign", True):
            singe_year_train(bandit, env, t, args, rms, rng, method, test_years, hist_size_total, farm_id, region,
                             comet_experiment)


        else:  # use multi campaign

            if concurrent_rolling:

                bandit, test_step, train_raw_reward = multi_year_train_concurrent(t, args, bandit, bandits,
                                                                                  best_eval_pickle, best_eval_step,
                                                                                  best_eval_sum, env,
                                                                                  set_subbandits_train_mode, rms, rng,
                                                                                  hist_size_total, method,
                                                                                  log_folder_name, farm_id, region,
                                                                                  test_step, test_years,
                                                                                  comet_experiment)

            # Train with one bandit
            else:

                train_raw_reward = []
                kickback = 5

                train_years = [y - kickback for y in test_years]

                train_farm_dict_by_year = {}
                for y in train_years:
                    with open(os.path.join(_SCENARIO_PATH, f"{env.region}", f"{y+kickback}", f"farmer_{env.farm_id}.yaml"),
                              'r') as f:
                        train_farm_dict_by_year[int(y)] = yaml.safe_load(f)

                # Ensure env has the right agent set (field ids) before reset
                env.get_rotation_year(test_years[0])

                th_np, info = env.reset(
                    options={
                        "year": int(train_years[0]),
                        "random_initial_conditions": True,
                        "eval_horizon_years": [int(y) for y in train_years],
                        "farm_dict_by_year": train_farm_dict_by_year,
                        "days_before_sowing": 7,
                    },
                    seed=args.seed,
                )

                done = False
                while not done:
                    # Normalize (flat) context
                    th_np = rms.norm_theta(th_np)
                    th_fields_np = env.unflatten_context_per_field(th_np)
                    th_fields = torch.from_numpy(th_fields_np.astype(np.float32))

                    X_list = env.build_full_candidates_per_field()

                    # Train occasionally
                    train_every = int(getattr(args, "train_every", 1))
                    if t % train_every == 0:
                        steps = int(getattr(args, "bandit_epochs", 100))
                        loss_val = float(bandit.train_step(steps=steps, lr=args.bandit_lr))
                        n_hist = hist_size_total()
                        loss_per_sample = loss_val / max(n_hist, 1)
                        print(f"round {t}, loss: {loss_per_sample} (train_steps={steps})")
                        if comet_experiment:
                            comet_experiment.log_metric("loss", loss_per_sample, step=t)

                    if method == "ucb":
                        x_t, sel_info = bandit.select_ucb_factored(
                            th_fields,
                            X_list,
                            delta=0.1,
                            global_budget=float(env.global_budget),
                            max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                            seed=args.seed,
                        )
                    else:
                        x_t, sel_info = bandit.select_ts_factored(
                            th_fields,
                            X_list,
                            global_budget=float(env.global_budget),
                            max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                            seed=args.seed,
                        )

                    th_np, raw_env, done, _, step_infos = env.step(x_t)

                    print(f"reward: {raw_env}")

                    train_raw_reward.append(raw_env)

                    # Update rolling historical context
                    # env.add_stats_to_context(env.filter_historical_info(step_info["AgentInfos"]))

                    n_surp_infos = [last_before_nan(step_infos['AgentInfos'][a]["Nsurp"]) for a in env.unwrapped.agents_order]
                    nue_infos = [last_before_nan(step_infos['AgentInfos'][a]["Nue"]) for a in env.unwrapped.agents_order]

                    # Per-field reward components (consistent with env._get_reward())
                    y_fields = env.compute_per_field_rewards(n_surp_infos, nue_infos)

                    # Optional observation noise (match your previous habit if desired)
                    noise = float(getattr(args, "reward_noise", 0.005))
                    if noise > 0:
                        y_fields = y_fields + noise * rng.randn(len(y_fields)).astype(np.float32)

                    # Update factored bandit
                    bandit.update_factored(th_fields, x_t, y_fields)

                train_raw_reward = np.mean(train_raw_reward)

                if comet_experiment:
                    comet_experiment.log_metrics({"reward/train/reward": float(train_raw_reward)}, step=t)

                # Periodic eval (daisy-chained multi-year evaluation)
                test_per_round = int(args.eval_steps)

                if t % test_per_round == 0:

                    # online update path
                    if online_update_mode:
                        bandit_eval = deepcopy(bandit)

                        for b in bandit_eval.sub_bandits:
                            b.model.train(False)
                    else:

                        set_subbandits_train_mode(False)

                    for scenario in ["full", "reduced"]:
                        print(f"\n\nEval scenario: {scenario}\n")
                        years = test_years
                        rewards = []
                        info_dict = {}

                        # Choose eval budget
                        if scenario == "full":
                            eval_budget = float(env.global_budget)
                        else:
                            eval_budget = float(0.7) * float(env.global_budget)

                        # Build farm_dict_by_year once (needed for chained campaigns)
                        farm_dict_by_year = {}
                        for y in years:
                            with open(os.path.join(_SCENARIO_PATH, f"{env.region}", f"{y}", f"farmer_{env.farm_id}.yaml"), 'r') as f:
                                farm_dict_by_year[int(y)] = yaml.safe_load(f)

                        # Ensure env has the right agent set (field ids) before reset
                        env.get_rotation_year(years[0])

                        # Reset ONCE with horizon options
                        th_np, info = env.reset(
                            options={
                                "year": int(years[0]),
                                "eval_horizon_years": [int(y) for y in years],
                                "farm_dict_by_year": farm_dict_by_year,
                                "preseason_allocation": True,
                                # simpler decision point: 7 days before sowing
                                "days_before_sowing": 7,
                            },
                            seed=args.seed,
                        )

                        done = False
                        while not done:
                            # Normalize (flat) context
                            th_np = rms.norm_theta(th_np)
                            th_fields_np = env.unflatten_context_per_field(th_np)
                            th_fields = torch.from_numpy(th_fields_np.astype(np.float32))

                            X_list = env.build_full_candidates_per_field()

                            if not online_update_mode:
                                if method == "ucb":
                                    x_eval, sel_info = bandit.select_ucb_factored(
                                        th_fields,
                                        X_list,
                                        delta=0.1,
                                        global_budget=eval_budget,
                                        max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                                        deterministic=True,
                                    )
                                else:
                                    x_eval, sel_info = bandit.select_ts_factored(
                                        th_fields,
                                        X_list,
                                        global_budget=eval_budget,
                                        max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                                        # seed=args.seed,
                                        # deterministic=True,
                                        # temperature=0.5,
                                    )
                            else:
                                if method == "ucb":
                                    x_eval, sel_info = bandit_eval.select_ucb_factored(
                                        th_fields,
                                        X_list,
                                        delta=0.1,
                                        global_budget=eval_budget,
                                        max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                                        deterministic=True,
                                    )
                                else:
                                    x_eval, sel_info = bandit_eval.select_ts_factored(
                                        th_fields,
                                        X_list,
                                        global_budget=eval_budget,
                                        max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                                        # seed=args.seed,
                                        deterministic=True,
                                        # temperature=0.5,
                                    )

                            # Budget violation check (only meaningful in reduced)
                            if scenario == "reduced":
                                applied = float((torch.as_tensor(env.max_budgets, dtype=torch.float32) - (x_eval.detach().cpu() * 10)).sum().item())
                                if applied > float(eval_budget) + 1e-3:
                                    print(f"\n\n[WARN] budget violated: applied={applied:.3f} > B={float(eval_budget):.3f}\n")

                            th_np, raw_reward, done, _, infos = env.step(x_eval)

                            season_year = int(infos["AgentInfos"][env.agents_order[0]]["SeasonYear"][-1])

                            if season_year is not None:
                                info_dict[int(season_year)] = infos["AgentInfos"]

                            rewards.append(float(raw_reward))
                            print(f"test season_year: {season_year}, reward: {raw_reward}")

                            if online_update_mode:

                                n_surp_infos = [last_before_nan(infos['AgentInfos'][a]["Nsurp"]) for a in
                                                env.unwrapped.agents_order]
                                nue_infos = [last_before_nan(infos['AgentInfos'][a]["Nue"]) for a in
                                             env.unwrapped.agents_order]

                                # Per-field reward components (consistent with env._get_reward())
                                y_fields = env.compute_per_field_rewards(n_surp_infos, nue_infos)

                                noise = float(getattr(args, "reward_noise", 0.005))
                                if noise > 0:
                                    y_fields = y_fields + noise * rng.randn(len(y_fields)).astype(np.float32)

                                # Update factored bandit
                                bandit_eval.update_factored(th_fields, x_eval, y_fields)

                                # _ = float(bandit_eval.train_step(steps=steps, lr=args.bandit_lr))

                            if comet_experiment:
                                comet_experiment.log_metrics(
                                    {f"reward/{scenario}/test_year:{season_year}/raw": float(raw_reward)},
                                    step=test_step,
                                )
                                fig = plot_results(
                                    infos["AgentInfos"],
                                    variable_list=["DVS", "Profit", "Action", "Yield", "BudgetLeft"],
                                    show=False,
                                )
                                comet_experiment.log_figure(
                                    figure_name=f"image/{scenario}/plot_year:{season_year}",
                                    figure=fig,
                                    step=test_step,
                                )
                                plt.close(fig)
                        else:
                            if comet_experiment:
                                fig = plot_results_daisy_chained(
                                    info_dict,  # dict[season_year] -> AgentInfos
                                    variable_list=["DVS", "Profit", "Action", "Yield", "BudgetLeft", "NAVAIL"],
                                    show=False,
                                )
                                comet_experiment.log_figure(
                                    figure_name=f"image/{scenario}/plot_eval",
                                    figure=fig,
                                    step=test_step,
                                )
                                plt.close(fig)

                        # Return this
                        env.unwrapped.farm.set_print_season_year(None)
                        # Aggregate scenario score
                        sum_reward = float(np.sum(rewards))
                        if comet_experiment:
                            comet_experiment.log_metric(f"reward/{scenario}/sum", sum_reward, step=test_step)

                            # Always save the latest eval info_dict for this scenario
                            latest_pickle_path = os.path.join(
                                _DEFAULT_LOGDIR,
                                f"Bandit_{args.model_dir}_{scenario}",
                                f"bandit_{region}_{farm_id}_info_{test_step}.pkl",
                            )
                            os.makedirs(os.path.dirname(latest_pickle_path), exist_ok=True)
                            with open(latest_pickle_path, "wb") as f:
                                pickle.dump(info_dict, f)

                            if comet_experiment:
                                comet_experiment.log_asset(
                                    file_data=latest_pickle_path,
                                    file_name=f"bandit_{region}_{farm_id}_{scenario}_info.pkl",
                                    step=test_step,
                                )

                            # Track + save best eval per scenario per seed
                            if sum_reward > best_eval_sum[scenario]:
                                best_eval_sum[scenario] = sum_reward
                                best_eval_step[scenario] = int(test_step)

                                os.makedirs(
                                    os.path.join(
                                        _DEFAULT_LOGDIR,
                                        f"Bandit_{args.model_dir}_{scenario}_seed{args.seed}",
                                    ), exist_ok=True
                                )

                                best_pickle_path = os.path.join(
                                    _DEFAULT_LOGDIR,
                                    f"Bandit_{args.model_dir}_{scenario}_seed{args.seed}",
                                    f"bandit_{region}_{farm_id}_BEST.pkl",
                                )
                                os.makedirs(os.path.dirname(best_pickle_path), exist_ok=True)
                                with open(best_pickle_path, "wb") as f:
                                    pickle.dump(info_dict, f)

                                best_eval_pickle[scenario] = best_pickle_path

                                print(f"[BEST] New best {scenario}: sum_reward={sum_reward:.6f} at eval_step={test_step}")
                                if comet_experiment:
                                    comet_experiment.log_metrics(
                                        {
                                            f"reward/{scenario}/best_sum": float(sum_reward),
                                        },
                                        step=test_step,
                                    )
                                    comet_experiment.log_asset(
                                        file_data=best_pickle_path,
                                        file_name=f"bandit_{region}_{farm_id}_{scenario}_BEST_info.pkl",
                                        step=test_step,
                                    )

                    test_step += test_per_round

                    if not online_update_mode:
                        set_subbandits_train_mode(True)
                    else:
                        del bandit_eval

                    # Save model
                    farm_name = f"{region}_{farm_id}" if region is not None else None
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




def multi_year_train_concurrent(t: int, args, bandit, bandits,
                                best_eval_pickle: dict[int, dict[str, None]] | dict[str, None],
                                best_eval_step: dict[int, dict[str, None]] | dict[str, None],
                                best_eval_sum: dict[int, dict[str, float]] | dict[str, float], env,
                                set_subbandits_train_mode, rms: BanditNormalizer,
                                rng, hist_size_total, method,
                                log_folder_name: str | None, farm_id: int | None, region: str | None,
                                test_step: int, test_years: list[int], comet_experiment):
    # loop bandits
    for model_eval_year, bandit in bandits.items():

        train_raw_reward = []

        train_years = list(range(2015, model_eval_year))  # not including model_eval_year

        train_farm_dict_by_year = {}
        for y in train_years:
            file_y = 2020 + (y % 5)
            with open(os.path.join(_SCENARIO_PATH, f"{env.region}", f"{file_y}", f"farmer_{env.farm_id}.yaml"),
                      'r') as f:
                train_farm_dict_by_year[int(y)] = yaml.safe_load(f)

        # Ensure env has the right agent set (field ids) before reset
        env.get_rotation_year(test_years[0])

        th_np, info = env.reset(
            options={
                "year": int(train_years[0]),
                "random_initial_conditions": True,
                "eval_horizon_years": [int(y) for y in train_years],
                "farm_dict_by_year": train_farm_dict_by_year,
                "days_before_sowing": 7,
            },
            seed=args.seed,
        )

        done = False
        while not done:
            # Normalize (flat) context
            th_np = rms.norm_theta(th_np)
            th_fields_np = env.unflatten_context_per_field(th_np)
            th_fields = torch.from_numpy(th_fields_np.astype(np.float32))

            X_list = env.build_full_candidates_per_field()

            # Train occasionally
            train_every = int(getattr(args, "train_every", 1))
            if t % train_every == 0:
                steps = int(getattr(args, "bandit_epochs", 100))
                loss_val = float(bandit.train_step(steps=steps, lr=args.bandit_lr))
                n_hist = hist_size_total()
                loss_per_sample = loss_val / max(n_hist, 1)
                print(f"Bandit {model_eval_year}, round {t}, loss: {loss_per_sample} (train_steps={steps})")
                if comet_experiment:
                    comet_experiment.log_metric(f"loss/bandit_{model_eval_year}", loss_per_sample, step=t)

            if method == "ucb":
                x_t, sel_info = bandit.select_ucb_factored(
                    th_fields,
                    X_list,
                    delta=0.1,
                    global_budget=float(env.global_budget),
                    max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                )
            else:
                x_t, sel_info = bandit.select_ts_factored(
                    th_fields,
                    X_list,
                    global_budget=float(env.global_budget),
                    max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                )

            th_np, raw_env, done, _, step_infos = env.step(x_t)

            n_surp_infos = [last_before_nan(step_infos['AgentInfos'][a]["Nsurp"]) for a in
                            env.unwrapped.agents_order]
            nue_infos = [last_before_nan(step_infos['AgentInfos'][a]["Nue"]) for a in
                         env.unwrapped.agents_order]
            print(f"bandit_{model_eval_year} reward: {raw_env}")

            train_raw_reward.append(raw_env)

            # Update rolling historical context
            # env.add_stats_to_context(env.filter_historical_info(step_info["AgentInfos"]))

            # Per-field reward components (consistent with env._get_reward())
            y_fields = env.compute_per_field_rewards(n_surp_infos, nue_infos)

            # Optional observation noise (match your previous habit if desired)
            noise = float(getattr(args, "reward_noise", 0.005))
            if noise > 0:
                y_fields = y_fields + noise * rng.randn(len(y_fields)).astype(np.float32)

            # Update factored bandit
            bandit.update_factored(th_fields, x_t, y_fields)

        mean_train_reward = np.mean(train_raw_reward)

        if comet_experiment:
            comet_experiment.log_metrics({f"train/reward/bandit_{model_eval_year}": float(mean_train_reward)}, step=t)

    # Periodic eval (daisy-chained multi-year evaluation)
    test_per_round = 2
    if t % test_per_round == 0:
        for bandit in bandits.values():
            set_subbandits_train_mode(False)

        for scenario in ["full", "reduced"]:
            print(f"\n\nEval scenario: {scenario}\n")
            # seed = args.seed + n
            years = test_years
            rewards = []
            info_dict = {}

            # Choose eval budget
            if scenario == "full":
                eval_budget = float(env.global_budget)
            else:
                eval_budget = float(0.7) * float(env.global_budget)

            # Build farm_dict_by_year once (needed for chained campaigns)
            farm_dict_by_year = {}
            for y in years:
                with open(os.path.join(_SCENARIO_PATH, f"{env.region}", f"{y}",
                                       f"farmer_{env.farm_id}.yaml"), 'r') as f:
                    farm_dict_by_year[int(y)] = yaml.safe_load(f)

            # Ensure env has the right agent set (field ids) before reset
            env.get_rotation_year(years[0])

            year_idx = 0

            # Reset ONCE with horizon options
            th_np, info = env.reset(
                options={
                    "year": int(years[0]),
                    "eval_horizon_years": [int(y) for y in years],
                    "farm_dict_by_year": farm_dict_by_year,
                    "preseason_allocation": True,
                    # simpler decision point: 7 days before sowing
                    "days_before_sowing": 7,
                },
                seed=args.seed,
            )

            use_lstm = bool(getattr(args, "lstm", False))
            seq_len = int(getattr(bandit, "seq_len", 5))
            theta_seq_eval = deque(maxlen=seq_len) if use_lstm else None

            done = False
            while not done:
                # Normalize (flat) context
                th_np = rms.norm_theta(th_np)
                th_fields_np = env.unflatten_context_per_field(th_np)
                th_fields = torch.from_numpy(th_fields_np.astype(np.float32))

                X_list = env.build_full_candidates_per_field()

                print(f"Choosing actions for year {years[year_idx]}!")

                if method == "ucb":
                    x_eval, sel_info = bandits[years[year_idx]].select_ucb_factored(
                        th_fields,
                        X_list,
                        delta=0.1,
                        global_budget=eval_budget,
                        max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                        deterministic=False,
                    )
                else:
                    x_eval, sel_info = bandits[years[year_idx]].select_ts_factored(
                        th_fields,
                        X_list,
                        global_budget=eval_budget,
                        max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
                        seed=args.seed,
                        deterministic=False,
                        # temperature=0.5,
                    )

                # Budget violation check (only meaningful in reduced)
                if scenario == "reduced":
                    applied = float((torch.as_tensor(env.max_budgets,
                                                     dtype=torch.float32) - (
                                             x_eval.detach().cpu() * 10)).sum().item())
                    if applied > float(eval_budget) + 1e-3:
                        print(
                            f"\n\n[WARN] budget violated: applied={applied:.3f} > B={float(eval_budget):.3f}\n")

                th_np, raw_reward, done, _, infos = env.step(x_eval)

                season_year = int(
                    infos["AgentInfos"][env.agents_order[0]]["SeasonYear"][-1])

                if season_year is not None:
                    info_dict[int(season_year)] = infos["AgentInfos"]

                rewards.append(float(raw_reward))
                print(f"test season_year: {season_year}, reward: {raw_reward}")

                if comet_experiment:
                    comet_experiment.log_metrics(
                        {f"reward/{scenario}/test_year:{season_year}/raw": float(
                            raw_reward)},
                        step=test_step,
                    )
                    fig = plot_results(
                        infos["AgentInfos"],
                        variable_list=["DVS", "Profit", "Action", "Yield", "BudgetLeft"],
                        show=False,
                    )
                    comet_experiment.log_figure(
                        figure_name=f"image/{scenario}/plot_year:{season_year}",
                        figure=fig,
                        step=test_step,
                    )
                    plt.close(fig)

                    # increase year in while loop
                    year_idx += 1
            else:
                if comet_experiment:
                    fig = plot_results_daisy_chained(
                        info_dict,  # dict[season_year] -> AgentInfos
                        variable_list=["DVS", "Profit", "Action", "Yield", "BudgetLeft",
                                       "NAVAIL"],
                        show=False,
                    )
                    comet_experiment.log_figure(
                        figure_name=f"image/{scenario}/plot_eval",
                        figure=fig,
                        step=test_step,
                    )
                    plt.close(fig)

            # Return this
            env.unwrapped.farm.set_print_season_year(None)
            # Aggregate scenario score
            sum_reward = float(np.sum(rewards))
            if comet_experiment:
                comet_experiment.log_metric(f"Metric/reward/{scenario}/sum", sum_reward,
                                            step=test_step)

                # Always save the latest eval info_dict for this scenario
                latest_pickle_path = os.path.join(
                    _DEFAULT_LOGDIR,
                    f"Bandit_{args.model_dir}_{scenario}",
                    f"bandit_{region}_{farm_id}_info_{test_step}.pkl",
                )
                os.makedirs(os.path.dirname(latest_pickle_path), exist_ok=True)
                with open(latest_pickle_path, "wb") as f:
                    pickle.dump(info_dict, f)

                if comet_experiment:
                    comet_experiment.log_asset(
                        file_data=latest_pickle_path,
                        file_name=f"bandit_{region}_{farm_id}_{scenario}_info.pkl",
                        step=test_step,
                    )

                # Track + save best eval per scenario per seed
                if sum_reward > best_eval_sum[scenario]:
                    best_eval_sum[scenario] = sum_reward
                    best_eval_step[scenario] = int(test_step)

                    os.makedirs(
                        os.path.join(
                            _DEFAULT_LOGDIR,
                            f"Bandit_{args.model_dir}_{scenario}_BEST_s{args.seed}"
                        ), exist_ok=True
                    )

                    best_pickle_path = os.path.join(
                        _DEFAULT_LOGDIR,
                        f"Bandit_{args.model_dir}_{scenario}_BEST_s{args.seed}",
                        f"bandit_{region}_{farm_id}_BEST.pkl",
                    )
                    os.makedirs(os.path.dirname(best_pickle_path), exist_ok=True)
                    with open(best_pickle_path, "wb") as f:
                        pickle.dump(info_dict, f)

                    best_eval_pickle[scenario] = best_pickle_path

                    print(
                        f"[BEST] New best {scenario}: sum_reward={sum_reward:.6f} at eval_step={test_step}")
                    if comet_experiment:
                        comet_experiment.log_metrics(
                            {
                                f"reward/{scenario}/best_sum_reward": float(sum_reward),
                            },
                            step=test_step,
                        )
                        comet_experiment.log_asset(
                            file_data=best_pickle_path,
                            file_name=f"bandit_{region}_{farm_id}_{scenario}_BEST_info.pkl",
                            step=test_step,
                        )

        test_step += test_per_round
        for bandit in bandits.values():
            set_subbandits_train_mode(True)

        # Save model
        farm_name = f"{region}_{farm_id}" if region is not None else None

        for y, bandit in bandits.items():
            file_dir = bandit.save(
                seed=args.seed,
                t=t,
                name=f"{y}_" + log_folder_name,
                args=args,
                rms=rms,
                farm_id=farm_name,
            )
            if comet_experiment:
                comet_experiment.log_asset(
                    file_dir,
                    file_name=f"{y}_s{args.seed}_{args.method}_bandit_{t}",
                    step=t,
                )
    return bandit, test_step, train_raw_reward


def singe_year_train(bandit, env, t: int, args, rms: BanditNormalizer, rng, method, test_years: list[int],
                     hist_size_total, farm_id: int | None, region: str | None, comet_experiment):
    if region is not None or farm_id is not None:
        env.get_rotation_year(rng.choice(test_years))

    theta_np, env_info = env.reset(
        options={
            "year": rng.choice(env.years),
            "random_initial_conditions": True,
        },
        seed=args.seed,
    )
    if comet_experiment:
        comet_experiment.log_metric("episode/year", int(env_info["year"]))

    # Normalize (flat) context
    theta_np = rms.norm_theta(theta_np)

    # Unflatten to per-field theta matrix
    theta_fields_np = env.unflatten_context_per_field(theta_np)
    theta_fields = torch.from_numpy(theta_fields_np.astype(np.float32))

    # FULL candidate set per field (tiny)
    X_list = env.build_full_candidates_per_field()

    # Train occasionally
    train_every = int(getattr(args, "train_every", 1))
    if t % train_every == 0:
        steps = int(getattr(args, "bandit_epochs", 100))
        loss_val = float(bandit.train_step(steps=steps, lr=args.bandit_lr))
        n_hist = hist_size_total()
        loss_per_sample = loss_val / max(n_hist, 1)
        print(f"round {t}, loss: {loss_per_sample} (train_steps={steps})")
        if comet_experiment:
            comet_experiment.log_metric("loss", loss_per_sample, step=t)

    # Select action vector
    if method == "ucb":
        x_t, info = bandit.select_ucb_factored(
            theta_fields,
            X_list,
            delta=0.1,
            global_budget=float(env.global_budget),
            max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
        )
    else:
        x_t, info = bandit.select_ts_factored(
            theta_fields,
            X_list,
            global_budget=float(env.global_budget),
            max_budgets=torch.as_tensor(env.max_budgets, dtype=torch.float32),
        )

    # violation check
    if getattr(bandit, "use_farm_budget", False):
        applied = float((torch.as_tensor(env.max_budgets, dtype=torch.float32) - (x_t.detach().cpu() * 10.0)).sum().item())
        if applied > float(env.global_budget) + 1e-3:
            print(f"[WARN] budget violated: applied={applied:.3f} > B={float(env.global_budget):.3f}")

    # Step env (x_t is vector length n_fields)
    _, reward_env, _, _, step_info = env.step(x_t)
    n_surp_infos = [step_info['AgentInfos'][a]["Nsurp"][-1] for a in env.unwrapped.agents_order]
    nue_infos = [step_info['AgentInfos'][a]["Nue"][-1] for a in env.unwrapped.agents_order]
    print(f"reward: {reward_env}")
    if comet_experiment:
        comet_experiment.log_metrics({"reward/train/reward": float(reward_env)}, step=t)

    # Update your rolling historical context
    # env.add_stats_to_context(env.filter_historical_info(step_info["AgentInfos"]))

    # Per-field reward components (consistent with env._get_reward())
    y_fields = env.compute_per_field_rewards(n_surp_infos, nue_infos)

    # Optional observation noise (match your previous habit if desired)
    noise = float(getattr(args, "reward_noise", 0.005))
    if noise > 0:
        y_fields = y_fields + noise * rng.randn(len(y_fields)).astype(np.float32)

    # Update factored bandit
    bandit.update_factored(theta_fields, x_t, y_fields)