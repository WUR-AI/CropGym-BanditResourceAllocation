import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import yaml

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
from cropgymzoo import _DEFAULT_LOGDIR, _DEFAULT_RESULTSDIR, _SCENARIO_PATH
from cropgymzoo.utils.plotters import plot_results, plot_results_daisy_chained


class BanditNormalizer:
    def __init__(
        self,
        env,
        rng,
        seed: int | None = None,
        clip: float = 5.0,
        n_calib: int = 100,
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
            env.get_rotation_year(rng.choice([2020, 2021, 2022, 2023, 2024]))
            theta_np, _ = env.reset(
                options={"year": rng.choice(env.years)},
                seed=seed or 0,
            )
            theta_buf.append(theta_np.astype(np.float32))

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

    bandit_action_mode = getattr(args, "bandit_action_mode", "factored")  # 'joint' or 'factored'

    if bandit_action_mode == "factored":
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
            use_farm_budget=True
        )

        training_loop_factored(
            env, bandit, args, comet_experiment, log_folder_name, region=region, farm_id=farm_id
        )

    else:
        bandit = NNAGPBandit(
            d_theta=d_theta,
            d_x=d_x,
            m=m,
            Q=args.q,
            lr=args.bandit_lr,
            device=torch.device("cpu"),
            posterior_type=args.bandit_posterior,
            coreset_size=args.coreset_size,
            coreset_mode=args.coreset_mode,
            action_mode="joint",
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

    rms = BanditNormalizer(env, rng)

    test_step = 0

    method = args.method

    # ---------------- Model-informed candidate sampler ----------------
    model_informed = True
    model_informed_ratio = 0.8  # fraction of candidates coming from learned sampler
    model_informed_topk = 16512  # use top-k UCB candidates to update sampler
    model_informed_alpha = 0.3  # EMA update speed
    model_informed_eps = 0.20  # exploration mixing with uniform

    # init env sampler once
    if model_informed and hasattr(env, "init_model_sampler"):
        env.init_model_sampler(eps=model_informed_eps, alpha=model_informed_alpha)

    # ---------------- Elite candidate memory ----------------
    # Keep the best action seen so far (by *raw* env reward) and its neighbors,
    # and always inject them into later candidate sets.
    elite_enabled = getattr(args, "elite_enabled", True)
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
        """Deduplicate rows (small tensors only).

        NOTE: This is intentionally kept for SMALL tensors (like an elite cloud),
        but should NOT be used on the full candidate set each round.
        """
        if X is None or X.numel() == 0:
            return X
        return torch.unique(X, dim=0)

    def _filter_new_elite_rows(elite_X: torch.Tensor, Xc: torch.Tensor, decimals: int = 6) -> torch.Tensor:
        """Return rows from elite_X that are NOT already present in Xc.

        We avoid `torch.unique` on the full stacked matrix because it's expensive.
        This function only builds a key-set from Xc once and filters elite rows.
        """
        if elite_X is None or elite_X.numel() == 0:
            return elite_X
        if Xc is None or Xc.numel() == 0:
            return elite_X

        # Build key-set for current candidates (Xc)
        # Xc is typically ~30k rows; this is fast enough in Python.
        Xc_np = np.round(Xc.detach().cpu().numpy().astype(np.float64), decimals)
        Xc_keys = set(map(tuple, Xc_np.tolist()))

        elite_np = np.round(elite_X.detach().cpu().numpy().astype(np.float64), decimals)
        keep_rows = []
        for row in elite_np:
            key = tuple(row.tolist())
            if key not in Xc_keys:
                keep_rows.append(row)

        if not keep_rows:
            return elite_X[:0]

        out = torch.from_numpy(np.asarray(keep_rows, dtype=np.float32))
        return out

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
            "center": x_t.detach().clone(),
            "cands": cand,
        })

        # Keep only top-K by reward
        items.sort(key=lambda z: z["reward"], reverse=True)
        del items[elite_top_k:]

        # Rebuild aggregated candidates used for injection
        _rebuild_elite_cands(scenario)

    def _inject_elite(scenario: str, Xc: torch.Tensor) -> torch.Tensor:
        """Prepend elite candidates to the sampled set without expensive global dedup.

        We only filter out elite rows that already exist in the current Xc.
        This keeps elite injection effective while avoiding `torch.unique` over ~30k+ rows.
        """
        if not elite_enabled:
            return Xc

        elite_c = elite[scenario]["cands"]
        if elite_c is None or elite_c.numel() == 0:
            return Xc

        elite_c = elite_c.to(dtype=Xc.dtype)

        # Filter only NEW elite rows (not already in Xc)
        elite_new = _filter_new_elite_rows(elite_c, Xc, decimals=6)
        if elite_new is None or elite_new.numel() == 0:
            return Xc

        # Cap how many elite rows we inject each time to avoid ballooning candidate sets
        # (keep the most recent/top ones, which are already sorted in _rebuild_elite_cands)
        max_inject = int(getattr(args, "elite_inject_max", 2000))
        if elite_new.shape[0] > max_inject:
            elite_new = elite_new[:max_inject]

        return torch.vstack([elite_new.to(Xc.device), Xc])

    # --------------------------------------------------------

    # put the training loop here
    for t in range(1, args.rounds + 1):
        if region is not None or farm_id is not None:
            env.get_rotation_year(rng.choice([2020, 2021, 2022, 2023, 2024]))
        theta_t, env_info = env.reset(
            options={
                'year': rng.choice(env.years),
                "random_initial_conditions": True,
            },
            seed=args.seed
        )
        if comet_experiment:
            comet_experiment.log_metric("episode/year", int(env_info['year']))

        # normalize
        # rms.update(theta_t)
        # theta_t = rms.norm(theta_t)
        theta_t = rms.norm_theta(theta_t)

        # ---- candidate sampling uses numpy theta_t ----
        # (keep theta_t as numpy here)

        # ---- candidate sampling ----
        # Crop-aware candidates: bias each field's reduction values toward crop-typical regimes.
        # This helps the bandit learn much faster when crop identity strongly drives good actions.
        use_crop_aware = bool(getattr(args, "crop_aware_candidates", True))

        if use_crop_aware:
            if model_informed and hasattr(env, "sample_model_informed_super_arms"):
                n_model = int(round(model_informed_ratio * num_candidates))
                n_crop = num_candidates - n_model

                x_model = env.sample_model_informed_super_arms(
                    n_candidates=n_model,
                    rng=rng,
                    reduced=False,
                    eps=model_informed_eps,
                )

                x_crop = env.sample_crop_grid_super_arms(
                    n_candidates=n_crop,
                    rng=rng,
                    reduced=False,
                    n_steps=int(getattr(args, "crop_grid_steps", 10)),
                    include_center=True,
                    max_cartesian=int(getattr(args, "crop_grid_max_cartesian", 200_000)),
                    unique=True,
                )

                # small random fallback (keeps exploration alive)
                x_rand = env.sample_super_arms(
                    n_candidates=256,
                    reduced=False,
                    rng=rng,
                )

                x_cand = np.vstack([x_model, x_crop, x_rand]).astype(np.float32)
                x_cand = np.unique(x_cand, axis=0)  # okay at ~30k
            else:
                x_cand = env.sample_crop_grid_super_arms(
                    n_candidates=num_candidates,
                    rng=rng,
                    reduced=False,
                    n_steps=int(getattr(args, "crop_grid_steps", 10)),
                    include_center=True,
                    max_cartesian=int(getattr(args, "crop_grid_max_cartesian", 200_000)),
                    unique=True,
                )
        else:
            x_cand = env.sample_super_arms(
                n_candidates=num_candidates,
                reduced=False,
                rng=rng,
            )

        # convert to torch
        theta_t = torch.from_numpy(theta_t)
        x_cand = torch.from_numpy(x_cand.astype(np.float32))

        # Always inject persistent elite candidates for training scenario="full"
        x_cand = _inject_elite("full", x_cand)

        train_every = int(getattr(args, "train_every", 1))
        if t % train_every == 0:
            # ---- train surrogate (EXPENSIVE for exact GP because of O(n^3) Cholesky)
            # Use fewer inner steps early and ramp up slowly.
            base_steps = int(getattr(args, "bandit_epochs", 100))

            loss_val = bandit.train_step(steps=base_steps, lr=args.bandit_lr)
            n = len(bandit.y_hist)
            loss_per_sample = loss_val / max(n, 1)
            print(f"round {t}, loss: {loss_per_sample}  (train_steps={base_steps})")

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

        # ---- model-informed sampler update (Option A) ----
        if model_informed and method == "ucb" and hasattr(env, "update_model_sampler_probs"):
            if selection_info.ucb is not None:
                env.update_model_sampler_probs(
                    X_candidates=x_cand.detach().cpu().numpy(),
                    scores=selection_info.ucb.detach().cpu().numpy(),
                    top_k=model_informed_topk,
                    alpha=model_informed_alpha,
                )

        # run env and normalize reward
        _, reward_env, _, _, step_info = env.step(x_t)
        # normalized_reward = min_max_normalize(float(reward_env))
        print(f"reward: {reward_env}")

        # Update elite memory using *raw* reward signal
        _update_elite(env, "full", x_t, float(reward_env))

        if elite_enabled and elite["full"]["items"]:
            best_center = elite["full"]["items"][0]["center"]  # we will store this
            env.set_elite_center_action(best_center.detach().cpu().numpy())

        if comet_experiment:
            comet_experiment.log_metrics(
                {
                    # "reward/train/normalized": float(normalized_reward),
                    "reward/train/reward": float(reward_env),
                },
                step=t
            )

        # update rolling historical average
        env.add_stats_to_context(env.filter_historical_info(step_info['AgentInfos']))

        # observe noisy reward
        y_t = float(float(reward_env) + 0.05 * torch.randn(()))
        if comet_experiment:
            comet_experiment.log_metric(
                "reward/raw",
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

            scenarios = [
                'full',
                # 'reduced'
            ]
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
                        ),
                        crop_aware_candidates=use_crop_aware,
                        model_informed_candidates=model_informed
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

    # Persistent best-eval trackers
    best_eval_sum = {"full": float("-inf"), "reduced": float("-inf")}
    best_eval_step = {"full": None, "reduced": None}
    best_eval_pickle = {"full": None, "reduced": None}

    def set_subbandits_train_mode(train: bool):
        # Your factored NNAGPBandit stores sub-bandits in bandit.sub_bandits
        for sb in getattr(bandit, "sub_bandits", []):
            sb.model.train(train)

    def hist_size_total() -> int:
        return int(sum(len(sb.y_hist) for sb in getattr(bandit, "sub_bandits", [])))

    for t in range(1, args.rounds + 1):
        # Keep your existing rotation-year behavior if you want it
        if region is not None or farm_id is not None:
            env.get_rotation_year(rng.choice([2020, 2021, 2022, 2023, 2024]))

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

        # Train surrogate occasionally
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
            applied = float((torch.as_tensor(env.max_budgets, dtype=torch.float32) - x_t.detach().cpu()).sum().item())
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

        # Periodic eval (daisy-chained multi-year evaluation)
        test_per_round = int(args.eval_steps)
        if t % test_per_round == 0:
            set_subbandits_train_mode(False)

            for scenario in ["full", "reduced"]:
                print(f"\n\nEval scenario: {scenario}\n")
                best_seed_reward = []
                for n in range(5):
                    seed = args.seed + n
                    years = [2020, 2021, 2022, 2023, 2024]
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
                                seed=seed,
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

                        if comet_experiment:
                            comet_experiment.log_metrics(
                                {f"s{seed}/reward/{scenario}/test_year:{season_year}/raw": float(raw_reward)},
                                step=test_step,
                            )
                            fig = plot_results(
                                infos["AgentInfos"],
                                variable_list=["DVS", "Profit", "Action", "Yield", "BudgetLeft"],
                                show=False,
                            )
                            comet_experiment.log_figure(
                                figure_name=f"s{seed}/image/{scenario}/plot_year:{season_year}",
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
                                figure_name=f"s{seed}/image/{scenario}/plot_eval",
                                figure=fig,
                                step=test_step,
                            )
                            plt.close(fig)

                    # Return this
                    env.unwrapped.farm.set_print_season_year(None)
                    # Aggregate scenario score
                    sum_reward = float(np.sum(rewards))
                    best_seed_reward.append(sum_reward)
                    if comet_experiment:
                        comet_experiment.log_metric(f"s{seed}/reward/{scenario}/sum", sum_reward, step=test_step)

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

                        # Track + save best eval per scenario
                        if sum_reward > best_eval_sum[scenario]:
                            best_eval_sum[scenario] = sum_reward
                            best_eval_step[scenario] = int(test_step)

                            os.makedirs(
                                os.path.join(
                                    _DEFAULT_LOGDIR,
                                    f"Bandit_{args.model_dir}_{scenario}_s{seed}"
                                ), exist_ok=True
                            )

                            best_pickle_path = os.path.join(
                                _DEFAULT_LOGDIR,
                                f"Bandit_{args.model_dir}_{scenario}_s{seed}",
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
                                        f"reward/{scenario}/best_at_step": int(test_step),
                                    },
                                    step=test_step,
                                )
                                comet_experiment.log_asset(
                                    file_data=best_pickle_path,
                                    file_name=f"bandit_{region}_{farm_id}_{scenario}_BEST_info.pkl",
                                    step=test_step,
                                )
                if comet_experiment:
                    comet_experiment.log_metric(f"Metrics/{scenario}/best_eval", max(best_seed_reward), step=test_step)

            test_step += test_per_round
            set_subbandits_train_mode(True)

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