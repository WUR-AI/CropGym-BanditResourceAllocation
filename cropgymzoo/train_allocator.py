import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

import numpy as np
import os

import datetime
import pickle

from cropgymzoo.eval_allocator import run_eval_allocator
from cropgymzoo.agents.nn_agp import NNAGPBandit
from cropgymzoo.utils.agent_helpers import min_max_normalize
from cropgymzoo.utils.callbacks import _setup_bandit_comet, log_selection_info, log_model_histograms
from cropgymzoo.envs.allocation_env import AllocationBandit
from cropgymzoo import _DEFAULT_LOGDIR
from tianshou.utils.statistics import RunningMeanStd


def farm_int_mapper(x: int):
    """
    Maps a global farm index (0–52) to (region, farmer_id).

    Regions:
        Gelderland: 0–11  (12 farms)
        Groningen: 12–24  (13 farms)
        Zeeland:   25–52  (27 farms)

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

    # initialize env
    env = AllocationBandit(
        warm_up_eps=2,
        args=args,
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

    print(f"Training Farm {region}_{farm_int}!")
    print(f"Using {args.model_dir} RL policy!")

    training_years = list(range(2000, 2020))

    log_folder_name = f"Bandit_{region}_{farm_int}_{datetime.datetime.now():%m%d}"
    # initialize comet if using
    comet_experiment = None
    if args.use_comet:
        comet_experiment = _setup_bandit_comet(args)
        comet_experiment.add_tags([f"{region}_{farm_id}"])
        comet_experiment.add_tag(f"{args.method}")
        comet_experiment.add_tag(f"{args.model_dir}")

    env = AllocationBandit(
        warm_up_eps=2,
        args=args,
        seed=args.seed,
        flat_context=True,
        years=training_years,
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
        device=torch.device("cpu")
    )

    training_loop(env, bandit, args, comet_experiment, log_folder_name, region=region, farm_id=farm_id)

    print(f"Training for Farm {region}_{farm_id} Complete!")



def training_loop(env: AllocationBandit, bandit: NNAGPBandit, args, comet_experiment = None,
                log_folder_name: str = None, region: str | None = None, farm_id: int | None = None):

    # misc
    rng = np.random.RandomState(args.seed)
    torch.set_default_dtype(torch.float32)

    # make action candidates for each round. Get super arms and randomly sample
    action_candidates = env.super_arms
    num_candidates = args.action_candidate_length

    # initialize running mean
    rms = RunningMeanStd()

    test_step = 0

    method = args.method

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
        rms.update(theta_t)
        theta_t = rms.norm(theta_t)

        # convert to numpy
        theta_t = torch.from_numpy(theta_t)

        if not args.streaming and args.action_candidate_length < action_candidates.shape[0]:
            # candidate set for actions; sampled from the super_arms array
            indices = torch.randperm(action_candidates.shape[0])[:num_candidates]
            x_cand = action_candidates[indices]
        else:
            x_cand = action_candidates
        x_cand = torch.from_numpy(x_cand)

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
                x_t, selection_info = bandit.select_ts(theta_t, x_cand, delta=0.1)
            if isinstance(x_t, np.ndarray):
                x_t = torch.from_numpy(x_t)
            if comet_experiment:
                log_selection_info(comet_experiment, selection_info, t)
        else:
            x_t, best = bandit.select_ucb_streaming(theta_t, torch.from_numpy(action_candidates), delta=0.1)
            if isinstance(x_t, np.ndarray):
                x_t = torch.from_numpy(x_t)
            if comet_experiment:
                comet_experiment.log_metrics(best, step=t) if best is not None else None

        # run env and normalize reward
        _, reward_env, _, _, step_info = env.step(x_t)
        normalized_reward = min_max_normalize(float(reward_env))
        if comet_experiment:
            comet_experiment.log_metrics(
                {
                    "reward/train/normalized": float(normalized_reward),
                    "reward/train/reward": float(reward_env),
                },
                step=t
            )

        # update rolling historical average
        env.add_stats_to_context(step_info['AgentInfos'])

        # observe noisy reward
        y_t = float(normalized_reward + 0.05 * torch.randn(()))
        if comet_experiment:
            comet_experiment.log_metric(
                "reward/noisy",
                float(y_t),
                step=t
            )
        bandit.update(theta_t, x_t, y_t)

        # eval the allocator after rounds
        if t % 10 == 0:
            # test bandit
            bandit.model.eval()
            # edit?
            years: list = [2020, 2021, 2022, 2023, 2024]

            scenarios = ['full', 'reduced']
            for scenario in scenarios:
                info_dict = {}
                for year in years:
                    raw_reward, normalized_reward, infos = run_eval_allocator(
                        env=env,
                        bandit=bandit,
                        year=year,
                        rms=rms,
                        experiment=comet_experiment,
                        step=t,
                        method=method,
                        candidate_size=9_000_000_000,
                        scenario=scenario,
                    )
                    info_dict[year] = infos
                    print(f"test year: {year}, reward: {raw_reward}")
                    if comet_experiment:
                        comet_experiment.log_metrics(
                            {
                                f"reward/scenario_{scenario}/test_year:{year}/raw": float(raw_reward),
                                f"reward/scenario_{scenario}/test_year:{year}/normalized": float(normalized_reward),
                            },
                            step=test_step,
                        )
                if comet_experiment:
                    pickle_path = os.path.join(_DEFAULT_LOGDIR, log_folder_name, f"bandit_{region}_{farm_id}_{scenario}_info.pkl")
                    with open(pickle_path, "wb") as f:
                        pickle.dump(info_dict, f)
                    comet_experiment.log_asset(
                        file_data=pickle_path,
                        file_name=f"bandit_{region}_{farm_id}_{scenario}_info.pkl",
                        step=test_step,
                    )

            test_step += 10

            bandit.model.train()

        farm_name = f"{region}_{farm_id}" if region is not None else None

        # save the model iteratively
        if t % 5 == 0:
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
