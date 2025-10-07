import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

import numpy as np

import datetime

from cropgymzoo.eval_allocator import run_eval_allocator
from cropgymzoo.agents.nn_agp import NNAGPBandit
from cropgymzoo.utils.agent_helpers import min_max_normalize
from cropgymzoo.utils.callbacks import _setup_bandit_comet, log_selection_info, log_model_histograms
from cropgymzoo.envs.allocation_env import AllocationBandit
from tianshou.utils.statistics import RunningMeanStd


def train_allocator(args):
    log_folder_name = f"NN-AGP-Bandit_{datetime.datetime.now():%m%d-%H%M}"
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

    # misc
    rng = np.random.RandomState(args.seed)
    torch.set_default_dtype(torch.float32)

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

    # make action candidates for each round. Get super arms and randomly sample
    action_candidates = env.super_arms
    num_candidates = args.action_candidate_length

    # initialize running mean
    rms = RunningMeanStd()

    test_step = 0

    # put the training loop here
    for t in range(1, args.rounds + 1):
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
            x_t, selection_info = bandit.select_ucb(theta_t, x_cand, delta=0.1)
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
            years: list = [2000]

            for year in years:
                raw_reward, normalized_reward = run_eval_allocator(
                    env=env,
                    bandit=bandit,
                    year=year,
                    rms=rms,
                    experiment=comet_experiment,
                    step=t,
                    candidate_size=50_0000,
                )
                print(f"test year: {year}, reward: {raw_reward}")
                if comet_experiment:
                    comet_experiment.log_metrics(
                        {
                            f"reward/test_year:{year}/raw": float(raw_reward),
                            f"reward/test_year:{year}/normalized": float(normalized_reward),
                        },
                        step=test_step,
                    )

            test_step += 10

            bandit.model.train()

        # save the model iteratively
        if t % 10 == 0:
            file_dir = bandit.save(
                seed=args.seed,
                t=t,
                name=log_folder_name,
                args=args,
                rms=rms,
            )

            if comet_experiment:
                comet_experiment.log_asset(
                    file_dir,
                    file_name=f"s{args.seed}_bandit_{t}",
                    step=t,
                )

    print("Training Complete!")

