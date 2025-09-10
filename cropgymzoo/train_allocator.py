import warnings

from cropgymzoo.utils.callbacks import _setup_bandit_comet, log_selection_info

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

import numpy as np

from cropgymzoo.agents.nn_acgp import NNAGPBandit
from cropgymzoo.envs.allocation_env import AllocationBandit
from tianshou.utils.statistics import RunningMeanStd


def min_max_normalize(x, min_val=0, max_val=300000) -> float:
    """Scale from [min_val, max_val] -> [0, 1]."""
    return (x - min_val) / (max_val - min_val)

def min_max_denormalize(x_norm, min_val=0, max_val=300000) -> float:
    """Scale from [0, 1] -> [min_val, max_val]."""
    return x_norm * (max_val - min_val) + min_val


def train_allocator(args):
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

    m = d_theta//10 + d_x//3 + 3
    print(f"d_theta: {d_theta}, d_x: {d_x}. So, m: {m}")

    # put the bandit algorithm here
    bandit = NNAGPBandit(
        d_theta=d_theta,
        d_x=d_x,
        m=m,
        Q=1,
        lr=args.bandit_lr,
        device=torch.device("cpu")
    )

    # make action candidates for each round. Get super arms and randomly sample
    action_candidates = env.super_arms
    num_candidates = args.action_candidate_length

    # initialize running mean
    rms = RunningMeanStd()

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

        # candidate set for actions; sampled from the super_arms array
        indices = torch.randperm(action_candidates.shape[0])[:num_candidates]
        x_cand = action_candidates[indices]
        x_cand = torch.from_numpy(x_cand)

        # train the surrogate a bit on accumulated data
        loss_val = bandit.train_step(steps=args.bandit_epochs, lr=args.bandit_lr)
        print(f"round {t}, loss: {loss_val}")
        if comet_experiment:
            comet_experiment.log_metric("loss", loss_val, step=t)
            comet_experiment.log_histogram_3d(
                x_cand.T,
                name="x_cand",
                step=t,
            )


        # pick by UCB (or switch to bandit.select_ts(...))
        x_t, selection_info = bandit.select_ucb(theta_t, x_cand, delta=0.1)
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t)
        if comet_experiment:
            log_selection_info(comet_experiment, selection_info, t)

        # run env and normalize reward
        _, reward_env, _, _, step_info = env.step(x_t)
        normalized_reward = min_max_normalize(float(reward_env))
        if comet_experiment:
            comet_experiment.log_metrics(
                {
                    "reward/normalised": float(normalized_reward),
                    "reward/reward": float(reward_env),
                },
                step=t
            )

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


        # save the model iteratively
        if t % 50 == 0:
            file_dir = bandit.save(
                seed=args.seed,
                t=t,
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

