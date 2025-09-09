import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from datetime import datetime

import comet_ml
from comet_ml import Experiment

import torch

import numpy as np

from cropgymzoo import _BASE_PATH, _SOURCE_PATH
from cropgymzoo.agents.nn_acgp import NNAGPBandit, SelectionInfo

from cropgymzoo.envs.allocation_env import AllocationBandit


def min_max_normalize(x, min_val=0, max_val=300000) -> float:
    """Scale from [min_val, max_val] -> [0, 1]."""
    return (x - min_val) / (max_val - min_val)

def min_max_denormalize(x_norm, min_val=0, max_val=300000) -> float:
    """Scale from [0, 1] -> [min_val, max_val]."""
    return x_norm * (max_val - min_val) + min_val

def _setup_bandit_comet(args):
    if not os.path.isdir(os.path.join(_BASE_PATH, 'comet')):
        print("Not using comet!")
        return

    with open(os.path.join(_BASE_PATH, 'comet', 'api'), 'r') as f:
        api_key = f.readline()
    # prefer env vars; fall back to sensible defaults
    experiment = Experiment(
        api_key=api_key,
        project_name="cropgymzoo_allocation_experiments",
        workspace="cropgymzoo",
        log_code=True,
        auto_metric_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_param_logging=True,
        auto_histogram_tensorboard_logging=True
    )

    experiment.log_code(folder=_SOURCE_PATH)

    name = f"s{args.seed}-allocation-agent-{datetime.now():%m%d-%H%M}"
    experiment.set_name(name)
    experiment.add_tag("allocation-bandit")
    experiment.add_tag("NN-ACGP")

    # log hyperparameters (robustly)
    experiment.log_parameters({k: v for k, v in vars(args).items()})

    return experiment

def log_selection_info(experiment: Experiment, info: SelectionInfo, t):
    experiment.log_histogram_3d(
        info.mu,
        name="mu",
        step=t
    )
    experiment.log_histogram_3d(
        info.std,
        name="std",
        step=t
    )
    if info.ucb is not None:
        experiment.log_histogram_3d(
            info.ucb,
            name="ucb",
            step=t
        )
    if info.beta_t:
        experiment.log_metric("beta_t", info.beta_t, step=t)
    if info.sampled_vals is not None:
        experiment.log_histogram_3d(
            info.sampled_vals,
            name="sampled_vals",
            step=t
        )

def log_candidates(experiment: Experiment, cand: torch.Tensor, t):
    fields = cand.shape[1]

    for field in range(fields):
        experiment.log_histogram_3d(
            f"field-{field+1}-candidates",
            cand[:, field].detach().cpu().numpy(),
            step=t
        )


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

    # put the bandit algorithm here
    bandit = NNAGPBandit(
        d_theta=d_theta,
        d_x=d_x,
        m=8,
        Q=1,
        device=torch.device("cpu")
    )

    # make action candidates for each round. Get super arms and randomly sample
    action_candidates = env.super_arms
    num_candidates = args.action_candidate_length

    # put the training loop here
    for t in range(1, args.rounds + 1):
        theta_t, env_info = env.reset(
            options={
                'year': rng.choice(env.years),
            },
            seed=args.seed
        )
        if comet_experiment:
            comet_experiment.log_metric("episode/year", int(env_info['year']), step=t)

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
            file_dir = bandit.save(args.seed, t, )
            if comet_experiment:
                comet_experiment.log_asset(
                    file_dir,
                    file_name=f"s{args.seed}_bandit_{t}",
                    step=t,
                )

    print("Training Complete!")

