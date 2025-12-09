from comet_ml import Experiment

import torch
import numpy as np

from cropgymzoo.utils.agent_helpers import min_max_normalize


def run_eval_allocator(
        env,
        bandit,
        year,
        rms,
        experiment: Experiment = None,
        step: int = None,
        seed=107,
        streaming=False,
        method='ucb',
        candidate_size=24000,
        scenario='full',
):

    """Run eval allocator."""
    theta_t, env_info = env.reset(
        options={
            'year': year,
        },
        seed=seed
    )

    # normalize
    theta_t = rms.norm(theta_t)

    # convert to numpy
    theta_t = torch.from_numpy(theta_t)

    allocation_actions = env.sample_super_arms(
            n_candidates=candidate_size,
            reduced=scenario == 'reduced',
        )

    if not streaming:
        # candidate set for actions; sampled from the super_arms array
        x_cand = allocation_actions
        x_cand = torch.from_numpy(x_cand)

        if method == 'ucb':
            # pick by UCB (or switch to bandit.select_ts(...))
            x_t, selection_info = bandit.select_ucb(
                theta_t,
                x_cand,
                delta=0.1,
                deterministic=True,
            )
        else:
            x_t, selection_info = bandit.select_ts(
                theta_t,
                x_cand,
                deterministic=True,
            )
    else:
        x_t, best = bandit.select_ucb_streaming(
            theta_t,
            torch.from_numpy(allocation_actions),
            delta=0.1,
            deterministic=True,
        )
    if experiment is not None:
        for i in range(x_t.shape[0]):
            experiment.log_metric(
                f"test/action/year-{year}/action-field-{i+1}",
                x_t[i].item(),
                step=step
            )
        experiment.log_metric(
            f"test/action/year-{year}/action_vector",
            f"{str(x_t)}",
            step=step,
        )
    if isinstance(x_t, np.ndarray):
        x_t = torch.from_numpy(x_t)

    # run env and normalize reward
    _, reward_env, _, _, step_info = env.step(x_t)
    normalized_reward = min_max_normalize(float(reward_env))

    return reward_env, normalized_reward, step_info['AgentInfos']
