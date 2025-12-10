from comet_ml import Experiment

import torch
import numpy as np

from cropgymzoo.utils.agent_helpers import min_max_normalize


def _score_candidates_mean(
    bandit,
    theta_t: torch.Tensor,
    Xc: torch.Tensor,
    method: str = "ucb",
):
    """
    Use the current posterior to score candidates by posterior mean.

    method="ucb" or "ts". For both, we call the deterministic variant,
    which returns SelectionInfo with mu (posterior mean).
    """
    if method == "ucb":
        _, sel = bandit.select_ucb(
            theta_t,
            Xc,
            delta=0.1,
            deterministic=True,  # beta_t = 0 ⇒ ucb = mu
        )
        scores = sel.mu  # (M,)
    else:
        _, sel = bandit.select_ts(
            theta_t,
            Xc,
            deterministic=True,  # deterministic TS = greedy on mu
        )
        scores = sel.mu  # (M,)

    return scores  # torch.Tensor, shape (M,)


def _hill_climb_topk(
    env,
    bandit,
    theta_t: torch.Tensor,
    initial_candidates: np.ndarray,
    method: str = "ucb",
    scenario: str = "full",
    top_k: int = 5,
    hill_steps: int = 3,
    neighbors_per_step: int = 32,
    rng: np.random.RandomState | None = None,
) -> torch.Tensor:
    """
    Given an initial batch of candidates, pick top-K by posterior mean, then
    run a small discrete hill-climb around each of them using env.sample_neighbors.

    Returns the best arm (as a torch.Tensor) found over all starts.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    # Torch-ify initial candidates
    Xc = torch.from_numpy(initial_candidates.astype(np.float32))
    scores = _score_candidates_mean(bandit, theta_t, Xc, method=method)

    M = Xc.shape[0]
    k = min(top_k, M)
    # top-k indices by posterior mean
    vals, idxs = torch.topk(scores, k=k)
    # global best across all restarts
    best_overall_score = -float("inf")
    best_overall_x = None

    reduced_flag = (scenario == "reduced")

    for start_idx in idxs.tolist():
        x_cur = Xc[start_idx].clone()
        best_score = scores[start_idx].item()

        # small number of hill-climb steps
        for _ in range(hill_steps):
            # sample neighbors around current best
            neigh_np = env.sample_neighbors(
                center=x_cur.detach().cpu().numpy(),
                n_neighbors=neighbors_per_step,
                reduced=reduced_flag,
            )
            neigh = torch.from_numpy(neigh_np.astype(np.float32))

            neigh_scores = _score_candidates_mean(bandit, theta_t, neigh, method=method)
            max_val, max_idx = torch.max(neigh_scores, dim=0)

            # if we found a better neighbor, move there and continue
            if max_val.item() > best_score:
                best_score = max_val.item()
                x_cur = neigh[max_idx].clone()
            else:
                # local optimum (under this neighbor scheme)
                break

        # track best across all restarts
        if best_score > best_overall_score or best_overall_x is None:
            best_overall_score = best_score
            best_overall_x = x_cur

    return best_overall_x

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
        use_hill_climb: bool = True,
        top_k: int = 5,
        hill_steps: int = 3,
        neighbors_per_step: int = 32,
):
    # ensure rotation is correct
    env.get_rotation_year(year)

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
        if use_hill_climb:
            # global + local search
            x_t = _hill_climb_topk(
                env=env,
                bandit=bandit,
                theta_t=theta_t,
                initial_candidates=allocation_actions,
                method=method,
                scenario=scenario,
                top_k=top_k,
                hill_steps=hill_steps,
                neighbors_per_step=neighbors_per_step,
                rng=np.random.RandomState(seed),
            )
            selection_info = None  # could return info if you want to log
        else:
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
