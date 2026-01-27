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
        use_hill_climb: bool = False,
        top_k: int = 5,
        hill_steps: int = 3,
        neighbors_per_step: int = 32,
        elite_candidates_np: np.ndarray | None = None,
        crop_aware_candidates: bool = True,
        model_informed_candidates: bool = False,
        model_informed_ratio: float = 0.6,
        model_informed_eps: float = 0.10,
        model_informed_rand: int = 256,
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
    theta_t = rms.norm_theta(theta_t)

    # convert to numpy
    theta_t = torch.from_numpy(theta_t)

    reduced_flag = (scenario == 'reduced')

    if crop_aware_candidates:
        # Mixture sampler during evaluation:
        #  - model-informed sampler (learned categorical probs from training) if available
        #  - crop-grid sampler as a robust backstop
        #  - small random set for exploration safety
        if model_informed_candidates and hasattr(env, "sample_model_informed_super_arms"):
            n_model = int(round(float(model_informed_ratio) * int(candidate_size)))
            n_model = max(0, min(int(candidate_size), n_model))
            n_crop = int(candidate_size) - n_model

            allocation_model = env.sample_model_informed_super_arms(
                n_candidates=n_model,
                reduced=reduced_flag,
                rng=np.random.RandomState(seed),
                eps=float(model_informed_eps),
            )

            allocation_crop = env.sample_crop_grid_super_arms(
                n_candidates=n_crop,
                reduced=reduced_flag,
                include_center=True,
                unique=True,
            )

            # allocation_rand = env.sample_super_arms(
            #     n_candidates=min(int(model_informed_rand), int(candidate_size)),
            #     reduced=reduced_flag,
            #     rng=np.random.RandomState(seed + 123),
            # )

            allocation_actions = np.vstack([
                allocation_model.astype(np.float32),
                allocation_crop.astype(np.float32),
                # allocation_rand.astype(np.float32),
            ])

            # Deduplicate (small enough for eval)
            allocation_actions = np.unique(allocation_actions, axis=0)
        else:
            allocation_actions = env.sample_crop_grid_super_arms(
                n_candidates=candidate_size,
                reduced=reduced_flag,
                include_center=True,
                unique=True,
            )
    else:
        allocation_actions = env.sample_super_arms(
            n_candidates=candidate_size,
            reduced=reduced_flag,
        )

    # Inject elite candidates into eval candidate set (if provided)
    if elite_candidates_np is not None:
        try:
            allocation_actions = np.vstack([
                elite_candidates_np.astype(np.float32),
                allocation_actions.astype(np.float32),
            ])
        except Exception:
            # If something mismatches, skip injection instead of crashing
            pass

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

    return reward_env, normalized_reward, step_info['AgentInfos'], x_t
