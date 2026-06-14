import itertools
from typing import Iterable
import numpy as np


def _make_base_arms(self, cap: float = 1.0) -> dict[str, np.ndarray]:
    """
    Per-farm allowed reductions in kg/ha: [0, delta, 2*delta, ..., floor(cap*budget/delta)*delta].
    """
    max_budgets = {a: self.farm.get_per_parcel_max_budget(a) for a in self.farm.possible_agents}
    base = {}
    for agent, max_budget in max_budgets.items():
        half_budget = max(0.0, float(max_budget) * cap)
        q = int(np.floor(half_budget / float(self.bins)))
        # divide by 10 to lower range
        base[agent] = (np.arange(q + 1, dtype=np.float32) * float(self.bins)) / 10
    return base


def _make_super_arms(self, base_arms: dict[str, np.ndarray], reduced: bool= False, to_reduce: float = 100.0) -> np.ndarray:
    """
    Cartesian product over the per-farm base arms, returned as an array of shape (num_arms, N).
    Each row is an N-length reduction vector, e.g., [10, 40, 20, 30, 40, 0] for N=6.
    """
    grids = [base_arms[a] for a in self.farm.possible_agents]  # fixed order
    super_arms = np.array(list(itertools.product(*grids)), dtype=np.float32)

    if not reduced:
        return super_arms

    # Compute the "reduced" global cap as if each field had at most `per_field_limit`
    max_budgets = np.array(
        [self.farm.get_per_parcel_max_budget(a) - to_reduce for a in self.farm.possible_agents],
        dtype=np.float32,
    )
    # Effective per-field cap when pretending each field was limited to `per_field_limit`
    # effective_caps = np.minimum(max_budgets, float(per_field_limit))
    global_cap = float(max_budgets.sum()) / 10.0

    # Keep only combinations whose total reduction stays within this global cap
    total_reductions = super_arms.sum(axis=1)
    mask = total_reductions <= global_cap + 1e-6  # small tolerance for float errors
    return super_arms[mask]

def _make_topk_super_arms(
    base_arms: dict[str, np.ndarray],
    agents_order: Iterable[str],
    top_k: int = 3,
    as_array: bool = True,
    dtype=np.float32,
):
    """
    Keep only the top_k largest values from each agent's base grid, then take the product.

    If an agent has fewer than top_k values, we keep them all.

    Returns:
        - np.ndarray of shape (∏_i min(top_k, len(grid_i)), N) if as_array=True
        - an iterator of tuples otherwise
    """
    grids = []
    for a in agents_order:
        g = np.asarray(base_arms[a], dtype=dtype)
        # unique + sorted just in case (base_arms may already be sorted)
        g = np.unique(g)
        k = min(top_k, len(g))
        grids.append(g[-k:])  # take the largest k values

    prod_iter = itertools.product(*grids)
    return np.array(list(prod_iter), dtype=dtype) if as_array else prod_iter


def _make_rank_range_super_arms(
    base_arms: dict[str, np.ndarray],
    agents_order: Iterable[str],
    start_rank: int = 0,
    end_rank: int = 3,
    as_array: bool = True,
    dtype=np.float32,
):
    """
    Keep only values whose *descending* rank is in [start_rank, end_rank] for each agent,
    then take the product. Rank 0 = largest value.

    Example: start_rank=0, end_rank=3 → keep the top 4 values per agent.
    """
    if end_rank < start_rank:
        raise ValueError("end_rank must be >= start_rank")

    grids = []
    for a in agents_order:
        g = np.asarray(base_arms[a], dtype=dtype)
        g = np.unique(g)
        g.sort()
        # convert descending rank slice into ascending index slice:
        # last element (largest) has rank 0 → index -1
        # ranks [start, end] → indices [-end-1 : -start] (Python slice end-exclusive)
        # handle short arrays gracefully
        k_available = len(g)
        # effective start/end limited by available length
        eff_end = min(end_rank, k_available - 1)
        eff_start = min(start_rank, eff_end)
        # map ranks to ascending indices
        lo = max(0, k_available - (eff_end + 1))
        hi = k_available - eff_start
        grids.append(g[lo:hi])

    prod_iter = itertools.product(*grids)
    return np.array(list(prod_iter), dtype=dtype) if as_array else prod_iter


def extract_info(agent_id, counters, rewards, info, agent_idx):
    counters[agent_id]['Naction'] = info[0]['Naction']
    counters[agent_id]['Reward'] = rewards[0][agent_idx[agent_id]]
    counters[agent_id]['Nue'] = info[0]['Nue']
    counters[agent_id]['Nsurp'] = info[0]['Nsurp']
    counters[agent_id]['Yield'] = info[0]['Yield']
    return counters


def min_max_normalize(x, min_val=0, max_val=10_000_000) -> float:
    """Scale from [min_val, max_val] -> [0, 1]."""
    return (x - min_val) / (max_val - min_val)


def min_max_denormalize(x_norm, min_val=0, max_val=10_000_000) -> float:
    """Scale from [0, 1] -> [min_val, max_val]."""
    return x_norm * (max_val - min_val) + min_val

def last_before_nan(x, default=np.nan):
    a = np.asarray(x, dtype=float)
    valid = a[~np.isnan(a)]
    return valid[-1] if valid.size else x[-1]
