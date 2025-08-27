import itertools
import numpy as np


def _make_base_arms(self, cap: float = 0.5) -> dict[str, np.ndarray]:
    """
    Per-farm allowed reductions in kg/ha: [0, delta, 2*delta, ..., floor(cap*budget/delta)*delta].
    """
    max_budgets = {a: self.farm.get_per_parcel_max_budget(a) for a in self.farm.possible_agents}
    base = {}
    for agent, max_budget in max_budgets.items():
        half_budget = max(0.0, float(max_budget) * cap)
        q = int(np.floor(half_budget / float(self.bins)))
        base[agent] = (np.arange(q + 1, dtype=np.float32) * float(self.bins))
    return base


def _make_super_arms(self, base_arms: dict[str, np.ndarray]) -> np.ndarray:
    """
    Cartesian product over the per-farm base arms, returned as an array of shape (num_arms, N).
    Each row is an N-length reduction vector, e.g., [10, 40, 20, 30, 40, 0] for N=6.
    """
    grids = [base_arms[a] for a in self.farm.possible_agents]  # fixed order → stable indexing
    super_arms = np.array(list(itertools.product(*grids)), dtype=np.float32)
    return super_arms


def extract_info(agent_id, counters, rewards, info, agent_idx):
    counters[agent_id]['Naction'] = info[0]['Naction']
    counters[agent_id]['Reward'] = rewards[0][agent_idx[agent_id]]
    counters[agent_id]['Nue'] = info[0]['Nue']
    counters[agent_id]['Nsurp'] = info[0]['Nsurp']
    counters[agent_id]['Yield'] = info[0]['Yield']
    return counters