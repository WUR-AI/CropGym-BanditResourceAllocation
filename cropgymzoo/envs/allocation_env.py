import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cropgymzoo.utils.agent_helpers import make_super_arms



# ---------------------------------------------------------------------
# Gymnasium env that works for any n_fields
# ---------------------------------------------------------------------
class AllocationBandit(gym.Env):
    """
    A one-Step combinatorial multi-armed bandit environment for resource allocation.

    One-step combinatorial bandit: split X = Q·δ kg over `n_fields` parcels.
    X :: the total available budget; could be randomized
    Q :: The quanta, or levels of actions availble for each field.
    δ :: The delta, or granularity of actions.
    Action  = index into self.super_arms (pre-enumerated allocations).
    Reward  = user-supplied function f(kg_vector)  (default is toy example).
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_fields: int = 6,
        delta_kg: float = 10.0,
        total_kg: float = 180.0,
        field_config: dict = None,
        reward_fn=None,
    ):
        super().__init__()
        assert n_fields >= 1, "n_fields must be positive"
        self.n_fields = n_fields
        self.delta = float(delta_kg)
        self.Q = int(total_kg // delta_kg)          # number of quanta
        self.super_arms = make_super_arms(n_fields, self.Q)
        self.super_arm_to_idx = {
            tuple(a): i for i, a in enumerate(self.super_arms)
        }

        self.action_space = spaces.Discrete(len(self.super_arms))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.reward_fn = reward_fn or self._dummy_yield

        self._construct_info()

    # ────────────────────────────────────────────────────────────────
    # gym API
    # ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action), "invalid action"
        self.info['alloc_quanta'] = self.super_arms[action]
        reward = float(self.reward_fn(self.info['alloc_quanta'] * self.delta))
        return np.zeros(1, dtype=np.float32), reward, True, False, self.info

    def _construct_info(self):
        self.info = {}

    # ────────────────────────────────────────────────────────────────
    # toy concave reward that prefers balanced splits
    # ────────────────────────────────────────────────────────────────
    def _dummy_yield(self, kg_vec):
        return kg_vec.mean() - 0.1 * np.var(kg_vec)


# ---------------------------------------------------------------------
# Minimal CUCB-style loop (works for any n_fields)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    env = AllocationBandit(n_fields=8, total_kg=200, delta_kg=10)
    Q, n = env.Q, env.n_fields

    counts  = np.zeros((n, Q + 1))
    rewards = np.zeros((n, Q + 1))

    def greedy_fill_ucb(ucb, Q):
        """Allocate one quantum at a time to the field that maximises margin."""
        alloc = np.zeros(n, dtype=int)
        for _ in range(Q):
            best_j = np.argmax([
                ucb[j, alloc[j] + 1] if alloc[j] < Q else -np.inf
                for j in range(n)
            ])
            alloc[best_j] += 1
        return alloc

    for season in range(1, 501):
        ucb = rewards / np.maximum(1, counts) + np.sqrt(
            2 * np.log(season) / np.maximum(1, counts)
        )
        alloc_vec = greedy_fill_ucb(ucb, Q)
        action = env.super_arm_to_idx[tuple(alloc_vec)]
        _, R, *_ = env.step(action)

        for j, l in enumerate(alloc_vec):
            counts[j, l]  += 1
            rewards[j, l] += R

        if season % 100 == 0:
            print(f"S{season:3d}  reward={R:6.2f}  alloc(kg)={alloc_vec*env.delta}")
