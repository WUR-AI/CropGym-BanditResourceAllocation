import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cropgymzoo.utils.agent_helpers import make_super_arms

from cropgymzoo.envs.worker_env import ParallelRLWorkers
from cropgymzoo.utils.defaults import get_default_years


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
        warm_up_eps: int = 100,
        reward_fn=None,
        random_allocation=True,
        global_budget=None,
        years: list = get_default_years(),
    ):
        super().__init__()

        self.global_budget = global_budget
        self.warm_up_eps = warm_up_eps

        self.years = years

        self._init_envs()

        assert n_fields >= 1, "n_fields must be positive"
        self.n_fields = n_fields
        self.delta = float(delta_kg)
        self.Q = int(total_kg // delta_kg)          # number of quanta
        self.super_arms = make_super_arms(n_fields, self.Q)
        self.super_arm_to_idx = {
            tuple(a): i for i, a in enumerate(self.super_arms)
        }

        self.action_space = spaces.Discrete(len(self.super_arms))
        self.observation_space = spaces.Dict(
            {agent: spaces.Box(-np.inf, np.inf, shape = (1,), dtype = np.float32)
             for agent in self.cropgyms.agents}
        )
        self.reward_fn = reward_fn or self._dummy_yield

        self.random_allocation = random_allocation

        self._construct_info()

    '''
    Gymnasium functions
    '''
    def reset(self, *, seed=None, options=None):
        assert 'global_budget' in options, "Please reset env with global_budget key!"

        self.global_budget = options.get('global_budget')

        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        if self.random_allocation:
            allocations = self._allocate_random_budgets()
            reward = 0
            return np.zeros(1, dtype=np.float32), reward, True, False, self.info
        else:
            assert self.action_space.contains(action), "invalid action"
            self.info['alloc_quanta'] = self.super_arms[action]
            reward = float(self.reward_fn(self.info['alloc_quanta'] * self.delta))
            return np.zeros(1, dtype=np.float32), reward, True, False, self.info

    def _construct_info(self):
        self.info = {}

    def _init_envs(self):
        self.cropgyms = ParallelRLWorkers(
            global_budget=self.global_budget,
            warm_up=self.warm_up_eps,
            years=self.years,
        )

    @staticmethod
    def _get_context_keys():
        return [
            "InitialNO3",
            "InitialNH4",
            "CropCode",
            "HistoricalCropPrice",
            "HistoricalFertilizerPrice",
            "HistoricalProfit",
            "HistoricalYield",
            "HistoricalBudget",
            "HistoricalNUE",
            "HistoricalNsurplus",
            "HistoricalPrecipitation",
            "HistoricalTemperature",
            "HistoricalIrrad",
        ]

    # ────────────────────────────────────────────────────────────────
    # toy concave reward that prefers balanced splits
    # ────────────────────────────────────────────────────────────────
    def _dummy_yield(self, kg_vec):
        return kg_vec.mean() - 0.1 * np.var(kg_vec)

    def _allocate_random_budgets(self) -> dict[str, float]:
        """
        Return a dict {agent_id: kg_budget} that d sums to self.global_budget
        and d never exceeds each field’s legal ceiling.
        Requires:
            • self.fields        : dict[str, ParcelEnv]
            • self.global_budget : float   (kg for this season)
            • self.crop_caps     : dict[str, float]  # e.g. {'wheat':240,…}
        """
        rng = np.random.default_rng()  # or use self.np_random
        agents = list(self.fields.keys())
        n = len(agents)
        q = 10 # kg/ha

        # ----------------------------------------------------------------
        # 1) find per-field ceiling  m_j  from either the parcel or a lookup
        # ----------------------------------------------------------------
        cap_q = np.empty(n, dtype=int)
        for k, ag in enumerate(agents):
            env = self.fields[ag]
            # priority 1: an attribute on the parcel env
            # TODO check this logic
            if hasattr(env, "max_allowed_kg"):
                caps = env.max_allowed_kg
            else:  # fallback from crop type
                caps = self._get_crop_caps[env.unwrapped.crop]  # e.g. 240, 150 …
            cap_q[k] = int(np.floor(caps / q))

        # ---- 2) global budget in quanta -------------------------------
        Q_total = int(np.round(self.global_budget / q))
        if Q_total > cap_q.sum():
            raise ValueError("Budget exceeds joint crop ceilings")

        alloc_q = np.zeros(n, dtype=int)
        remaining_q = Q_total
        remaining_idx = np.arange(n)

        # ---- 3) iterative multinomial with clipping -------------------
        while remaining_q > 0 and remaining_idx.size:
            probs = rng.dirichlet(np.ones(remaining_idx.size))
            # sample how many quanta each remaining field *would* get
            proposal_q = rng.multinomial(remaining_q, probs)
            room_q = cap_q[remaining_idx] - alloc_q[remaining_idx]
            applied_q = np.minimum(proposal_q, room_q)  # clip
            alloc_q[remaining_idx] += applied_q
            remaining_q -= applied_q.sum()
            # keep only fields that can still accept quanta
            remaining_idx = remaining_idx[(room_q - applied_q) > 0]

        if remaining_q > 0:
            raise RuntimeError("Could not allocate all quanta; all fields full")

        return {ag: float(alloc_q[k] * q) for k, ag in enumerate(agents)}



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
