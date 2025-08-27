import argparse
import itertools

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cropgymzoo.utils.agent_helpers import _make_super_arms, _make_base_arms

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.defaults import get_default_years
from cropgymzoo.train_tianshou import load_model, make_ppo_policy
from cropgymzoo.eval_tianshou import MultiRLAgent


# ---------------------------------------------------------------------
# Gymnasium env that works for any n_fields
# ---------------------------------------------------------------------
class AllocationBandit(gym.Env):
    """
    A one-Step combinatorial multi-armed bandit environment for resource allocation.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        delta_kg: float = 10.0,
        warm_up_eps: int = 10,
        reward_fn=None,
        years: list = get_default_years(),
        seed: int = 107,
        action_type: str = 'multi_discrete',
        args: argparse.Namespace = None,
    ):
        super().__init__()

        self.warm_up_eps = warm_up_eps

        assert action_type in ['discrete', 'multi_discrete']
        self.action_type = action_type

        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)

        self.years = years
        self.year = None

        # The MARL env
        self._init_envs(args)

        # set up per parcel budgets
        self._init_meta_info()

        # init spaces
        self.bins = float(delta_kg)
        self._init_spaces()

        self._construct_info()

    '''
    Gymnasium functions
    '''
    def reset(self, *, seed=None, options=None):

        options = self._get_default_reset_options() if options is None else options

        assert 'year' in options, "If testing, make sure to pass 'year' in the options dictionary!"

        options['seed'] = seed

        self.farm.reset(seed=seed, options=options)

        return self._get_context(), self._construct_info(options)

    def step(self, action):
        # check if action is valid
        assert self.action_space.contains(action), "invalid action"

        # save action this episode
        self.infos['AllocationAction'] = self.super_arms[action]

        # TODO make method in MultiFieldEnv
        self.farm.allocate_bandit_budgets(self.infos['AllocationAction'])

        # runs one episode of the MARL agent
        infos_agents = self.env_agent.run([self.infos['year']])

        self.infos['AgentInfos'] = infos_agents
        reward = self._get_reward()

        return np.zeros(1, dtype=np.float32), reward, True, False, self.infos


    def _get_reward(self):
        # convert budget left as profit
        budget_lefts = np.array([self.farm.infos['AgentInfos'][agent]['BudgetLeft'][-1] for agent in self.parcel_meta_infos.keys()])
        fertilizer_prices = np.array([self.farm.infos['AgentInfos'][agent]['FertilizerPrice'][-1] for agent in self.parcel_meta_infos.keys()])
        budget_left_profit = budget_lefts @ fertilizer_prices

        # add with actual profit
        profit = np.array([np.sum(self.farm.infos['AgentInfos'][agent]['Profit']) for agent in self.parcel_meta_infos.keys()])
        reward = profit + budget_left_profit

        return reward


    @staticmethod
    def _get_context_keys():
        return [
            "InitialN",
            "CropPrice",
            "CropCode",
            "FertilizerPrice",
            "HistoricalCropPrices",
            "HistoricalFertilizerPrices",
            "HistoricalProfit",
            "HistoricalYield",
            "HistoricalFertilizerUse",
            "HistoricalBudget",
            "HistoricalBudgetLeft",
            "HistoricalNUE",
            "HistoricalNsurplus",
            "HistoricalPrecipitation",
            "HistoricalTemperatureMin",
            "HistoricalTemperatureMax",
            "HistoricalIrradiation",
        ]

    '''
    Helper functions
    '''

    def _construct_info(self, options=None):
        if options is not None:
            self.infos = {'options': options, 'seed': options.get('seed', 0)}
        else:
            self.infos = {}

    def _get_default_reset_options(self):
        return {'year': self.rng.choice(self.years)}

    '''
    Context helper functions
    '''

    def _get_context(self):
        return {
            "InitialN": list(self.farm.get_initial_n().values()),
            "CropPrice": list(self.farm.get_per_field_crop_price().values()),
            "CropCode": list(self.farm.get_per_field_crop_code().values()),
            "FertilizerPrice": list(self.farm.get_per_field_fertilizer_price().values()),  # sample from year
            "HistoricalCropPrices": self._get_historical_end_season_features('CropPrice'),
            "HistoricalFertilizerPrices": self._get_historical_end_season_features('FertilizerPrice'),
            "HistoricalProfit": self._get_historical_end_season_features('Profit'),
            "HistoricalYield": self._get_historical_end_season_features('Yield'),
            "HistoricalFertilizerUse": self._get_historical_end_season_features('Naction'),
            "HistoricalBudget": self._get_historical_end_season_features('BudgetTotal'),
            "HistoricalBudgetLeft": self._get_historical_end_season_features('BudgetLeft'),
            "HistoricalNUE": self._get_historical_end_season_features('Nue'),
            "HistoricalNsurplus": self._get_historical_end_season_features('Nsurp'),
            "HistoricalPrecipitation": self._get_historical_weather_features('RAIN'),
            "HistoricalTemperatureMin": self._get_historical_weather_features('TMIN'),
            "HistoricalTemperatureMax": self._get_historical_weather_features('TMAX'),
            "HistoricalIrradiation": self._get_historical_weather_features('IRRAD'),
        }

    def _get_historical_end_season_features(self, feature):
        # returns vector length of n_fields based on average end season feature
        return [
            np.mean([
                self.farm.warm_up_infos[i][agent][feature][-1]
                for i in self.farm.warm_up_infos
            ])
            for agent in self.parcel_meta_infos.keys()
        ]

    def _get_historical_weather_features(self, feature):
        # returns vector length of n_fields with mean of weather
        return [
            np.mean([
                np.mean(self.farm.warm_up_infos[i][agent][feature])
                for i in self.farm.warm_up_infos
            ])
            for agent in self.parcel_meta_infos.keys()
        ]

    '''
    Init helpers
    '''

    def _init_envs(self, args):
        self.farm = MultiFieldEnv(
            warm_up=self.warm_up_eps,
            years=self.years,
        )

        if args is not None and hasattr(args, 'use_model'):
            saved_model = load_model(args)
            self.env_agent = MultiRLAgent(
                env = self.farm,
                saved_model=saved_model,
                render=False,
            )
            self.farm = self.env_agent.env

    def _init_spaces(self):
        # Set up action space based on farm
        self.base_arms = _make_base_arms(self)
        self.super_arms = _make_super_arms(self, self.base_arms)
        self.super_arm_to_idx = {
            tuple(a): i for i, a in enumerate(self.super_arms)
        }

        # Action space
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(len(self.super_arms))
        if self.action_type == 'multi_discrete':
            self.action_space = spaces.MultiDiscrete([len(self.base_arms[a]) for a in self.farm.possible_agents])

        # Observation space
        self.observation_space = spaces.Dict(
            {
                feature: spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.n_fields,),
                    dtype=np.float32
                )
                for feature in self._get_context_keys()
            }
        )

    def _init_meta_info(self):
        self.n_fields = len(self.farm.possible_agents)
        self.global_budget = self.farm.global_budget
        self.parcel_meta_infos = {
            agent: {'max_budget': self.farm.fields[agent].unwrapped.max_budget_n,
                    'crop': self.farm.fields[agent].unwrapped.crop,
                    'crop_code': self.farm.fields[agent].unwrapped.CROP_CODE_MAP[
                        self.farm.fields[agent].unwrapped.crop
                    ],
                    'soil_type': self.farm.fields[agent].unwrapped.soil_type,
                    'area': self.farm.fields[agent].unwrapped.area, }
            for agent in self.farm.possible_agents
        }


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
            env = self.farm.fields[ag]
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
            print(f"S{season:3d}  reward={R:6.2f}  alloc(kg)={alloc_vec*env.bins}")
