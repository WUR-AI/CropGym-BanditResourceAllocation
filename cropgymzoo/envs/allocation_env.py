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
        action_type: str = 'continuous',
        args: argparse.Namespace = None,
    ):
        super().__init__()

        self.warm_up_eps = warm_up_eps

        assert action_type in ['discrete', 'multi_discrete', 'continuous']
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
        self.infos['AllocationAction'] = action

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
        # dot product below
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

    def _get_historical_context_keys(self):
        return self._get_context_keys()[4:]

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

    def _get_historical_end_season_features(self, feature: str):
        """Return [mean_over_iters( last value of feature for this agent ), for each agent]."""
        out = []
        for agent in self.parcel_meta_infos.keys():
            vals = []
            for iter_info in self.farm.warm_up_infos:  # iter_info: dict per iteration
                agent_info = iter_info.get(agent)
                seq = agent_info.get(feature)
                vals.append(seq[-1])
            out.append(float(np.mean(vals)))
        return out

    def _get_historical_weather_features(self, feature: str):
        """Return [mean_over_iters( mean of the feature sequence for this agent ), per agent]."""
        out = []
        for agent in self.parcel_meta_infos.keys():
            vals = []
            for iter_info in self.farm.warm_up_infos:
                agent_info = iter_info.get(agent)
                seq = agent_info.get(feature)
                vals.append(np.mean(seq))
            out.append(float(np.mean(vals)))
        return out

    def add_stats_to_context(self, info):
        self.farm.warm_up_infos.append(info)

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
        # discrete and multi_discrete is not implemented properly yet.
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(len(self.super_arms))
        if self.action_type == 'multi_discrete':
            self.action_space = spaces.MultiDiscrete(
                [
                    len(self.base_arms[a])
                    for a in self.farm.possible_agents
                ]
            )
        if self.action_type == 'continuous':
            self.action_space = spaces.Box(low=0, high=1, shape=(self.n_fields,), dtype=np.float32)

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
