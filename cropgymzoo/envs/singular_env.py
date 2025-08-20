import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import math
from copy import deepcopy

import yaml

import functools
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle

import numpy as np
import datetime

from pcse.input.sitedataproviders import WOFOST81SiteDataProvider_SNOMIN
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider

from cropgymzoo import _WOFOST_CONFIG, _AGRO_CALENDAR_CONFIG, _CROPS_PATH, _SOIL_PATH, _SITE_PATH, _SOILGRIDS_PATH, _CROPS_CONFIG

import cropgymzoo.utils.process_pcse_output as process_pcse
from cropgymzoo.utils.rewards import (Rewards, ActionsContainer, reward_functions_with_baseline,
                                      reward_functions_end, calculate_nue)
from cropgymzoo.utils.nitrogen_helpers import (get_surplus_n, get_nh4_deposition_pcse,
                                               get_no3_deposition_pcse, convert_year_to_n_concentration, m2_to_ha,
                                               is_leap, co2_levels)
import cropgymzoo.envs.pcse_env as pcse_env
from cropgymzoo.utils.defaults import (get_wofost_default_crop_features,
                                       get_default_weather_features,
                                       get_default_action_features,
                                       get_default_misc_features)

import torch as th
import torch.nn as nn

def make_parcel_env(*, training: bool = False, **kwargs):
    env = ParcelEnv(training=training, **kwargs)          # build the base env
    # if training:
    #     env = PCSERandomizer(env) # add arguments if your wrapper needs them
    return env

class ParcelEnv(pcse_env.PCSEEnv, EzPickle):
    """
    This is a class that inherits PCSE.
    It will be instantiated multiple times for the MARL environment.
    """
    '''
    Some class variables, so they do not get copied for each instance of this class.
    '''
    """
    WARNING! Learned agents rely on this crop code mapping; don't inadvertently change it!
    """

    CROP_CODE_MAP = {
        'winterwheat': 1,
        'sugarbeet': 2,
        'potato': 3,
        'soybean': 4,
        'barley': 5,
        'seed_onion': 6,
        'sunflower': 7,
        'fababean': 8,
        'chickpea': 9,
        'sweetpotato': 10,
        'cowpea': 11,
        'rapeseed': 12,
        'rice': 13,
        'groundnut': 14,
        'cassava': 15
    }

    '''
    Crop parameters to add noise to
    '''

    CROP_PARAMS = [
        "TBASE",  # lower threshold temperature for ageing of leaves
        "SPAN",  # life span of leaves growing at 35 Celsius
        "TDWI",  # initial total crop dry weight
        "CVL",  # efficiency of conversion into leaves
        "CVO",  # efficiency of conversion into storage organs
        "CVR",  # efficiency of conversion into roots
        "CVS",  # efficiency of conversion into stems
        "PERDL",  # maximum relative death rate of leaves due to water stress
        "RGRLAI_MIN"  # maximum relative increase in LAI
        "RNUPTAKEMAX",  # Maximum rate of daily nitrogen uptake
        "DVS_N_TRANSL"  # development stage above which N translocation to storage organs does occur
    ]

    CROP_SOIL_MAX = {
        "winterwheat": {'clay': 245, 'sand': 160, 'silt': 190, 'peat': 160},
        "sugarbeet": {'clay': 150, 'sand': 145, 'silt': 116, 'peat': 145},
        "potato": {'clay': 275, 'sand': 260, 'silt': 204, 'peat': 270},
        "barley": {'clay': 80, 'sand': 80, 'silt': 80, 'peat': 80},
        "seed_onion": {'clay': 170, 'sand': 120, 'silt': 120, 'peat': 120},
        'rapeseed': {'clay': 205, 'sand': 190, 'silt': 152, 'peat': 195},
        'sunflower': {'clay': 150, 'sand': 150, 'silt': 150, 'peat': 150},
    }

    '''
    Initialize Env for each RL agent
    '''
    def __init__(self,
                 crop_features: list = get_wofost_default_crop_features(),
                 weather_features: list = get_default_weather_features(),
                 action_features: list = get_default_action_features(),
                 misc_features: list = get_default_misc_features(),
                 location: list | tuple = None,
                 year: int = None,
                 year_list: list = None,
                 timestep: int = 7,
                 reward: str = 'PNB',
                 action_multiplier: float = 1,
                 action_space: gym.spaces = gym.spaces.Discrete(9),
                 costs_nitrogen: int = 0,
                 crop: str = 'winterwheat',
                 name: str = None,
                 area: float = 12,
                 model_config: str = _WOFOST_CONFIG,
                 agro_config: str = _AGRO_CALENDAR_CONFIG,
                 type: str = 'clay',
                 site_path: str = _SITE_PATH,
                 soil_path: str = _SOIL_PATH,
                 seed: int = 107,
                 training: bool = False,
                 original: bool = True,
                 flatten_obs: bool = True,
                 **kwargs,
    ):
        EzPickle.__init__(
            self,
            crop_features=crop_features,
            weather_features=weather_features,
            action_features=action_features,
            misc_features=misc_features,
            location = location,
            year = year,
            year_list = year_list,
            timestep = timestep,
            reward = reward,
            action_multiplier = action_multiplier,
            action_space = action_space,
            costs_nitrogen = costs_nitrogen,
            crop=crop,
            name=name,
            area=area,
            model_config=model_config,
            agro_config=agro_config,
            type=type,
            site_path=site_path,
            soil_path=soil_path,
            seed=seed,
            training=training,
            original=original,
            flatten_obs=flatten_obs,
            **kwargs,
        )
        # instance metadata
        self.original = original
        self.training = training
        self.flatten_obs = flatten_obs
        self.name = name
        self.soil_type = type

        # pcse variables
        self.crop = crop
        self.crop_features = crop_features
        self.weather_features = weather_features
        self.action_features = action_features
        self.misc_features = misc_features
        self.year = year
        self.location = location
        self.year_list = year_list

        # field specific stuff
        self.max_budget_n = self.CROP_SOIL_MAX[self.crop][self.soil_type]
        self.budget_n = self.max_budget_n
        self.budget_left = self.budget_n
        self.area = area

        # back to PCSE stuff

        self.agro_config = agro_config

        crop_parameters, site_parameters, soil_parameters = self._init_configs()

        with open(_CROPS_CONFIG, 'r') as f:
            crop_info = yaml.safe_load(f)

        super(ParcelEnv, self).__init__(
            model_config=model_config,
            agro_config=agro_config,
            crop_parameters=crop_parameters,
            site_parameters=site_parameters,
            soil_parameters=soil_parameters,
            locations=location,
            crop=crop,
            crop_info=crop_info,
            training=training,
        )

        # safeguard for randomisation
        self.original_agmt = getattr(self, "agmt")

        # possibly deprecated
        # self.costs_nitrogen = costs_nitrogen
        self.action_multiplier = action_multiplier

        # env stuff
        self.action_space = action_space
        self._timestep = timestep
        self.reward_function = reward

        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)

        # Training stuff
        self.random_weather = False
        self.random_init = False
        self.random_params = False
        if self.training:
            self.random_weather = True
            self.random_init = True
            self.random_params = True

        # prices of crops and fertilizers
        self._init_prices()

        # initialize variables pertaining to RL agent actions
        self._init_action_variables()

        # initialize reward function
        self._init_reward_function(costs_nitrogen, kwargs)

        # initialize observation key list
        self._init_obs_keys()

        # initialize infos
        self._init_infos()

        # initialize Zero nitrogen simulations
        self._init_zero_env(**kwargs)

        # reset the env at start
        # Need to change?
        super().reset(seed=seed, options={'year': self.year})

        # self._env_baseline.get_key(self)


    '''
    Gymnasium functions
    '''

    def step(self, action):
        """
        Computes customized reward and populates info
        """
        self.n_steps += 1

        # advance one step of the PCSEEngine wrapper and apply action(s)
        obs_pcse, _, terminated, truncated, _ = super().step(action)

        # update actions
        self._update_action_variables(action)

        # transform and flatten observations
        obs = self._observation(obs_pcse, terminated)

        # get pcse output
        pcse_output = self.model.get_output()

        # process output to get the reward and growth of the crop
        reward, growth = self._process_output(action, pcse_output, terminated)

        # append new information to the infos dict
        self._populate_infos(pcse_output, action, reward, terminated)

        return obs, reward, terminated, truncated, self.infos

    def reset(self, seed=None, options=None, **kwargs):
        """
        Resets parcel episode. The key `year` must be in the ::options dictionary.
        Here, growing season budget is also determined.
        """

        assert 'year' in options, "Please reset environment with a year"

        # Only for invalid action masking
        # self.reset_non_zero_action_count()

        # return the original agro management class in case of shifts during randos
        self.agmt = deepcopy(self.original_agmt)

        # Can change training mode on reset
        self.check_if_training()

        # reset reward runners
        self.reward_container.reset()
        self.rewards_obj.reset()

        # reset various variables
        self._reset_action_variables()
        self._reset_prices()

        # overwrite for new eps
        self.overwrite_year(year=options['year'])

        # use options. Shift soil and randomise N conditions
        site_params = self._special_init_conditions()
        options['site_params'] = site_params

        # randomise domain
        options = self._randomise_domain(options)

        # reset PCSE
        if self.reward_function in reward_functions_with_baseline() and self.original is True:
            self.baseline_env.reset(seed=seed, options=options)
        obs = super().reset(seed=seed, options=options)

        # get infos
        self._init_infos()
        self._populate_infos(self.model.get_output(), 0, 0, False)

        return self._observation(obs), self.infos

    def action_mask(self) -> list | np.ndarray:
        """
        Returns a list of valid actions based on the budget left!
        """
        max_units = max(int(self.budget_left // 10), 0)
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[: max_units + 1] = True  # valid actions are 0 … max_units
        return np.array(mask, dtype=np.int8)

    def sample_masked_action(self):
        mask = self.action_mask()
        valid_actions = np.where(mask)[0]
        action = np.random.choice(valid_actions)
        return action



    '''
    Callable class methods
    '''

    def overwrite_year(self, year):
        self.year = year
        self.agro_management = self.agmt.update_attributes(crop_start_date=lambda d: d.replace(year=year),
                                                           campaign_date=lambda d: self._safe_replace_year(d, year),)
        if self.reward_function in reward_functions_with_baseline() and self.original is True:
            self.baseline_env.agro_management = self.agmt.update_attributes(crop_start_date=lambda d: d.replace(year=year),
                                                                            campaign_date=lambda d: self._safe_replace_year(d, year),)

    def render(self, mode="human"):
        pass

    def get_latest_info(self, feature):
        return self.infos[feature][-1]

    def set_budget(self, budget):
        self.budget_n = budget
        self.budget_left = self.budget_n
        self._update_budget_left()

    def get_max_allowed(self):
        # TODO fill in
        ...

    #For AECEnv in the MultiEnv setting
    def observe(self) -> dict | np.ndarray:
        output_pcse = self.model.get_output()
        obs = super()._get_observation(output_pcse)
        return self._observation(obs)

    '''
    Helper functions for various things
    '''

    def check_if_training(self):
        if self.training:
            self.random_weather = True
            self.random_init = True
            self.random_params = True
        elif not self.training:
            self.random_weather = False
            self.random_init = False
            self.random_params = False

    @staticmethod
    def _safe_replace_year(d, year):
        try:
            return d.replace(year=year)
        except ValueError:
            # fallback for Feb 29 to Feb 28
            return d.replace(year=year, day=28)

    def _reset_prices(self):
        self.fertilizer_price = self._get_fertilizer_price()

        self.crop_price = self._get_crop_price()

        self._update_budget_left()

    def _update_budget_left(self):
        self.reward_class.budget_left = self.budget_left

    def _update_action_variables(self, action):
        if isinstance(action, np.ndarray):
            action = action[0] if action.shape else action

        if action > 0:
            self.n_action += action * 10
            self.non_zero_action_count += action
            self.steps_since_last_action = 0

            # budget count
            self.budget_left -= action * 10         # Convert to kg/ha
        else:
            self.steps_since_last_action += 1

    def _reset_action_variables(self):
        self.n_steps = 0
        self.n_action = 0
        self.non_zero_action_count = 0
        self.steps_since_last_action = 0
        self.budget_left = self.budget_n


    def _process_output(self, action, output, terminated):
        if isinstance(action, np.ndarray):
            action = action.item()

        # TIMES 10 HERE
        amount = action * 10  #kg/ha

        output_baseline = []
        if self.reward_function in reward_functions_with_baseline():

            zero_nitrogen_results = self.zero_nitrogen_env_storage.get_episode_output(self.baseline_env)

            # convert zero_nitrogen_results to pcse_output
            var_name = process_pcse.get_name_storage_organ(zero_nitrogen_results.keys())
            for (k, v) in zero_nitrogen_results[var_name].items():
                if k <= output[-1]['day']:
                    filtered_dict = {'day': k, var_name: v}
                    output_baseline.append(filtered_dict)
            assert len(output_baseline) != 0, f'OUTPUT BASELINE EMPTY'

        self.rewards_obj.update_profit(output, amount, year=self.date.year)

        prices = {
            'price_fertilizer': self.fertilizer_price,
            'price_crop': self.crop_price,
            'budget_left': self.budget_left,
        }
        reward, growth = self.reward_class.return_reward(
            output,
            amount,
            output_baseline=output_baseline,
            obj=self.reward_container,
            **prices
            if self.reward_function in ['PNY', 'PNB']
            else {}
        )
        del prices

        reward += self._terminated_reward_signal(output, reward, terminated)

        return reward, growth

    def _terminated_reward_signal(self, output, reward, terminated):
        if terminated and self.reward_function in reward_functions_end():
            reward = self.reward_container.dump_cumulative_positive_reward - abs(reward)
            return reward

        elif terminated and self.reward_function == 'HAR':
            reward = self.yield_modifier * self.reward_container.dump_cumulative_positive_reward - abs(reward)
            return reward

        elif terminated and self.reward_function in ['NUE', 'DNE']:
            reward = (self.reward_container.calculate_reward_nue(
                n_fertilized=self.reward_container.get_total_fertilization,
                n_output=process_pcse.get_n_storage_organ(output),
                no3_depo=get_no3_deposition_pcse(output),
                nh4_depo=get_nh4_deposition_pcse(output),)
            )
            return reward

        elif terminated and self.reward_function == 'PNY':
            reward = (self.reward_class.return_final_reward(
                obj=self.reward_container,
                n_fertilized=self.reward_container.get_total_fertilization,
                n_output=process_pcse.get_n_storage_organ(output),
                no3_depo=get_no3_deposition_pcse(output),
                nh4_depo=get_nh4_deposition_pcse(output),
                crop_name=self.crop)
            )
            return reward

        elif terminated and self.reward_function == 'PNB':
            reward = (
                self.reward_class.return_final_reward(
                    obj=self.reward_container,
                    n_fertilized=self.reward_container.get_total_fertilization,
                    n_output=process_pcse.get_n_storage_organ(output),
                    no3_depo=get_no3_deposition_pcse(output),
                    nh4_depo=get_nh4_deposition_pcse(output),
                    budget_left=self.budget_left,
                    crop_name=self.crop,
                )
            )
            return reward

        return 0

    def _overwrite_initial_conditions(self):
        # N initial conditions
        list_nh4i, list_no3i = self._generate_realistic_n()
        self.eval_nh4i = list_nh4i
        self.eval_no3i = list_no3i

        site_parameters = {'NH4I': list_nh4i, 'NO3I': list_no3i, }
        return site_parameters

    def _overwrite_nitrogen_rain_concentration(self):
        # N concentration in rain for deposition
        nh4concr, no3concr = convert_year_to_n_concentration(self.date.year,
                                                             loc=self.loc,
                                                             wdp=self.weather_data_provider,)

        site_parameters = {'NH4ConcR': nh4concr, 'NO3ConcR': no3concr, }
        return site_parameters

    @staticmethod
    def _crop_model_sum_last(pcse, var, normalise=1.0):
        return np.sum(pcse[var][-1]) / normalise

    def _get_key_transformations(self, crop_model):
        return {
            "NH4": lambda: self._crop_model_sum_last(crop_model, "NH4", m2_to_ha),
            "NO3": lambda: self._crop_model_sum_last(crop_model, "NO3", m2_to_ha),
            "SM": lambda: self._crop_model_sum_last(crop_model, "SM"),
            "WC": lambda: self._crop_model_sum_last(crop_model, "WC"),
            "RNO3DEPOSTT": lambda: self._crop_model_sum_last(crop_model, "RNO3DEPOSTT", m2_to_ha),
            "RNH4DEPOSTT": lambda: self._crop_model_sum_last(crop_model, "RNH4DEPOSTT", m2_to_ha),
        }

    @staticmethod
    def _transform_crop_feature(pcse_output, feature):
        if feature in ["NH4", "NO3", "RNO3DEPOSTT", "RNH4DEPOSTT"]:
            return np.sum(pcse_output[feature]) / m2_to_ha
        if feature in ["SM", "WC"]:
            return np.sum(pcse_output[feature])
        return pcse_output[feature]

    def _observation(self, observation, terminated = False):
        """
        Flatten the structured `observation` dict into a 1-D numpy array that
        matches `self.observation_space.shape`.
        """

        if isinstance(observation, tuple):
            observation = observation[0]

        # flattened to vector
        crop_model = observation["crop_model"]
        act = self._action_features_mapper()
        weather = observation["weather"]
        misc = self._misc_features_mapper(terminated)

        # perform some transformations
        crop_values = [
            self._get_key_transformations(crop_model).get(f, lambda: crop_model[f][-1])()  # fall back to plain last value
            for f in self.crop_features
        ]

        action_values = [act[a] for a in self.action_features]

        misc_values = [misc[m] for m in self.misc_features]

        weather_matrix = np.vstack(
            [
                weather[f][:self.timestep]
                if f not in "IRRAD"
                else [w / 1_000_000 for w in weather[f][:self.timestep]]
                for f in self.weather_features
            ]
        ).T
        weather_values = weather_matrix.ravel()

        if self.flatten_obs:
            return np.array(crop_values + action_values + misc_values + list(weather_values), dtype=np.float32)

        obs = {
            **{k: float(v) for k, v in zip(self.crop_features, crop_values)},
            **{k: float(v) for k, v in zip(self.action_features, action_values)},
            **{k: float(v) for k, v in zip(self.misc_features, misc_values)},
            **{
                f"{var_name}_{t}": float(weather_matrix[t, var_idx])
                for var_idx, var_name in enumerate(self.weather_features)
                for t in range(self.timestep)
            }
        }

        return obs

    @functools.lru_cache(maxsize=None)
    def _get_observation_space(self):
        nvars = self._get_obs_len()
        if self.flatten_obs:
            return gym.spaces.Box(-np.inf, np.inf, shape=(nvars,), dtype=np.float32)
        else:
            return gym.spaces.Dict({name: gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
                                   for name in self._get_obs_keys()})

    @functools.lru_cache(maxsize=None)
    def _get_obs_len(self):
        nvars = (len(self.crop_features) + len(self.action_features) +
                 len(self.misc_features) + len(self.weather_features) * self.timestep)
        return nvars

    @functools.lru_cache(maxsize=None)
    def _get_obs_keys(self):
        return (
            self.crop_features + self.action_features + self.misc_features +
            [f"{self.weather_features[i]}_{t}" for t in range(self.timestep) for i, _ in enumerate(self.weather_features)]
        )

    def _apply_action(self, action):
        action = action * 10  # kg N / ha
        return action

    def _get_reward(self):
        # Reward gets overwritten in step()
        return 0.0

    def _init_infos(self):
        self.infos = {
            "Date": [], "SinDay": [], "CosDay": [],
            **{name: [] for name in self.crop_features},
            **{name: [] for name in self.weather_features},
            **{name: [] for name in self.action_features},
            **{name: [] for name in self.misc_features},
            'Reward': [], 'Action': [], 'Yield': [], 'NAVAIL': [],
            'BudgetTotal': [], 'BudgetLeft': [], 'CropName': [],
            'Nue': [], 'Nsurp': [], 'Profit': [], "CO2": [],
            'Alive': [], 'ActionMask': []
        }

    def _init_random_init_conditions_params(self):
        self.mean_total_N = 50  # kg/ha
        self.std_dev_total_N = 35  # kg/ha
        self.percentage_NO3 = 0.85
        self.percentage_NH4 = 0.15
        self.top_30cm_fraction = 0.7
        self.bottom_70cm_fraction = 0.3
        # Sanity check
        # If soil profile is 1m, 30% will have 70% of the total inorganic N
        self.len_top_layers = int(np.ceil(self.len_soil_layers * 0.3))

    def _init_prices(self):
        from cropgymzoo import _CROPS_PRICE, _CONFIG_PATH
        import pandas as pd

        df_crop = pd.read_csv(_CROPS_PRICE)
        df_fert = pd.read_csv(os.path.join(_CONFIG_PATH, 'fertilizer_price.csv'))

        # 2) Drop the “No” column and any unnamed extras
        cols_to_drop = ["No"] + [c for c in df_crop.columns if c.lower().startswith("unnamed")]
        df_crop = df_crop.drop(columns=cols_to_drop, errors="ignore")

        df_crop["Year"] = df_crop["Year"].astype(int)  # or .astype("Int64") if Years can be missing
        df_crop = df_crop.set_index("Year")

        df_fert["Year"] = df_fert["Year"].astype(int)  # or .astype("Int64") if Years can be missing
        df_fert = df_fert.set_index("Year")

        # 4) Build the nested-dict {crop: {year: value}}
        self.crop_prices = {
            crop: {
                int(year): val / 100  # since all prices are euros/ha
                for year, val in df_crop[crop].items()
                if not pd.isna(val)
            }
            for crop in df_crop.columns  # each remaining column is a crop
        }

        self.fertilizer_prices = {
            int(year): val / 100
            for year, val in df_fert["Value"].items()
            if not pd.isna(val)
        }

        # initialize price
        self.fertilizer_price = self._get_fertilizer_price()
        self.costs_nitrogen = self.fertilizer_price

        self.crop_price = self._get_crop_price()


    def _generate_realistic_n(self, random: bool=True, len_soil: int | None = None) -> tuple[list, list]:
        """ method to overwrite a random N initial condition for every call of reset()
            Implemented based on discussions with Herman Berghuijs, for NL conditions
        """

        '''Comments for sanity check'''
        # Generate total inorganic N from seeded normal distribution and clip so that no outliers become negative
        if random:
            total_inorganic_n = self.rng.normal(self.mean_total_N, self.std_dev_total_N)
            total_inorganic_n = np.clip(total_inorganic_n, 0, 100)
        else:
            total_inorganic_n = 40

        # Split total inorganic N into NO3 and NH4
        total_no3 = total_inorganic_n * self.percentage_NO3
        total_nh4 = total_inorganic_n * self.percentage_NH4

        # Distribute 70% of the total inorganic N in the upper 30 cm and 30% in the lower 70 cm
        no3_top = total_no3 * self.top_30cm_fraction
        no3_bottom = total_no3 * self.bottom_70cm_fraction
        nh4_top = total_nh4 * self.top_30cm_fraction
        nh4_bottom = total_nh4 * self.bottom_70cm_fraction

        # Create lists of per layer N content
        if len_soil is not None:
            self.len_soil_layers = len_soil
        no3_distribution = np.zeros(self.len_soil_layers)
        nh4_distribution = np.zeros(self.len_soil_layers)


        if random:
            # Considering 1m soil profile
            # Upper 30 cm distribution (first layers), multiply list of dirichlet with fraction of total in topsoil layers
            no3_distribution[:self.len_top_layers] = self.rng.dirichlet(np.ones(self.len_top_layers), size=1) * no3_top
            nh4_distribution[:self.len_top_layers] = self.rng.dirichlet(np.ones(self.len_top_layers), size=1) * nh4_top

            # Lower 70 cm distribution (last layers), same for remaining bottom layers
            no3_distribution[self.len_top_layers:] = self.rng.dirichlet(np.ones(self.len_soil_layers - self.len_top_layers),
                                                                        size=1) * no3_bottom
            nh4_distribution[self.len_top_layers:] = self.rng.dirichlet(np.ones(self.len_soil_layers - self.len_top_layers),
                                                                        size=1) * nh4_bottom
        else:
            no3_distribution[:self.len_top_layers] = [no3_top / self.len_top_layers] * self.len_top_layers
            nh4_distribution[:self.len_top_layers] = [nh4_top / self.len_top_layers] * self.len_top_layers

            # Lower 70 cm distribution (last layers), same for remaining bottom layers
            no3_distribution[self.len_top_layers:] = ([no3_top / (self.len_soil_layers - self.len_top_layers)] *
                                                      (self.len_soil_layers - self.len_top_layers))
            nh4_distribution[self.len_top_layers:] = ([nh4_top / (self.len_soil_layers - self.len_top_layers)] *
                                                      (self.len_soil_layers - self.len_top_layers))

        # Ensure no negative values in the distributions, might skew the distribution by a teeny bit
        list_nh4i = list(np.maximum(nh4_distribution, 0))
        list_no3i = list(np.maximum(no3_distribution, 0))

        return list_nh4i, list_no3i

    def _get_carbon_dioxide_levels(self):
        """
        Use CMIP5 rcp54 recommendations; range from year 1765 - 2500
        """
        # deprecated linear equation
        # level = np.clip(1.6567 * self.year - 2939, 290.0, 450.0)
        self.carbon_dioxide_level = co2_levels()[self.year]
        return self.carbon_dioxide_level

    def _special_init_conditions(self):
        site_params = None
        if self.random_init:
            site_params = self._overwrite_initial_conditions()
            # for N deposition
            site_params = site_params | self._overwrite_nitrogen_rain_concentration()
        elif not self.random_init:
            site_params = self._overwrite_nitrogen_rain_concentration()

        site_params['CO2'] = self._get_carbon_dioxide_levels()
        return site_params

    @staticmethod
    def _encode_doy(date: datetime.date | datetime.datetime, period: float | None = None):
        if isinstance(date, datetime.datetime):
            date = date.date()

        day_of_year = date.timetuple().tm_yday - 1  # 0-based
        days_in_year = period or (366 if is_leap(date.year) else 365)
        angle = 2 * math.pi * day_of_year / days_in_year

        return math.sin(angle), math.cos(angle)

    def _get_crop_code(self):
        return self.CROP_CODE_MAP[self.crop]

    def _get_fertilizer_price(self):
        try:
            fert_price = self.fertilizer_prices[self.year] \
                            if not self.training \
                            else self.rng.choice(list(self.fertilizer_prices.values()))
            return fert_price
        except KeyError:
            fert_price = list(self.fertilizer_prices.values())[0]
            return fert_price

    def _get_crop_price(self):
        try:
            return self.crop_prices[self.crop][self.year] \
                    if not self.training \
                    else self.rng.choice(list(self.crop_prices[self.crop].values()))
        except KeyError:
            return list(self.crop_prices[self.crop].values())[0]

    def _populate_infos(self, pcse_output, action, reward, terminate):

        self.infos["Date"].append(pcse_output[-1]['day'])

        for feature in self.crop_features:
            f = self._transform_crop_feature(pcse_output[-1], feature)
            self.infos[feature].append(f)

        for feature in self.weather_features:
            self.infos[feature].append(getattr(self.wdp(self.infos["Date"][-1]), feature))

        for feature in self.action_features:
            self.infos[feature].append(self._action_features_mapper()[feature])

        for feature in self.misc_features:
            self.infos[feature].append(self._misc_features_mapper(terminate)[feature])

        self.infos['NAVAIL'].append(pcse_output[-1]['NAVAIL'])

        self.infos['Reward'].append(reward)
        self.infos['Action'].append(action)
        self.infos['Yield'].append(pcse_output[-1]['WSO'])
        self.infos['CropName'].append(self.crop)
        self.infos['Alive'].append(True if not terminate else False)
        self.infos['ActionMask'].append(self.action_mask())

        self.infos['Profit'].append(self.reward_container.cum_profit)
        not self.infos['CO2'] and self.infos['CO2'].append(self.carbon_dioxide_level)

    def _action_features_mapper(self):
        act_mapper = {
            'Naction': self.n_action,
            'Nsteps': self.n_steps,
            'StepsSinceLastAction': self.steps_since_last_action,
            'BudgetTotal': self.budget_n,
            'BudgetLeft': self.budget_left,
        }
        return {k: act_mapper[k] for k in self.action_features if k in act_mapper}

    def _misc_features_mapper(self, terminated = False):
        pcse_output = self.model.get_output() if terminated else None
        misc_process = {
            'SinDay': lambda: self._encode_doy(self.date)[0],
            'CosDay': lambda: self._encode_doy(self.date)[1],
            'FertilizerPrice': lambda: self.fertilizer_price,
            'CropPrice': lambda: self.crop_price,
            'CropCode': lambda: self._get_crop_code(),
            'CO2': lambda: self.carbon_dioxide_level,
            'Nue': lambda: (
                0.0 if not terminated else calculate_nue(
                    n_input=self.reward_container.actions,
                    n_so=process_pcse.get_n_storage_organ(pcse_output),
                    nh4_depo=get_nh4_deposition_pcse(pcse_output),
                    no3_depo=get_no3_deposition_pcse(pcse_output),
                    crop_name=self.crop,
                )
            ),
            'Nsurp': lambda: (
                0.0 if not terminated else get_surplus_n(
                    n_input=self.reward_container.get_total_fertilization,
                    n_so=process_pcse.get_n_storage_organ(pcse_output),
                    nh4_depo=get_nh4_deposition_pcse(pcse_output),
                    no3_depo=get_no3_deposition_pcse(pcse_output),
                    crop_name=self.crop,
                )
            )
        }

        return {k: misc_process[k]() for k in self.misc_features if k in misc_process}

    '''
    Randomizers. NOTE: The weather randomizer is under utils/domain_randomizer.py
    '''

    def _randomise_domain(self, options):
        if self.training:
            self._shift_sowing_date()
            self._perturb_parameters()
            options['site_params']['CO2'] = self._perturb_carbon_dioxide(self._get_carbon_dioxide_levels())
        return options

    def _perturb_parameters(self):
        # get and filter relevant crop params
        crop_params = {key: val for key, val in self._parameter_provider._cropdata.items()
                       if key in self.CROP_PARAMS and isinstance(val, float)}

        for key, val in crop_params.items():
            # perturb by 2 percent
            self._parameter_provider.set_override(key, val*self.rng.normal(1.0, 0.02), check=False)

    def _perturb_carbon_dioxide(self, co2):
        return co2 * self.rng.normal(1.0, 0.1)

    def _shift_sowing_date(self):
        # shift sowing date by normal randomiser with std of 10
        shift = self.rng.normal(loc=0, scale=10)
        shifted_date = self.agmt.crop_start_date + datetime.timedelta(days=shift)
        # shift sowing date and also campaign date proportionally by the random sowing
        self.agro_management = self.agmt.update_attributes(crop_start_date=shifted_date,
                                                            campaign_date=shifted_date - datetime.timedelta(weeks=8))

    '''
    Init helpers
    '''

    def _init_configs(self):
        crop_parameters = YAMLCropDataProvider(fpath=_CROPS_PATH, force_reload=True)

        with open(os.path.join(_SOILGRIDS_PATH, f'soil_{self.location[1]}_{self.location[0]}.yaml'), 'r') as f:
            soil_parameters = yaml.safe_load(f)

        self.carbon_dioxide_level = self._get_carbon_dioxide_levels()

        nh4conc, no3conc = convert_year_to_n_concentration(self.year,
                                                         loc=self.location,)

        # initialize soil variables
        self._init_soil_variables(
            len_soil=len(soil_parameters['SoilProfileDescription']['SoilLayers'])
        )

        nh4i, no3i = self._generate_realistic_n(
            random=False,
        )

        site_parameters = WOFOST81SiteDataProvider_SNOMIN(
            WAV=30,
            CO2=self.carbon_dioxide_level,
            # default init; need to change?
            NH4I=nh4i,
            NO3I=no3i,
            NH4ConcR=nh4conc,
            NO3ConcR=no3conc,
        )
        return crop_parameters, site_parameters, soil_parameters

    def _init_reward_function(self, costs_nitrogen, kwargs):

        self.rewards_obj: Rewards = Rewards(kwargs.get('reward_var'), self.timestep, costs_nitrogen)
        self.reward_container: ActionsContainer | Rewards.__class__ = ActionsContainer()
        self.reward_class: Rewards.__class__ = None

        if self.reward_function == 'ANE':
            self.reward_class = self.rewards_obj.DEF(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerANE(self.timestep)

        elif self.reward_function == 'DEF':
            self.reward_class = self.rewards_obj.DEF(self.timestep, costs_nitrogen)

        elif self.reward_function == 'GRO':
            self.reward_class = self.rewards_obj.GRO(self.timestep, costs_nitrogen)

        elif self.reward_function == 'LOS':
            self.reward_class = self.rewards_obj.LOS(self.timestep, costs_nitrogen)

        elif self.reward_function == 'DEP':
            self.reward_class = self.rewards_obj.DEP(self.timestep, costs_nitrogen)

        elif self.reward_function in reward_functions_end():
            self.reward_class = self.rewards_obj.END(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerEND(self.timestep, costs_nitrogen)

        elif self.reward_function == 'NUE':
            self.reward_class = self.rewards_obj.NUE(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)

        elif self.reward_function == 'PNB':
            self.reward_class = self.rewards_obj.PNB(self.timestep, costs_nitrogen, budget_left=self.budget_left)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)
            self.rewards_obj.crop_price = self.crop_price
            self.rewards_obj.fertilizer_price = self.fertilizer_price

        elif self.reward_function == 'PNY':
            self.reward_class = self.rewards_obj.PNY(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)
            self.rewards_obj.crop_price = self.crop_price
            self.rewards_obj.fertilizer_price = self.fertilizer_price

        elif self.reward_function == 'DNE':
            self.reward_class = self.rewards_obj.DNE(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)

        elif self.reward_function == 'DSO':
            self.reward_class = self.rewards_obj.DSO(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)

        elif self.reward_function == 'NUP':
            self.reward_class = self.rewards_obj.NUP(self.timestep, costs_nitrogen)

        elif self.reward_function == 'HAR':
            self.yield_modifier = 0.2
            self.reward_class = self.rewards_obj.HAR(self.timestep, costs_nitrogen, 200, 5, 1)
            self.reward_container = self.rewards_obj.ContainerEND(self.timestep, costs_nitrogen)

        elif self.reward_function == 'DNU':
            self.reward_class = self.rewards_obj.DNU(self.timestep, costs_nitrogen)

        elif self.reward_function == 'FIN':
            self.reward_class = self.rewards_obj.FIN(self.timestep, costs_nitrogen)

        else:
            raise Exception('please choose valid reward function')

    def _init_soil_variables(self, len_soil: int = None):
        """ Get number of soil layers if using WOFOST snomin"""
        self.mean_total_N = None
        self.std_dev_total_N = None
        self.percentage_NO3 = None
        self.percentage_NH4 = None
        self.top_30cm_fraction = None
        self.bottom_70cm_fraction = None
        self.len_soil_layers = None
        self.len_top_layers = None

        self.len_soil_layers = self.get_len_soil_layers if len_soil is None else len_soil
        self._init_random_init_conditions_params()

    def _init_action_variables(self):
        self.n_action = 0
        self.steps_since_last_action = 0
        """Masking variables"""
        self.n_steps = 0
        # Non Zero constraint
        self.non_zero_action_count = 0
        self.max_non_zero_actions = 4
        # Consecutive constraint
        self.mask_duration = 3
        self.consecutive_mask_counter = 0

    def _init_zero_env(self, **kwargs):
        if self.reward_function in reward_functions_with_baseline() and self.original is not False:
            self._env_baseline = ParcelEnv(
                crop_features=self.crop_features,
                weather_features=self.weather_features,
                location=self.loc,
                year=self.year,
                timestep=self._timestep,
                reward=self.reward_function,
                training=self.training,
                action_space=self.action_space,
                crop=self.crop,
                model_config=self._model_config,
                agro_config=self.agro_config,
                seed=self.seed,
                original=False,
                **kwargs,
            )
            self.zero_nitrogen_env_storage = ZeroNitrogenEnvStorage()

    def _init_obs_keys(self):
        self.obs_keys = tuple(
            self.crop_features +
            self.action_features +
            list(
                f"{name}_{t}"
                for name in self.weather_features
                for t in range(self.timestep)
            )
        )

    def get_harvest_year(self):
        return self.agmt.crop_start_date

    @property
    def get_len_soil_layers(self):
        return len(self.model.kiosk.SM)

    @property
    def model(self):
        return self._model

    @property
    def sb3_env(self):
        return self._env

    @property
    def baseline_env(self):
        return self._env_baseline

    @property
    def date(self):
        return self.model.day

    @property
    def timestep(self):
        return self._timestep

    @property
    def obs_len(self):
        return self._get_obs_len()

    @property
    def act_len(self):
        return len(self.action_space.shape)

    @property
    def static_features(self):
        return ["FertilizerPrice", "CropPrice", "CropCode", "CO2"]

    @property
    def loc(self):
        return self._location

    @loc.setter
    def loc(self, location):
        self._location = location

    @property
    def agro_management(self):
        return self._agro_management

    @agro_management.setter
    def agro_management(self, agro):
        self._agro_management = agro

    @property
    def weather_data_provider(self):
        return self._weather_data_provider

    @weather_data_provider.setter
    def weather_data_provider(self, weather):
        self._weather_data_provider = weather

    @property
    def max_single_dose(self):
        return self.action_space.n - 1

    @property
    def available_doses(self):
        return [a * 10 for a in range(self.action_space.n)]

class ZeroNitrogenEnvStorage:
    """
    Container to store results from zero nitrogen policy (for re-use)
    """

    def __init__(self):
        self.results = {}

    @staticmethod
    def run_episode(env):
        env.reset()
        terminated, truncated = False, False
        infos_this_episode = []
        while not terminated or truncated:
            _, _, terminated, truncated, info = env.step(0)
            infos_this_episode.append(info)
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        return episode_info

    @staticmethod
    def get_key(env):
        # year = env.date.year
        year = env.get_harvest_year()
        location = env.loc
        key = f'{year}-{location}'
        assert 'None' not in key
        return key

    def get_episode_output(self, env):
        key = self.get_key(env)
        if key not in self.results.keys():
            results = self.run_episode(env)
            self.results[key] = results
        assert bool(self.results[key]), "key empty; check PCSE output"
        return self.results[key]

    @property
    def get_result(self):
        return self.results


class CustomFeatureExtractor(nn.Module):
    """
    Average-pools the weather time-series part of the observation and
    concatenates it with the scalar part (crop features, last actions, …).

    Parameters
    ----------
    n_timeseries : int
        Number of weather variables per time step.
    n_scalars : int
        Number of scalar features that are *not* part of the time series.
    n_actions : int, default 0
        If you append the previous action(s) to the observation vector,
        specify how many extra scalar dimensions that represents.
    n_timesteps : int, default 7
        Length of the time window for each weather variable.
    """

    def __init__(
        self,
        n_timeseries: int,
        n_scalars: int,
        n_actions: int = 0,
        n_timesteps: int = 7,
    ):
        super().__init__()
        self.n_timeseries = n_timeseries
        self.n_scalars   = n_scalars
        self.n_actions   = n_actions
        self.n_timesteps = n_timesteps

        # 1-D average pooling over the time dimension
        # (kernel = full window ⇒ one value per weather variable)
        self.avg_timeseries = nn.AvgPool1d(kernel_size=n_timesteps)

        # For convenience if you need the size later (e.g. to build policy heads)
        self.features_dim = n_timeseries + n_scalars + n_actions

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        observations : torch.Tensor
            Shape (batch,  n_scalars + n_actions + n_timeseries * n_timesteps)

        Returns
        -------
        torch.Tensor
            Shape (batch, features_dim)
        """
        batch_size = observations.shape[0]

        # Split the flat vector into [scalars | timeseries]
        flat_scalars = observations[:, : self.n_scalars + self.n_actions]
        flat_series  = observations[:, self.n_scalars + self.n_actions :]

        # Reshape to (batch, channels, timesteps) for AvgPool1d
        series = flat_series.view(batch_size,
                                  self.n_timeseries,
                                  self.n_timesteps)

        # Pool, squeeze time dimension, keep channels
        pooled = self.avg_timeseries(series).squeeze(-1)  # (batch, n_timeseries)

        # Concatenate pooled weather with scalar part
        return th.cat((pooled, flat_scalars), dim=1)