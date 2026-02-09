import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import math
from copy import deepcopy

import yaml
from collections import OrderedDict

import functools
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle

import numpy as np
import datetime

from pcse.input.sitedataproviders import WOFOST81SiteDataProvider_SNOMIN
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider

from cropgymzoo import (
    _WOFOST_CONFIG,
    _AGRO_CALENDAR_CONFIG,
    _CROPS_PATH,
    _SOIL_PATH,
    _SITE_PATH,
    _SOILGRIDS_PATH,
    _CROPS_CONFIG
)

import cropgymzoo.utils.process_pcse_output as process_pcse
from cropgymzoo.utils.rewards import (
    Rewards,
    ActionsContainer,
    reward_functions_with_baseline,
    reward_functions_end,
    calculate_nue
)
from cropgymzoo.utils.nitrogen_helpers import (
    get_surplus_n,
    get_nh4_deposition_pcse,
    get_no3_deposition_pcse,
    convert_year_to_n_concentration,
    m2_to_ha,
    is_leap,
    co2_levels
)
from cropgymzoo.utils.curriculum import RandomiseStage
from cropgymzoo.utils_soil.env_soil_functions import soil_to_latent_pca
import cropgymzoo.envs.pcse_env as pcse_env
from cropgymzoo.utils.defaults import (
    get_wofost_default_crop_features,
    get_default_weather_features,
    get_default_action_features,
    get_default_misc_features,
    get_default_soil_pc_features,
    get_concise_misc_features,
    get_wofost_concise_crop_features,
)
from cropgymzoo.utils.curriculum import make_default_stage_manager
from cropgymzoo.utils.scenario_utils import get_scenario_based_on_name, get_scenario_based_on_loc, get_coords_for_soil
from cropgymzoo.utils.agent_helpers import last_before_nan

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

    # modified/lowered by 5kg/ha to match RL agent actions
    CROP_SOIL_MAX = {
        "winterwheat": {'clay': 250, 'sand': 160, 'silt': 190, 'peat': 160},
        "sugarbeet": {'clay': 150, 'sand': 150, 'silt': 120, 'peat': 140},
        "potato": {'clay': 280, 'sand': 260, 'silt': 210, 'peat': 270},
        "barley": {'clay': 80, 'sand': 80, 'silt': 80, 'peat': 80},
        "seed_onion": {'clay': 170, 'sand': 120, 'silt': 120, 'peat': 120},
        'rapeseed': {'clay': 200, 'sand': 190, 'silt': 150, 'peat': 190},
        'sunflower': {'clay': 150, 'sand': 150, 'silt': 150, 'peat': 150},
    }

    ACTION_LEVELS = [0, 1, 3, 7, 12]  # index 0..4

    '''
    Initialize Env for each RL agent
    '''
    def __init__(self,
                 crop_features: list = get_wofost_default_crop_features(),
                 weather_features: list = get_default_weather_features(),
                 action_features: list = get_default_action_features(),
                 misc_features: list = get_default_misc_features(),
                 soil_features: list = get_default_soil_pc_features(),
                 location: list | tuple = None,
                 year: int = None,
                 year_list: list = None,
                 timestep: int = 7,
                 reward: str = 'PNR',
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
                 original: bool = True,
                 flatten_obs: bool = True,
                 training: bool = False,
                 keep_soil_moisture: bool = False,
                 domain_repeat: int = 10,
                 special_action_space: bool = False,
                 concise_obs: bool = False,
                 **kwargs,
    ):
        EzPickle.__init__(
            self,
            crop_features=crop_features,
            weather_features=weather_features,
            action_features=action_features,
            misc_features=misc_features,
            soil_features=soil_features,
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
            keep_soil_moisture=keep_soil_moisture,
            domain_repeat=domain_repeat,
            special_action_space=special_action_space,
            concise_obs=concise_obs,
            **kwargs,
        )
        # instance metadata
        self.original = original
        self.training = training
        self.flatten_obs = flatten_obs
        self.name = name
        self.soil_type = type
        self.keep_soil_moisture = keep_soil_moisture
        self.special_action_space = special_action_space
        if self.special_action_space:
            action_space = gym.spaces.Discrete(5)
        self.concise_obs = concise_obs
        # pcse variables
        self.crop = crop
        self.crop_features = crop_features
        self.weather_features = weather_features
        self.action_features = action_features
        self.misc_features = misc_features
        self.soil_features = soil_features
        if self.concise_obs:
            self.crop_features = get_wofost_concise_crop_features()
            self.misc_features = get_concise_misc_features()
        self.year = year
        self.location = location
        self.year_list = year_list

        # field specific stuff
        self.max_budget_n = self.CROP_SOIL_MAX[self.crop][self.soil_type]
        self.budget_n = self.max_budget_n
        self.budget_left = self.budget_n
        self.area = float(area)
        self.area_orig = float(area)
        self.day_of_planting: datetime.date | None = None

        # random generator
        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)

        self.domain_repeat = int(domain_repeat)
        self._domain_repeat_left = 0  # how many episodes left to reuse the current domain
        self._domain_spec = None  # the cached domain specification

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

        self.feature_index_map = self._build_index_map()

        # possibly deprecated
        # self.costs_nitrogen = costs_nitrogen
        self.action_multiplier = action_multiplier

        # env stuff
        self.action_space = action_space
        self._timestep = timestep
        self.reward_function = reward

        # Training stuff
        self.random_manager = make_default_stage_manager()

        # get coords for soil
        self.soil_coords = get_coords_for_soil(get_scenario_based_on_name(self.name))

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
        self.flag_gather_zero_env = True
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
        if self.special_action_space:
            if not isinstance(action, int):
                action = action.item()
            action = self.ACTION_LEVELS[action]

        # make sure SM is above a certain level
        if self.keep_soil_moisture:
            self._do_auto_irrigation()

        # align crop name from active campaign
        specs = getattr(self, "_campaign_specs", None)
        ptr = getattr(self, "_campaign_ptr", 0)
        self.crop = (specs[ptr].get("crop_name", self.crop) if specs else self.crop)

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

        # if terminated:
        #     self.infos["infos_by_year"] = self._group_infos_by_season_year()

        return obs, reward, terminated, truncated, self.infos

    def reset(self, seed=None, options=None, **kwargs):
        """
        Resets parcel episode. The key `year` must be in the ::options dictionary.
        Here, growing season budget is also determined.
        """

        if options is None:
            options = {}
        assert 'year' in options, "Please reset environment with a year"
        # Keep `self.year` consistent for downstream code paths (prices, logging, etc.)
        self.year = int(options['year'])

        # Only for invalid action masking
        # self.reset_non_zero_action_count()

        # return the original agro management class shifts during randos
        self.agmt = deepcopy(self.original_agmt)
        self.area = self.area_orig

        # reset reward runners
        self.reward_container.reset()
        self.rewards_obj.reset()

        # reset various variables
        self._reset_action_variables()

        if "eval_horizon_years" in options and "farm_dict_by_year" in options:
            horizon_years = list(options["eval_horizon_years"])
            farm_by_year = options["farm_dict_by_year"]

            def _safe_replace(d: datetime.date, year: int) -> datetime.date:
                try:
                    return d.replace(year=year)
                except ValueError:
                    return d.replace(year=year, day=28)

            campaign_specs = []
            for y in horizon_years:
                crop_y = farm_by_year[y][self.name]["crop"]  # self.name is field id like "field-3"
                tpl = self.crop_info[crop_y]

                start_year = (y - 1) if crop_y == "winterwheat" else y

                campaign_date = _safe_replace(self.agmt.str_to_datetime(tpl["campaign_date"]), start_year)
                crop_start_date = _safe_replace(self.agmt.str_to_datetime(tpl["crop_start_date"]), start_year)

                crop_end_date = None
                if "crop_end_date" in tpl and tpl["crop_end_date"] not in (None, "", "null"):
                    crop_end_date = _safe_replace(self.agmt.str_to_datetime(tpl["crop_end_date"]), y)

                campaign_specs.append({
                    "campaign_date": campaign_date,
                    "crop_name": tpl.get("crop_name", crop_y),
                    "variety_name": tpl.get("variety_name", self.agmt.variety_name),
                    "crop_start_date": crop_start_date,
                    "crop_start_type": tpl.get("crop_start_type", self.agmt.crop_start_type),
                    "crop_end_date": crop_end_date,
                    "crop_end_type": tpl.get("crop_end_type", self.agmt.crop_end_type),
                    "max_duration": int(tpl.get("max_duration", self.agmt.max_duration))
                                    if self.agmt.max_duration is not None else None,
                })

            # Attach label years to specs for grouping (year label is your season year)
            for spec, y in zip(campaign_specs, horizon_years):
                spec["label_year"] = int(y)

            self._init_campaign_tracking(campaign_specs)
            # Precompute per-season prices so evaluation runs can reflect year-varying prices
            self._init_price_tracking(horizon_years=horizon_years, farm_by_year=farm_by_year)

            self._agro_management = pcse_env.AgroManagementContainer.build_multi_campaign_structure(campaign_specs)

            # Evaluation: don't skip to sowing, and ensure multi-crop parameters exist
            options["wait_for_crop"] = bool(options.get("wait_for_crop", False))
            options["multi_crop"] = True

        else:
            # Original single-season behavior
            self.overwrite_year(year=options["year"])

            self._init_campaign_tracking(None)
            if hasattr(self, '_price_by_season_year'):
                self._price_by_season_year = {}

        # use options. Shift soil and randomise N conditions
        # site_params = self._special_init_conditions()
        # options['site_params'] = site_params

        # randomise domain
        options = self._randomise_domain(options)

        # reset prices
        self._reset_prices()

        # manual override for randomisation
        if options.get("random_initial_conditions", False):
            options["site_params"] = self._overwrite_initial_conditions(random=True)
            options["site_params"] = {'WAV': self.rng.normal(30, 10)}

        # reset PCSE
        obs = super().reset(seed=seed, options=options)

        # For multi-year runs, `agmt` may not reflect the full chained structure; use the first campaign start when possible.
        if getattr(self, '_campaign_specs', None):
            try:
                self.day_of_planting = self._campaign_specs[0].get('crop_start_date', self.agmt.crop_start_date)
            except Exception:
                self.day_of_planting = self.agmt.crop_start_date
        else:
            self.day_of_planting = self.agmt.crop_start_date
        self._update_budget_left()

        # get infos
        self._init_infos()
        self._populate_infos(self.model.get_output(), 0, 0, False)

        # reset baseline
        if self.reward_function in reward_functions_with_baseline() and self.original is True:
            self.baseline_env.year = self.year
            self.baseline_information['infos'], self.baseline_information['pcse_output'] = \
                self.zero_nitrogen_env_storage.get_episode_output(
                    self.baseline_env,
                    spec=self._domain_spec
                )
            # self.baseline_env.rng.bit_generator.state = self.rng.bit_generator.state
            # self.baseline_env.reset(seed=seed, options=options)

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
        end_date = {
            'crop_end_date': lambda d: d.replace(
                year=(
                    year - 1
                    if self.crop == 'winterwheat'
                    else year
                )
            )
        } if self.agmt.crop_end_type == "harvest" else {}
        self.agro_management = self.agmt.update_attributes(
            crop_start_date=lambda d: d.replace(
                year=(
                    year - 1
                    if self.crop == 'winterwheat'
                    else year
                )
            ),
            campaign_date=lambda d: self._safe_replace_year(
                d,
                (
                    year - 1
                    if self.crop == 'winterwheat'
                    else year
                )
            ),
            **end_date
        )
        if self.reward_function in reward_functions_with_baseline() and self.original is True:
            self.baseline_env.agro_management = self.agmt.update_attributes(
                crop_start_date=lambda d: d.replace(
                    year=(
                        year - 1
                        if self.crop == 'winterwheat'
                        else year
                    )
                ),
                campaign_date=lambda d: self._safe_replace_year(
                    d,
                    (
                        year - 1
                        if self.crop == 'winterwheat'
                        else year
                    )
                ),
                **end_date
            )

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

    @staticmethod
    def obs_constraint_features():
        return ['DVS', 'NonZeroActionCount', 'Nue', 'Nsurp']

    @staticmethod
    def obs_budget_features():
        return ['BudgetLeft', 'BudgetTotal']

    def make_special_action_space(self):
        self.action_space = gym.spaces.Discrete(5)
        self.special_action_space = True

    def reconfigure(
            self,
            *,
            crop: str,
            year: int,
            location: tuple[float, float],
            area: float,
            soil_type: str,
    ):
        """
        This method only updates configuration so that the NEXT reset()
        rebuilds the PCSE model correctly. It must not rebuild the model itself.
        """

        # ---- config metadata ----
        self.crop = crop
        self.year = int(year)

        self.location = tuple(location)
        self._location = self.location  # PCSEEnv uses _location internally

        self.soil_type = soil_type

        # reset() restores area from area_orig -> update it here to persist
        self.area = float(area)
        self.area_orig = float(area)

        # ---- budgets ----
        self.max_budget_n = self.CROP_SOIL_MAX[self.crop][self.soil_type]
        self.budget_n = self.max_budget_n
        self.budget_left = self.max_budget_n

        # ---- parameters used by PCSEEnv reset ----
        crop_parameters, site_parameters, soil_parameters = self._init_configs()
        self._crop_params = crop_parameters
        self._site_params = site_parameters
        self._soil_params = soil_parameters

        # ---- agromanagement ----
        if getattr(self, "crop_info", None):
            self._agro_management = self.agmt.update_attributes(**self.crop_info[self.crop])

            # reset() restores agmt from original_agmt -> update snapshot
            self.original_agmt = deepcopy(self.agmt)

        # DO NOT touch self._model here.
        # Caller should do: reset(options={'year': self.year, ...})

    # ------------------------------------------------------------------
    # Multi-season evaluation helpers
    # ------------------------------------------------------------------
    def begin_new_season(
        self,
        *,
        season_year: int,
        crop_name: str | None = None,
        max_budget_n: float | None = None,
        reset_reward: bool = True,
        reset_actions: bool = True,
        reset_prices: bool = True,
    ) -> None:
        """Prepare this parcel env for a new season *without* resetting the PCSE model.

        Used for daisy-chained evaluation where the PCSE state (soil N, moisture, etc.)
        must carry over across campaigns.

        What this resets:
        - action counters / action-derived info variables (Naction, NonZeroActionCount, etc.)
        - season budget (budget_n, budget_left, max_budget_n)
        - reward accumulators/prices (optional)

        What this does NOT reset:
        - the PCSE model state (Engine/kiosk), weather provider, soil water, mineral N pools
        """

        # Update season/year label used by price lookups etc.
        self.year = int(season_year)

        # Update crop label used throughout reward + info (PCSE crop itself is controlled by agromanager)
        if crop_name is not None:
            self.crop = str(crop_name)

        # Reset reward trackers (profit, constraint accumulators, etc.) per season
        if reset_reward:
            self.reward_container.reset()
            self.rewards_obj.reset()

        # Reset action tracking / counters per season
        if reset_actions:
            self._reset_action_variables()

        # Reset seasonal budgets
        if max_budget_n is not None:
            self.max_budget_n = float(max_budget_n)
            self.budget_n = float(max_budget_n)
            # budget_left should be full at start of season
            self.budget_left = float(max_budget_n)

        # Refresh prices for the season/crop if requested
        if reset_prices:
            self._reset_prices()


    '''
    Helper functions for various things
    '''

    def _build_index_map(self) -> dict[str, int]:
        """Create a mapping from feature keyword to its index in the obs vector."""
        index_map = {}
        offset = 0

        # crop features
        for f in self.crop_features:
            index_map[f] = offset
            offset += 1

        # action features
        for f in self.action_features:
            index_map[f] = offset
            offset += 1

        # misc features
        for f in self.misc_features:
            index_map[f] = offset
            offset += 1

        # weather features across timesteps
        for i, f in enumerate(self.weather_features):
            for t in range(self._timestep):
                index_map[f"{f}_{t}"] = offset + i * self._timestep + t
        # offset += len(self.weather_features) * self.timestep   # not needed unless chaining more

        return index_map

    def get_idx_features(self, feature_list: list[str]) -> list[int]:
        out = []
        for f in feature_list:
            if f not in self.feature_index_map:
                raise KeyError(f"Feature '{f}' not in feature_index_map")
            out.append(self.feature_index_map[f])
        return out


    @staticmethod
    def _safe_replace_year(d, year):
        try:
            return d.replace(year=year)
        except ValueError:
            # fallback for Feb 29 to Feb 28
            return d.replace(year=year, day=28)

    def _reset_prices(self):
        if getattr(self, "_domain_spec", None):
            fp = self._domain_spec.get('fertilizer_price', None)
            cp = self._domain_spec.get('crop_price', None)
            if fp is not None and cp is not None:
                self.fertilizer_price = float(fp)
                self.costs_nitrogen = self.fertilizer_price
                self.crop_price = float(cp)
                return

        self.fertilizer_price = self._get_fertilizer_price()
        self.crop_price = self._get_crop_price()

    def _init_price_tracking(self, horizon_years: list[int], farm_by_year: dict):
        """Precompute per-season prices for multi-year evaluation.

        Stores mapping: season_label_year -> {'crop': str, 'crop_price': float, 'fertilizer_price': float}

        This uses the existing `_get_crop_price()` and `_get_fertilizer_price()` logic by temporarily
        setting `self.year`/`self.crop` for each season.
        """
        price_map: dict[int, dict] = {}
        # snapshot current state
        _year0 = getattr(self, 'year', None)
        _crop0 = getattr(self, 'crop', None)
        _cp0 = getattr(self, 'crop_price', None)
        _fp0 = getattr(self, 'fertilizer_price', None)

        try:
            for y in horizon_years:
                crop_y = farm_by_year[y][self.name]['crop']
                # temporarily set context for existing price helpers
                self.year = int(y)
                self.crop = str(crop_y)
                try:
                    cp = float(self._get_crop_price())
                except Exception:
                    cp = float(_cp0) if _cp0 is not None else 0.0
                try:
                    fp = float(self._get_fertilizer_price())
                except Exception:
                    fp = float(_fp0) if _fp0 is not None else 0.0
                price_map[int(y)] = {
                    'crop': str(crop_y),
                    'crop_price': cp,
                    'fertilizer_price': fp,
                }
        finally:
            # restore
            if _year0 is not None:
                self.year = _year0
            if _crop0 is not None:
                self.crop = _crop0
            if _cp0 is not None:
                self.crop_price = _cp0
            if _fp0 is not None:
                self.fertilizer_price = _fp0

        self._price_by_season_year = price_map


    def _maybe_update_prices_for_date(self, d: datetime.date):
        """Update `self.crop_price` / `self.fertilizer_price` for the current simulation date.

        For multi-year evaluation runs, this ensures prices reflect the active season label year.
        """
        season_year, active_crop = self._active_campaign_for_date(d)
        if season_year is None:
            return
        mp = getattr(self, '_price_by_season_year', None)
        if not mp or season_year not in mp:
            return
        rec = mp[season_year]
        # keep crop in sync with active campaign for any downstream logic
        if active_crop is not None:
            self.crop = active_crop
        if 'crop_price' in rec:
            self.crop_price = float(rec['crop_price'])
        if 'fertilizer_price' in rec:
            self.fertilizer_price = float(rec['fertilizer_price'])


    def _update_budget_left(self):
        self.reward_class.budget_left = self.budget_left

    def _update_action_variables(self, action):
        if isinstance(action, np.ndarray):
            action = action[0] if action.shape else action

        self._update_budget_left()

        self.action: int = action
        if action > 0:
            self.n_action += action * 10
            self.non_zero_action_count += 1
            self.steps_since_last_action = 0

            # budget count
            self.budget_left -= action * 10         # Convert to kg/ha
        else:
            self.steps_since_last_action += 1

    def _reset_action_variables(self):
        self.action = 0
        self.n_steps = 0
        self.n_action = 0
        self.non_zero_action_count = 0
        self.steps_since_last_action = 0
        self.budget_left = self.budget_n

    # For constraints

    # def _get_frequency_constraint(self) -> float:
    #     return 1.0 if self.non_zero_action_count > 4 else 0.0

    def _get_frequency_constraint(self, terminated: bool) -> float:
        acted = float(self.action) > 0.0
        step_cost = 0.0
        if acted:
            # small per-step penalty only when over K
            step_cost = 0.2 if self.non_zero_action_count > 4 else 0.0

        # term_cost = 0.0
        # if terminated:
        #     if self.non_zero_action_count > 4:
        #         diff = float(self.non_zero_action_count - 4)
        #         term_cost = (diff / 4) ** 2
        #     else:
        #         term_cost = 0.0  # no penalty if under or equal to max

        return step_cost # + term_cost

    def _get_consecutive_constraint(self) -> float:
        acted = float(self.action) > 0.0
        if acted:
            if self.steps_since_last_action < 3:
                return (3 - self.steps_since_last_action) * 0.1
            else:
                return 0.0
        else:
            return 0.0

    def _get_dvs_constraint(self) -> float:
        dvs = self.model.get_output()[-1]['DVS']
        acted = self.action > 0  # or whatever check means "fertilizer applied"
        if acted and not (0.01 < dvs <= 1):
            return 0.2
        return 0.0

    def _get_budget_constraint(self, terminated) -> float:
        if terminated:
            return self.budget_left // 10
        else:
            return 0

    def _get_nue_constraint(self) -> float:
        return (
            0
            if self.infos['Nue'][-1] == 0.0
               or 0.5 <= self.infos['Nue'][-1] <= 0.9
            else 1
        )

    def _get_nsurp_constraint(self) -> float:
        return (
            0
            if 0.0 <= self.infos['Nsurp'][-1] <= 40
            else 1
        )

    def _calculate_constraints(self, terminated):
        """
        Function to calculate constraints
        1. Action constraint
        2. Development stage constraint
        3. Budget constraint (if not using masked actions)
        4. Nue and Nsurp constraints
        Commented out some constraints to not let the agents use them
        """
        total_constraint = 0.0
        total_constraint += self._get_frequency_constraint(terminated)
        total_constraint += self._get_dvs_constraint()
        if self.crop == 'winterwheat':
            total_constraint += self._get_consecutive_constraint()
        # total_constraint += self._get_budget_constraint(terminated)
        # total_constraint += self._get_nue_constraint()
        # total_constraint += self._get_nsurp_constraint()

        return total_constraint


    def _process_output(self, action, output, terminated):
        if isinstance(action, np.ndarray):
            action = action.item()

        # TIMES 10 HERE
        amount = action * 10  #kg/ha

        output_baseline = []
        if self.reward_function in reward_functions_with_baseline() and self.original is True:

            output_baseline = self.baseline_information['pcse_output']
            # Trim to current date
            output_baseline = output_baseline[:len(output)]

        self.rewards_obj.update_profit(output, amount, year=self.date.year)

        prices = {
            'price_fertilizer': self.fertilizer_price,
            'price_crop': self.crop_price,
            'budget_left': self.budget_left,
        }
        yield_fn = {
            "fresh_yield_fn": self._get_fresh_weight,
        }
        reward, growth = self.reward_class.return_reward(
            output,
            amount,
            output_baseline=output_baseline,
            obj=self.reward_container,
            **prices
            if self.reward_function in ['PNY', 'PNB', 'PNR', 'MPN', "NSU"]
            else {},
            **yield_fn
            if self.crop in ['winterwheat', 'sugarbeet', 'potato']
            and self.reward_function in ['PNY', 'PNB', 'PNR', 'MPN', 'NSU']
            else {},
        )
        del prices
        del yield_fn

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

        elif terminated and self.reward_function in ['PNB', 'PNR']:
            if not self.original:
                self.infos['FinalReward'] = reward
            if self.baseline_information is not None and self.original:
                x = self.baseline_information['infos']['Reward'][-1] - self.baseline_information['infos']['FinalReward'] - reward
            final_reward = (
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
            if self.baseline_information is not None and self.original:
                final_reward = round(final_reward - x, 1)
            return final_reward

        elif terminated and self.reward_function == 'NSU':
            _ = self.reward_class.return_final_reward(
                obj=self.reward_container,
                n_fertilized=self.reward_container.get_total_fertilization,
                n_output=process_pcse.get_n_storage_organ(output),
                no3_depo=get_no3_deposition_pcse(output),
                nh4_depo=get_nh4_deposition_pcse(output),
                budget_left=self.budget_left,
                crop_name=self.crop,
            )
            return self.reward_container.calculate_reward_nsurp(
                n_fertilized=self.reward_container.get_total_fertilization,
                n_output=process_pcse.get_n_storage_organ(output),
                no3_depo=get_no3_deposition_pcse(output),
                nh4_depo=get_nh4_deposition_pcse(output),
                crop_name=self.crop
            )

        return 0

    def _overwrite_initial_conditions(self, random=True):
        # N initial conditions
        list_nh4i, list_no3i = self._generate_realistic_n(random=random)
        self.eval_nh4i = list_nh4i
        self.eval_no3i = list_no3i

        site_parameters = {'NH4I': list_nh4i, 'NO3I': list_no3i, }
        return site_parameters

    def _overwrite_nitrogen_rain_concentration(self):
        # N concentration in rain for deposition
        nh4concr, no3concr = convert_year_to_n_concentration(2021,
                                                             loc=self.location,
                                                             wdp=self.weather_data_provider,)

        site_parameters = {'NH4ConcR': nh4concr, 'NO3ConcR': no3concr, }
        return site_parameters

    def _do_auto_irrigation(self):
        sum_moisture = np.sum(self.model.get_output()[-1]["SM"])
        if sum_moisture < 2.5:
            self.model._flag_irrigate = True

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
        soil = self._soil_features_mapper()

        # perform some transformations
        crop_values = [
            self._get_key_transformations(crop_model).get(f, lambda: crop_model[f][-1])()  # fall back to plain last value
            for f in self.crop_features
        ]

        action_values = [act[a] for a in self.action_features]

        misc_values = [misc[m] for m in self.misc_features]

        soil_values = [soil[m] for m in self.soil_features]

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
            return np.array(
                crop_values + action_values + misc_values + soil_values + list(weather_values),
                dtype=np.float32
            )

        obs = {
            **{k: float(v) for k, v in zip(self.crop_features, crop_values)},
            **{k: float(v) for k, v in zip(self.action_features, action_values)},
            **{k: float(v) for k, v in zip(self.misc_features, misc_values)},
            **{k: float(v) for k, v in zip(self.soil_features, soil_values)},
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
        nvars = (len(self.crop_features) + len(self.action_features) + len(self.soil_features) +
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
            "Date": [],
            'SeasonYear': [],
            'CropActive': [],
            **{name: [] for name in self.crop_features},
            **{name: [] for name in self.weather_features},
            **{name: [] for name in self.action_features},
            **{name: [] for name in self.misc_features},
            'Reward': [], 'Action': [], 'Yield': [], 'NAVAIL': [],
            'BudgetTotal': [], 'BudgetLeft': [], 'CropName': [],
            'Nue': [], 'Nsurp': [], 'Profit': [], "CO2": [],
            'Alive': [], 'ActionMask': [], 'RFTRA': [], 'WC': [],
            'TotalConstraint': [], 'FrequencyConstraint': [],
            'DVSConstraint': [], 'BudgetConstraint': [],
            'NueConstraint': [], 'NsurpConstraint': [], 'ConsecutiveConstraint': [],
            'TotalEpisodicConstraint': [], 'DaysAfterPlanting': [],
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
        self.carbon_dioxide_level = 430  # co2_levels()[self.year]
        return self.carbon_dioxide_level

    def _special_init_conditions(self):
        site_params = None
        if self.training and self.random_manager.initial_n:
            site_params = self._overwrite_initial_conditions()
            # for N deposition
            site_params = site_params | self._overwrite_nitrogen_rain_concentration()
        else:
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
        return self.CROP_CODE_MAP[(self.infos["CropName"][-1] if self.infos.get("CropName") else self.crop)]

    def _get_fertilizer_price(self):
        try:
            fert_price = self.fertilizer_prices[self.year] \
                            if not self.training \
                            else self.rng.choice(list(self.fertilizer_prices.values()))
            return fert_price
        except KeyError:
            fert_price = list(self.fertilizer_prices.values())[-1]
            return fert_price

    def _get_fresh_weight(
            self,
            wso,
    ):
        if wso is None:
            return np.nan
        if self.crop == 'winterwheat':
            return wso / 0.85  # 15% water/moisture assumption
        elif self.crop == 'sugarbeet':
            return wso / 0.23  # 77% water
        elif self.crop == 'potato':
            return wso / 0.20  # 80% water
        else:
            return wso

    def _get_crop_price(self):
        try:
            return self.crop_prices[self.crop][self.year] \
                    if not self.training \
                    else self.rng.choice(list(self.crop_prices[self.crop].values()))
        except KeyError:
            return list(self.crop_prices[self.crop].values())[-1]

    def _init_campaign_tracking(self, campaign_specs: list[dict] | None):
        """
        Initialize per-episode campaign tracking for daisy-chained multi-year runs.

        campaign_specs elements must include:
          - campaign_date (datetime.date)
          - label_year (int)
          - crop_name (str)
        """
        if campaign_specs is None:
            self._campaign_specs = None
            self._campaign_ptr = None
            return
        self._campaign_specs = sorted(campaign_specs, key=lambda s: s["campaign_date"])
        self._campaign_ptr = 0

    def _active_campaign_for_date(self, d: datetime.date) -> tuple[int | None, str | None]:
        """Return (season_label_year, crop_name) for a given simulation date d."""
        if not getattr(self, "_campaign_specs", None):
            return None, None

        specs = self._campaign_specs
        ptr = int(getattr(self, "_campaign_ptr", 0) or 0)
        ptr = max(0, min(ptr, len(specs) - 1))

        # advance pointer while next campaign has started
        while (ptr + 1) < len(specs) and specs[ptr + 1]["campaign_date"] <= d:
            ptr += 1
        self._campaign_ptr = ptr

        spec = specs[ptr]
        y = spec.get("label_year", None)
        return (int(y) if y is not None else None), spec.get("crop_name", None)

    def _group_infos_by_season_year(self) -> dict:
        """Return a dict mapping SeasonYear -> dict of sliced info series."""
        if "SeasonYear" not in self.infos or not self.infos["SeasonYear"]:
            return {}
        years = [y for y in self.infos["SeasonYear"] if y is not None]
        if not years:
            return {}

        uniq = sorted(set(int(y) for y in years))
        by_year = {}

        for y in uniq:
            idx = [i for i, yy in enumerate(self.infos["SeasonYear"]) if yy == y]
            sub = {}
            for k, seq in self.infos.items():
                try:
                    sub[k] = [seq[i] for i in idx]
                except Exception:
                    continue
            by_year[int(y)] = sub

        return by_year

    def get_latest_season_info(
            self,
            key: str,
            season_year: int,
            default=None,
            *,
            skip_nan: bool = True,
            skip_none: bool = True,
            skip_inf: bool = True,
            require_crop_active: bool = True,
    ):
        """Return the last *valid* value of infos[key] restricted to SeasonYear==season_year.

        Robust to post-harvest/maturity tails that may contain None/NaN/Inf.

        If require_crop_active=True and infos['CropActive'] exists, only consider entries where
        CropActive is True (useful for daisy-chained multi-season runs where values after harvest
        or between campaigns can be NaN/None).
        """

        def _is_invalid(v) -> bool:
            if skip_none and v is None:
                return True

            # numeric scalars (python + numpy)
            if isinstance(v, (int, float, np.integer, np.floating)):
                try:
                    fv = float(v)
                except Exception:
                    return False  # if it can't be cast, treat as non-numeric
                if skip_nan and np.isnan(fv):
                    return True
                if skip_inf and np.isinf(fv):
                    return True
                if (skip_nan or skip_inf) and (not np.isfinite(fv)):
                    return True

            return False

        try:
            infos = getattr(self, "infos", {})
            sy_list = infos.get("SeasonYear", [])
            seq = infos.get(key, [])
            if not sy_list or not seq:
                return default

            crop_active_list = infos.get("CropActive", None)

            n = min(len(sy_list), len(seq))
            if crop_active_list is not None:
                n = min(n, len(crop_active_list))

            target = int(season_year) if season_year is not None else season_year

            # walk backwards: last matching season AND valid value (and optionally CropActive)
            for i in range(n - 1, -1, -1):
                if sy_list[i] != target:
                    continue

                if require_crop_active and crop_active_list is not None:
                    if not bool(crop_active_list[i]):
                        continue

                v = seq[i]
                if _is_invalid(v):
                    continue
                return v

            # fallback: if we required CropActive and found nothing, try again without it
            if require_crop_active and crop_active_list is not None:
                for i in range(n - 1, -1, -1):
                    if sy_list[i] != target:
                        continue
                    v = seq[i]
                    if _is_invalid(v):
                        continue
                    return v

            return default
        except Exception:
            return default

    def get_season_indices(self, season_year: int) -> list[int]:
        """Indices i where SeasonYear[i] == season_year."""
        try:
            sy_list = self.infos.get("SeasonYear", [])
            return [i for i, yy in enumerate(sy_list) if yy == season_year]
        except Exception:
            return []

    def _populate_infos(self, pcse_output, action, reward, terminate):

        self.infos["Date"].append(pcse_output[-1]['day'])

        # Multi-year daisy-chaining support: map each day to a season label year + active crop
        d = self.infos["Date"][-1]
        season_year, active_crop = self._active_campaign_for_date(d)
        self.infos["SeasonYear"].append(season_year)

        # Determine whether crop is active (helps interpret NaNs between campaigns)
        dvs_val = pcse_output[-1].get("DVS", None)
        try:
            crop_active = (dvs_val is not None) and (not np.isnan(dvs_val))
        except Exception:
            crop_active = dvs_val is not None
        self.infos["CropActive"].append(bool(crop_active))

        for feature in self.crop_features:
            f = self._transform_crop_feature(pcse_output[-1], feature)
            self.infos[feature].append(f)

        for feature in self.weather_features:
            self.infos[feature].append(getattr(self.wdp(self.infos["Date"][-1]), feature))

        for feature in self.action_features:
            self.infos[feature].append(self._action_features_mapper()[feature])

        for feature in self.misc_features:
            self.infos[feature].append(self._misc_features_mapper(terminate)[feature])

        self.infos['Reward'].append(reward)
        self.infos['Action'].append(action if not isinstance(action, (np.ndarray, th.Tensor)) else action.item())
        self.infos['Yield'].append(self._get_fresh_weight(pcse_output[-1]['WSO']))
        # Use campaign-derived crop name when available (daisy-chained multi-crop runs)
        self.infos["CropName"].append(active_crop if active_crop is not None else self.crop)
        self.infos['Alive'].append(True if not terminate else False)
        self.infos['ActionMask'].append(self.action_mask())
        self.infos['RFTRA'].append(pcse_output[-1]['RFTRA'])
        self.infos['WC'].append(np.sum(pcse_output[-1]['WC']))

        self.infos['Profit'].append(self.reward_container.cum_profit)
        not self.infos['CO2'] and self.infos['CO2'].append(self.carbon_dioxide_level)

        self.infos['TotalConstraint'].append(self._calculate_constraints(terminate))
        self.infos['FrequencyConstraint'].append(self._get_frequency_constraint(terminate))
        self.infos['DVSConstraint'].append(self._get_dvs_constraint())
        self.infos['BudgetConstraint'].append(self._get_budget_constraint(terminate))
        self.infos['ConsecutiveConstraint'].append(self._get_consecutive_constraint())
        self.infos['NueConstraint'].append(self._get_nue_constraint())
        self.infos['NsurpConstraint'].append(self._get_nsurp_constraint())

        self.infos['TotalEpisodicConstraint'].append(
            np.cumsum(self.infos['TotalConstraint'])[-1]
        )
        self.infos['DaysAfterPlanting'].append(self._calculate_dap())

    def _calculate_dap(self):
        return int((self.infos['Date'][-1] - self.day_of_planting).days)

    def _action_features_mapper(self):
        act_mapper = {
            'Naction': self.n_action,
            'Nsteps': self.n_steps,
            'StepsSinceLastAction': self.steps_since_last_action,
            'BudgetTotal': self.budget_n,
            'BudgetLeft': self.budget_left,
            'NonZeroActionCount': self.non_zero_action_count,
        }
        return {k: act_mapper[k] for k in self.action_features if k in act_mapper}

    def _soil_features_mapper(self):
        pcas = soil_to_latent_pca(
            self._soil_params,
            "all"
        )
        soil_mapper = {
            'pc1': pcas[0],
            'pc2': pcas[1],
            'pc3': pcas[2],
            'pc4': pcas[3],
            'pc5': pcas[4],
        }
        return {k: soil_mapper[k] for k in self.soil_features if k in soil_mapper}

    def _misc_features_mapper(self, terminated = False):
        pcse_output = self.model.get_output()
        crop_active = self._crop_active_from_pcse_output(pcse_output)

        misc_process = {
            'SinDay': lambda: self._encode_doy(self.date)[0],
            'CosDay': lambda: self._encode_doy(self.date)[1],
            'FertilizerPrice': lambda: self.fertilizer_price,
            'CropPrice': lambda: self.crop_price,
            'CropCode': lambda: self._get_crop_code(),
            'CO2': lambda: self.carbon_dioxide_level,
            'Nue': lambda: (
                calculate_nue(
                    n_input=self.reward_container.actions,
                    n_so=process_pcse.get_n_storage_organ(pcse_output),
                    nh4_depo=get_nh4_deposition_pcse(pcse_output),
                    no3_depo=get_no3_deposition_pcse(pcse_output),
                    crop_name=self.crop,
                ) if crop_active else np.nan
            ),
            'Nsurp': lambda: (
                get_surplus_n(
                    n_input=self.reward_container.get_total_fertilization,
                    n_so=process_pcse.get_n_storage_organ(pcse_output),
                    nh4_depo=get_nh4_deposition_pcse(pcse_output),
                    no3_depo=get_no3_deposition_pcse(pcse_output),
                    crop_name=self.crop,
                ) if crop_active else np.nan
            ),
            'area': lambda: self.area,
        }

        return {k: misc_process[k]() for k in self.misc_features if k in misc_process}

    @staticmethod
    def _crop_active_from_pcse_output(pcse_output) -> bool:
        if not pcse_output:
            return False
        dvs = pcse_output[-1].get("DVS", None)
        try:
            return dvs is not None and np.isfinite(float(dvs))
        except Exception:
            return False

    '''
    Randomizers. NOTE: The weather randomizer is under utils/domain_randomizer.py
    '''

    def _randomise_domain(self, options):
        if self.training:
            # reuse cached spec if we still have repeats left
            if self.domain_repeat > 1 and self._domain_repeat_left > 0 and self._domain_spec is not None:
                options = self._apply_domain_spec(self._domain_spec, options)
                self._domain_repeat_left -= 1
                return options

            # else: sample a fresh domain and cache it
            spec = self._sample_domain_spec(options)
            self._domain_spec = spec
            self._domain_repeat_left = max(0, self.domain_repeat - 1)
            print(f"Sampled new domain for year {self.year}")
        return options

    def _randomise_area(self):
        self.area = self.rng.uniform(low=0.0, high=20.0)

    def _perturb_parameters(self):
        # get and filter relevant crop params
        crop_params = {key: val for key, val in self._parameter_provider._cropdata.items()
                       if key in self.CROP_PARAMS and isinstance(val, float)}

        for key, val in crop_params.items():
            # perturb by 2 percent
            self._parameter_provider.set_override(key, val*self.rng.normal(1.0, 0.02), check=False)

    def _perturb_carbon_dioxide(self, co2):
        return co2 * self.rng.normal(1.0, 0.1)

    def _shift_sowing_date(self, shift_days: int = None):
        # shift sowing date by normal randomiser with std of 5
        if shift_days is None:
            shift_days = int(round(self.rng.normal(loc=0.0, scale=5.0)))
        shifted_date = self.agmt.crop_start_date + datetime.timedelta(days=shift_days)
        # shift sowing date and also campaign date proportionally by the random sowing
        end_shift = {"crop_end_date": self.agmt.crop_end_date + datetime.timedelta(days=shift_days)}\
            if self.agmt.crop_end_type == "harvest" else {}
        self.agro_management = self.agmt.update_attributes(crop_start_date=shifted_date,
                                                            campaign_date=shifted_date - datetime.timedelta(weeks=8),
                                                           **end_shift)
        return shift_days

    def _sample_domain_spec(self, options: dict) -> dict:
        spec = {}

        # sowing date
        if self.random_manager.sowing:
            shift = self._shift_sowing_date()  # returns the shift used
            spec['sow_shift_days'] = shift
        else:
            spec['sow_shift_days'] = None

        # CO2
        if self.random_manager.co2:
            co2_val = self._perturb_carbon_dioxide(self._get_carbon_dioxide_levels())
            options.setdefault('site_params', {})
            options['site_params']['CO2'] = co2_val
            spec['co2'] = co2_val
        else:
            spec['co2'] = None

        # weather flag
        if self.random_manager.weather:
            spec['weather'] = bool(self.random_manager.weather)
            spec['weather_seed'] = int(self.rng.integers(0, 2 ** 31 - 1))
        else:
            spec['weather'] = None

        # area
        if self.random_manager.area:
            self._randomise_area()
            spec['area'] = float(self.area)
        else:
            spec['area'] = None

        # soil
        if self.random_manager.soil:
            spec['soil_params'] = None
            coor = self.rng.choice(self.soil_coords)
            soil_fname = f"soil_{coor[0]}_{coor[1]}.yaml"
            with open(os.path.join(_SOILGRIDS_PATH, soil_fname), 'r') as f:
                options['soil_params'] = yaml.safe_load(f)
            spec['soil_file'] = soil_fname
        else:
            spec['soil_file'] = None
            spec['soil_params'] = None

        fert_choices = list(self.fertilizer_prices.values())
        crop_choices = list(self.crop_prices[self.crop].values())
        spec['fertilizer_price'] = float(self.rng.choice(fert_choices))
        spec['crop_price'] = float(self.rng.choice(crop_choices))

        return spec

    def _apply_domain_spec(self, spec: dict, options: dict) -> dict:
        # sowing shift (apply relative to the already restored original_agmt)
        if spec.get('sow_shift_days') is not None:
            shift = int(spec['sow_shift_days'])
            _ = self._shift_sowing_date(shift)

        # area
        if spec.get('area') is not None:
            self.area = float(spec['area'])

        # CO2
        if spec.get('co2') is not None:
            options.setdefault('site_params', {})
            options['site_params']['CO2'] = spec['co2']

        # soil
        if spec.get('soil_file'):
            with open(os.path.join(_SOILGRIDS_PATH, spec['soil_file']), 'r') as f:
                options['soil_params'] = yaml.safe_load(f)
        elif spec.get('soil_params') is not None:
            options['soil_params'] = spec['soil_params']

        # weather randomization flag
        if spec.get('weather') is not None:
            options['weather'] = True

        return options

    '''
    Init helpers
    '''

    def _init_configs(self):
        crop_parameters = YAMLCropDataProvider(fpath=_CROPS_PATH, force_reload=False)

        with open(os.path.join(_SOILGRIDS_PATH, f'soil_{self.location[1]}_{self.location[0]}.yaml'), 'r') as f:
            soil_parameters = yaml.safe_load(f)

        self.carbon_dioxide_level = self._get_carbon_dioxide_levels()

        # initialize soil variables
        self._init_soil_variables(
            len_soil=len(soil_parameters['SoilProfileDescription']['SoilLayers'])
        )

        nh4i, no3i = self._generate_realistic_n(
            random=False,
        )

        site_parameters = WOFOST81SiteDataProvider_SNOMIN(
            WAV=40,
            CO2=self.carbon_dioxide_level,
            # default init; need to change?
            NH4I=nh4i,
            NO3I=no3i,
            NH4ConcR=1.32,
            NO3ConcR=0.6,
        )
        return crop_parameters, site_parameters, soil_parameters

    def _init_reward_function(self, costs_nitrogen, kwargs):

        self.rewards_obj: Rewards = Rewards(kwargs.get('reward_var'), self.timestep, costs_nitrogen)
        self.reward_container: ActionsContainer | Rewards.ContainerEND | Rewards.ContainerANE = ActionsContainer()
        self.reward_class: Rewards = None

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

        elif self.reward_function == "NSU":
            self.reward_class = self.rewards_obj.NSU(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)

        elif self.reward_function == 'PNB':
            self.reward_class = self.rewards_obj.PNB(self.timestep, costs_nitrogen, budget_left=self.budget_left)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)
            self.rewards_obj.crop_price = self.crop_price
            self.rewards_obj.fertilizer_price = self.fertilizer_price

        elif self.reward_function == 'PNR':
            self.reward_class = self.rewards_obj.PNR(self.timestep, costs_nitrogen, budget_left=self.budget_left)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)
            self.rewards_obj.crop_price = self.crop_price
            self.rewards_obj.fertilizer_price = self.fertilizer_price

        elif self.reward_function == 'PNY':
            self.reward_class = self.rewards_obj.PNY(self.timestep, costs_nitrogen)
            self.reward_container = self.rewards_obj.ContainerNUE(self.timestep, costs_nitrogen)
            self.rewards_obj.crop_price = self.crop_price
            self.rewards_obj.fertilizer_price = self.fertilizer_price

        elif self.reward_function == 'MPN':
            self.reward_class = self.rewards_obj.MPN(self.timestep, costs_nitrogen, budget_left=self.budget_left)
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
        self.action = 0
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
        self._env_baseline = None
        self.zero_nitrogen_env_storage = None
        self.baseline_information = None
        if self.reward_function in reward_functions_with_baseline() and self.original is not False:
            self._env_baseline = ParcelEnv(
                crop_features=self.crop_features,
                weather_features=self.weather_features,
                action_features=self.action_features,
                location=self.location,
                year=self.year,
                reward=self.reward_function,
                training=self.training,
                crop=self.crop,
                name=self.name,
                seed=self.seed,
                original=False,  # important!
                flatten_obs=True,
                type=self.soil_type,
                domain_repeat=self.domain_repeat,
                area=self.area,
                **kwargs,
            )
            self.zero_nitrogen_env_storage = ZeroNitrogenEnvStorage()
            self.baseline_information = {}

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
    def domain_repeat_left(self):
        return self._domain_repeat_left

    @domain_repeat_left.setter
    def domain_repeat_left(self, value):
        self._domain_repeat_left = value

    @property
    def domain_spec(self):
        return self._domain_spec

    @domain_spec.setter
    def domain_spec(self, spec):
        self._domain_spec = spec

    @domain_spec.setter
    def domain_spec(self, spec):
        self._domain_spec = spec

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

    def __init__(self, maxlen=50):
        self.results = OrderedDict()
        self.results_output = OrderedDict()
        self.maxlen = maxlen

    @staticmethod
    def run_episode(env, spec=None):
        if spec is not None:
            env.domain_spec = spec
            env.domain_repeat_left = 10
        env.reset(options={'year': env.year})
        terminated, truncated = False, False
        info = {}
        while not terminated or truncated:
            _, _, terminated, truncated, info = env.step(0)
        return info

    @staticmethod
    def get_key(env):
        year = env.year
        location = env.location
        crop = env.crop
        reg = get_scenario_based_on_loc(env.location)
        key = f'{year}-{location}-{crop}-{reg}'
        assert 'None' not in key
        return key

    def get_episode_output_robust(self, env, spec=None):
        result = self.run_episode(env, spec=spec)
        return result

    def get_episode_output(self, env, spec=None):
        key = self.get_key(env)
        if spec is not None and getattr(env, "domain_spec", None) is not None:
            if env.domain_spec == spec and key in self.results:
                return self.results[key], self.results_output[key]
        if key not in self.results.keys():
            results = self.run_episode(env, spec=spec)
            self.results[key] = results
            self.results_output[key] = env.model.get_output()
            if len(self.results) > self.maxlen:
                self.results.popitem(last=False)
                self.results_output.popitem(last=False)
        assert bool(self.results[key]), "key empty; check PCSE output"
        return self.results[key], self.results_output[key]

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