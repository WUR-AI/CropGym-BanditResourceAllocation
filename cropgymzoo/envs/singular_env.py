import os
import gymnasium as gym

import numpy as np
import datetime

from cropgymzoo import _WOFOST_CONFIG, _AGRO_CALENDAR_CONFIG, _CROPS_LIST, _SOIL_PATH, _SITE_PATH

import cropgymzoo.utils.process_pcse_output as process_pcse
from cropgymzoo.utils.rewards import Rewards, ActionsContainer, reward_functions_with_baseline, reward_functions_end, calculate_nue
from cropgymzoo.utils.nitrogen_helpers import get_surplus_n, get_nh4_deposition_pcse, get_no3_deposition_pcse, convert_year_to_n_concentration
import cropgymzoo.envs.pcse_env as pcse_env

class ParcelEnv(pcse_env.PCSEEnv):
    """

    """

    def __init__(self,
                 crop_features,
                 weather_features,
                 action_features,
                 locations = None,
                 years = None,
                 timestep = 7,
                 reward = 'NUE',
                 action_multiplier = 1,
                 action_space = gym.spaces.Discrete(9),
                 costs_nitrogen = 2,
                 crop: str = 'winterwheat',
                 model_config: str = _WOFOST_CONFIG,
                 agro_config: str = _AGRO_CALENDAR_CONFIG,
                 site_path: str = _SITE_PATH,
                 soil_path: str = _SOIL_PATH,
                 seed = 107,
                 **kwargs,
    ):
        self.crop = crop
        self.crop_features = crop_features
        self.weather_features = weather_features
        self.action_features = action_features
        self.years = [years] if isinstance(years, int) else years
        self.locations = [locations] if isinstance(locations, tuple) else locations

        # TODO change agro config here
        # TODO get parameters from yaml

        crop_parameters, site_parameters, soil_parameters = self._init_configs(crop)

        super(ParcelEnv, self).__init__(
            model_config=model_config,
            agro_config=agro_config,
            crop_parameters=crop_parameters,
            site_parameters=site_parameters,
            soil_parameters=soil_parameters,
            locations=locations,)
        self.costs_nitrogen = costs_nitrogen
        self.action_multiplier = action_multiplier
        self.action_space = action_space
        self._timestep = timestep
        self.reward_function = reward

        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)

        self._init_action_variables()

        self._init_soil_variables()

        """ Initialize reward function """

        self._init_reward_function(costs_nitrogen, kwargs)

        super().reset(seed=seed)

    def _init_reward_function(self, costs_nitrogen, kwargs):

        self.rewards_obj = Rewards(kwargs.get('reward_var'), self.timestep, costs_nitrogen)
        self.reward_container = ActionsContainer()

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

    def _init_soil_variables(self):
        """ Get number of soil layers if using WOFOST snomin"""
        self.mean_total_N = None
        self.std_dev_total_N = None
        self.percentage_NO3 = None
        self.percentage_NH4 = None
        self.top_30cm_fraction = None
        self.bottom_70cm_fraction = None
        self.len_soil_layers = None
        self.len_top_layers = None

        if self.pcse_env == 2:
            self.len_soil_layers = self.get_len_soil_layers
            self.init_random_init_conditions_params()

    def _init_action_variables(self):
        """Masking variables"""
        self.n_steps = 0
        # Non Zero constraint
        self.non_zero_action_count = 0
        self.max_non_zero_actions = 4
        # Consecutive constraint
        self.mask_duration = 3
        self.consecutive_mask_counter = 0

    def _get_observation_space(self):
        nvars = self._get_obs_len()
        return gym.spaces.Box(-10, np.inf, shape=(nvars,))

    def _get_obs_len(self):
        if self.sb3_env.no_weather:
            nvars = len(self.crop_features)
        else:
            nvars = len(self.crop_features) + len(self.action_features) + len(self.weather_features) * self.timestep
        if self.mask_binary:  # TODO: test with weather features
            nvars = nvars + len(self.po_features)
        return nvars

    def step(self, action):
        """
        Computes customized reward and populates info
        """

        # advance one step of the PCSEEngine wrapper and apply action(s)
        obs, _, terminated, truncated, info = self.sb3_env.step(action)

        # grab output of PCSE
        output = self.sb3_env.model.get_output()

        # process output to get observation, reward and growth of winterwheat
        obs, reward, growth = self.process_output(action, output, obs, terminated)

        # normalize observations and reward if not using VecNormalize wrapper
        if self.normalize:
            measure = None
            if isinstance(action, np.ndarray):
                measure = action[1:]
            obs = self.norm.normalize_measure_obs(obs, measure)
            self.norm.update_running_rew(reward)
            reward = self.norm.normalize_reward(reward)

        info = self.grab_infos(output, info, reward, growth)

        return obs, reward, terminated, truncated, info

    def process_output(self, action, output, obs, terminated):

        if self.po_features and isinstance(action, np.ndarray) and action.dtype != np.float32:
            measure = None
            if isinstance(action, np.ndarray):
                action, measure = action[0], action[1:]
            amount = action * self.action_multiplier
            reward, growth = self.get_reward_and_growth(output, amount, terminated)
            obs, cost = self.measure_features.measure_act(obs, measure)
            measurement_cost = sum(cost)
            # if self.reward_function in reward_functions_end() and self.reward_function == 'NUE':
            #     self.rewards.calc_misc_cost(self.reward_container, measurement_cost)
            reward -= measurement_cost
            return obs, reward, growth
        else:
            if isinstance(action, np.ndarray):
                action = action.item()
            amount = action * self.action_multiplier
            reward, growth = self.get_reward_and_growth(output, amount, terminated)
            return obs, reward, growth

    def get_reward_and_growth(self, output, amount, terminated):
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

        reward, growth = self.reward_class.return_reward(output, amount,
                                                         output_baseline=output_baseline,
                                                         multiplier=self.sb3_env.multiplier_amount,
                                                         obj=self.reward_container)
        self.rewards_obj.update_profit(output, amount, year=self.sb3_env.date.year,
                                       multiplier=self.sb3_env.multiplier_amount)
        reward += self.terminate_reward_signal(output, reward, terminated)
        return reward, growth

    def terminate_reward_signal(self, output, reward, terminated):
        if terminated and self.reward_function in reward_functions_end():
            reward = self.reward_container.dump_cumulative_positive_reward - abs(reward)

        elif terminated and self.reward_function == 'HAR':
            reward = self.yield_modifier * self.reward_container.dump_cumulative_positive_reward - abs(reward)

        elif terminated and self.reward_function in ['NUE', 'DNE']:
            reward = (self.reward_container.calculate_reward_nue(
                n_fertilized=self.reward_container.get_total_fertilization * 10,
                n_output=process_pcse.get_n_storage_organ(output),
                no3_depo=get_no3_deposition_pcse(output),
                nh4_depo=get_nh4_deposition_pcse(output),)
            )
        return reward

    def grab_infos(self, output, info, reward, growth):
        # fill in infos
        if 'reward' not in info.keys(): info['reward'] = {}
        info['reward'][self.date] = reward
        if 'growth' not in info.keys(): info['growth'] = {}
        info['growth'][self.date] = growth

        if self.pcse_env == 2:
            if 'NUE' not in info.keys():
                info['NUE'] = {}
            info['NUE'][self.date] = self.rewards_obj.calculate_nue_on_terminate(
                n_input=self.reward_container.get_total_fertilization * 10,
                n_so=process_pcse.get_n_storage_organ(output),
                year=self.date.year,
                no3_depo=get_no3_deposition_pcse(output),
                nh4_depo=get_nh4_deposition_pcse(output),)
            if 'Nsurplus' not in info.keys():
                info['Nsurplus'] = {}
            info['Nsurplus'][self.date] = get_surplus_n(self.reward_container.get_total_fertilization * 10,
                                                        n_so=process_pcse.get_n_storage_organ(output),
                                                        year=self.date.year,
                                                        no3_depo=get_no3_deposition_pcse(output),
                                                        nh4_depo=get_nh4_deposition_pcse(output),)

        if 'profit' not in info.keys():
            info['profit'] = {}
        info['profit'][self.date] = self.rewards_obj.profit

        # save info of random initial conditions
        # if terminated and self.random_init:
        #     if 'init_n' not in info.keys():
        #         info['init_n'] = {}
        #     info['init_n']['no3'] = self.eval_no3i
        #     info['init_n']['nh4'] = self.eval_nh4i

        return info

    def overwrite_year(self, year):
        self.years = year
        if self.reward_function in reward_functions_with_baseline():
            self.baseline_env.agro_management = self.sb3_env.agmt.replace_years(year)
        self.sb3_env.agro_management = self.sb3_env.agmt.replace_years(year)

    def set_location(self, location):
        if self.reward_function in reward_functions_with_baseline():
            self.baseline_env.loc = location
            self.baseline_env.weather_data_provider = (
                pcse_env.get_weather_data_provider(location, random_weather=self.random_weather))
        self.sb3_env.loc = location
        self.sb3_env.weather_data_provider = (
            pcse_env.get_weather_data_provider(location, random_weather=self.random_weather))

    def overwrite_location(self, location):
        self.locations = location
        self.set_location(location)

    def overwrite_initial_conditions(self):
        # N initial conditions
        list_nh4i, list_no3i = self.generate_realistic_n()
        self.eval_nh4i = list_nh4i
        self.eval_no3i = list_no3i

        site_parameters = {'NH4I': list_nh4i, 'NO3I': list_no3i, }
        return site_parameters

    def overwrite_nitrogen_rain_concentration(self):
        # N concentration in rain for deposition
        nh4concr, no3concr = convert_year_to_n_concentration(self.sb3_env.agmt.crop_end_date.year,
                                                             agmt=self.sb3_env.agmt,
                                                             random_weather=self.random_weather,
                                                             loc=self.loc)

        site_parameters = {'NH4ConcR': nh4concr, 'NO3ConcR': no3concr, }
        return site_parameters

    def init_random_init_conditions_params(self):
        self.mean_total_N = 50  # kg/ha
        self.std_dev_total_N = 35  # kg/ha
        self.percentage_NO3 = 0.85
        self.percentage_NH4 = 0.15
        self.top_30cm_fraction = 0.7
        self.bottom_70cm_fraction = 0.3
        # Sanity check
        # If soil profile is 1m, 30% will have 70% of the total inorganic N
        self.len_top_layers = int(np.ceil(self.len_soil_layers * 0.3))

    def generate_realistic_n(self) -> tuple[list, list]:
        """ method to overwrite a random N initial condition for every call of reset()
            Implemented based on discussions with Herman Berghuijs, for NL conditions
        """

        '''Comments for sanity check'''
        # Generate total inorganic N from seeded normal distribution and clip so that no outliers become negative
        total_inorganic_n = self.rng.normal(self.mean_total_N, self.std_dev_total_N)
        total_inorganic_n = np.clip(total_inorganic_n, 0, 100)

        # Split total inorganic N into NO3 and NH4
        total_no3 = total_inorganic_n * self.percentage_NO3
        total_nh4 = total_inorganic_n * self.percentage_NH4

        # Distribute 70% of the total inorganic N in the upper 30 cm and 30% in the lower 70 cm
        no3_top = total_no3 * self.top_30cm_fraction
        no3_bottom = total_no3 * self.bottom_70cm_fraction
        nh4_top = total_nh4 * self.top_30cm_fraction
        nh4_bottom = total_nh4 * self.bottom_70cm_fraction

        # Create lists of per layer N content
        no3_distribution = np.zeros(self.len_soil_layers)
        nh4_distribution = np.zeros(self.len_soil_layers)

        # Considering 1m soil profile
        # Upper 30 cm distribution (first layers), multiply list of dirichlet with fraction of total in topsoil layers
        no3_distribution[:self.len_top_layers] = self.rng.dirichlet(np.ones(self.len_top_layers), size=1) * no3_top
        nh4_distribution[:self.len_top_layers] = self.rng.dirichlet(np.ones(self.len_top_layers), size=1) * nh4_top

        # Lower 70 cm distribution (last layers), same for remaining bottom layers
        no3_distribution[self.len_top_layers:] = self.rng.dirichlet(np.ones(self.len_soil_layers - self.len_top_layers),
                                                                    size=1) * no3_bottom
        nh4_distribution[self.len_top_layers:] = self.rng.dirichlet(np.ones(self.len_soil_layers - self.len_top_layers),
                                                                    size=1) * nh4_bottom

        # Ensure no negative values in the distributions, might skew the distribution by a teeny bit
        list_nh4i = list(np.maximum(nh4_distribution, 0))
        list_no3i = list(np.maximum(no3_distribution, 0))

        return list_nh4i, list_no3i

    def special_init_conditions(self):
        site_params = None
        if self.random_init and self.pcse_env == 2:
            site_params = self.overwrite_initial_conditions()
            # for N deposition
            site_params = site_params | self.overwrite_nitrogen_rain_concentration()
        elif not self.random_init and self.pcse_env == 2:
            site_params = self.overwrite_nitrogen_rain_concentration()

        return site_params

    def reset(self, seed=None, options=None, **kwargs):

        # Only for invalid action masking
        self.reset_non_zero_action_count()

        site_params = self.special_init_conditions()

        if isinstance(options, dict):
            site_params = self.special_init_conditions() | options

        if isinstance(self.years, list):
            year = self.np_random.choice(self.years)
            if self.reward_function in reward_functions_with_baseline():
                self.baseline_env.agro_management = self.sb3_env.agmt.replace_years(year)
            self.sb3_env.agro_management = self.sb3_env.agmt.replace_years(year)

        if isinstance(self.locations, list):
            location = self.locations[self.np_random.choice(len(self.locations), 1)[0]]
            self.set_location(location)

        self.reward_container.reset()
        self.rewards_obj.reset()

        if self.reward_function in reward_functions_with_baseline():
            self.baseline_env.reset(seed=seed, options=site_params)
        obs = self.sb3_env.reset(seed=seed, options=site_params)

        # TODO: check whether info should/could be filled
        info = {}

        if self.normalize:
            obs = self.norm.normalize_measure_obs(obs, None)

        return obs, info

    def action_masks(self):
        assert isinstance(self.action_space, gym.spaces.Discrete)
        if (self.non_zero_action_count >= self.max_non_zero_actions or
            self.consecutive_mask_counter > 0 or
            self.n_steps < self.start_actions or
            self.n_steps >= self.end_actions):
            mask = [False for _ in range(self.action_space.n)]
            mask[0] = True
            return mask
        else:
            return [True for _ in range(self.action_space.n)]

    def reset_non_zero_action_count(self):
        self.non_zero_action_count = 0
        self.consecutive_mask_counter = 0
        self.n_steps = 0

    def update_non_zero_action_count(self, actions):
        self.n_steps += 1
        if np.any(actions != 0):
            self.non_zero_action_count += np.sum(actions != 0).item()
            self.consecutive_mask_counter = self.mask_duration
        elif self.consecutive_mask_counter > 0:
            self.consecutive_mask_counter -= 1

    def _init_configs(self):
        crop = self.crop

        crop_parameters = ...
        site_parameters = ...
        soil_parameters = ...

        return crop_parameters, site_parameters, soil_parameters

    def render(self, mode="human"):
        pass

    @property
    def get_len_soil_layers(self):
        return len(self.sb3_env.model.kiosk.SM)

    @property
    def model(self):
        return self.sb3_env.model

    @model.setter
    def model(self, model):
        self.sb3_env.model = model

    @property
    def norm(self):
        return self._norm

    @property
    def norm_rew(self):
        return self._rew_norm

    @property
    def sb3_env(self):
        return self._env

    @property
    def baseline_env(self):
        return self._env_baseline

    @property
    def date(self) -> datetime.date:
        return self.sb3_env.model.day

    @property
    def loc(self) -> tuple:
        return self.sb3_env.loc

    @property
    def timestep(self):
        return self._timestep

    @property
    def obs_len(self):
        return self._get_obs_len()

    @property
    def act_len(self):
        return len(self.action_space.shape)