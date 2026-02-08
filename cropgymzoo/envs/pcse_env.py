import datetime
import os
import copy
import functools

import numpy as np
import yaml

import gymnasium as gym

import pcse

"""
    Gymnasium Environment built around the PCSE library for crop simulation
    Gym:  https://github.com/Farama-Foundation/Gymnasium
    PCSE: https://github.com/ajwdewit/pcse

    Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
    and the extended cropgym.ai built by Michiel Kallenberg and Ron van Bree (https://github.com/WUR-AI/PCSE-Gym)
    With additional augments from Hilmy Baja
"""


class AgroManagementContainer:
    _ALLOWED_FIELDS = {
        "crop_name",
        "variety_name",
        "crop_start_date",
        "crop_start_type",
        "crop_end_date",
        "crop_end_type",
        "max_duration",
        "campaign_date",
    }

    _DATES_FIELDS = {
        "crop_start_date",
        "crop_end_date",
        "campaign_date",
    }

    def __init__(self, agro_management: list, crop: str = None):
        self.agro_structure = agro_management
        self.campaign_date: datetime.date = list(agro_management[0].keys())[0]
        self.crop_name: str = crop if crop is not None else agro_management[0][self.campaign_date]['CropCalendar']['crop_name']
        self.variety_name: str = agro_management[0][self.campaign_date]['CropCalendar']['variety_name']
        self.crop_start_date: datetime.date = agro_management[0][self.campaign_date]['CropCalendar']['crop_start_date']
        self.crop_start_type: str = agro_management[0][self.campaign_date]['CropCalendar']['crop_start_type']
        self.crop_end_date: datetime.date = agro_management[0][self.campaign_date]['CropCalendar']['crop_end_date']
        self.crop_end_type: str = agro_management[0][self.campaign_date]['CropCalendar']['crop_end_type']
        try:
            self.max_duration: int | None = agro_management[0][self.campaign_date]['CropCalendar']['max_duration']
        except KeyError:
            self.max_duration = None

        self.structure = None
        self.build_structure()

    def build_structure(self):
        self.structure = yaml.load(f'''
                    - {self.campaign_date}:
                        CropCalendar:
                            crop_name: {self.crop_name}
                            variety_name: {self.variety_name}
                            crop_start_date: {self.crop_start_date}
                            crop_start_type: {self.crop_start_type}
                            crop_end_date: {self._yaml_value(self.crop_end_date)}
                            crop_end_type: {self.crop_end_type}
                            max_duration: {self._yaml_value(self.max_duration)}
                        TimedEvents: null
                        StateEvents: null
                ''', Loader=yaml.SafeLoader)

    @staticmethod
    def build_multi_campaign_structure(campaign_specs: list[dict]) -> list:
        """
        Build PCSE agromanagement structure with multiple daisy-chained campaigns.

        Each spec requires:
          - campaign_date (datetime.date)
          - crop_name (str)
          - variety_name (str)
          - crop_start_date (datetime.date)
          - crop_start_type (str)
          - crop_end_type (str)
          - max_duration (int)
        Optional:
          - crop_end_date (datetime.date or None)
        """
        specs = sorted(campaign_specs, key=lambda s: s["campaign_date"])
        structure = []
        for s in specs:
            cdate = s["campaign_date"]
            structure.append({
                cdate: {
                    "CropCalendar": {
                        "crop_name": s["crop_name"],
                        "variety_name": s.get("variety_name", "") or "",
                        "crop_start_date": s["crop_start_date"],
                        "crop_start_type": s.get("crop_start_type", "sowing"),
                        "crop_end_date": s.get("crop_end_date", None),
                        "crop_end_type": s.get("crop_end_type", "maturity"),
                        "max_duration": int(s.get("max_duration", 365)) if s.get("max_duration") is not None else None,
                    },
                    "TimedEvents": None,
                    "StateEvents": None,
                }
            })
        return structure

    def update_attributes(self, **changes: dict):
        """
        Update one or more attributes and rebuild the agro YAML.

        Examples
        --------
        agmt.update(crop_name="maize", crop_start_date=datetime.date(2025, 4, 15))
        agmt.update(crop_end_date=datetime.date(2026, 8, 30), max_duration=480)
        """
        invalid = [k for k in changes if k not in self._ALLOWED_FIELDS]
        if invalid:
            raise ValueError(f"Unknown field(s): {', '.join(invalid)}")

        for attr, spec in changes.items():
            current = getattr(self, attr)

            # transformer function --> attr = spec(attr)
            if callable(spec):
                new_val = spec(current)

            # dict with kwargs for .replace(**spec)
            elif isinstance(spec, dict):
                if hasattr(current, "replace"):
                    new_val = current.replace(**spec)
                else:
                    raise TypeError(
                        f"{attr} does not support .replace(**kwargs); "
                        f"pass a callable instead."
                    )

            # plain overwrite
            else:
                new_val = spec

            new_val = self.str_to_datetime(new_val) if attr in self._DATES_FIELDS else new_val

            setattr(self, attr, new_val)

        # Do some checks

        if 'crop_start_date' in changes.keys():
            if self.crop_start_date < self.campaign_date:
                self.crop_start_date = self.campaign_date - datetime.timedelta(weeks=4)

        self.build_structure()
        return self.structure

    @staticmethod
    def str_to_datetime(date_str: str | None) -> datetime.date | None | str:
        if isinstance(date_str, datetime.date):
            return date_str
        if date_str is None:
            return ''
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

    # TODO change; made for winterwheat
    def replace_years(self, y):
        """
            Years replaced are the harvest date.
        """
        if isinstance(y, list):
            y = y[0]
        self.campaign_date = self.campaign_date.replace(year=y)
        self.crop_start_date = self.crop_start_date.replace(year=y)
        self.crop_end_date = self.crop_end_date.replace(year=y)

        self.build_structure()
        return self.structure

    def replace_sow_date(self, year, month, day):
        self.crop_start_date = self.crop_start_date.replace(year=year, month=month, day=day)

        self.build_structure()
        return self.structure

    def replace_harvest_date(self, year, month, day):
        self.crop_end_date = self.crop_end_date.replace(year=year, month=month, day=day)

        self.build_structure()
        return self.structure

    def replace_start_type(self, start):
        assert start == 'sowing' or start == 'emergence'
        self.crop_start_type = start

        self.build_structure()
        return self.structure

    def replace_variety_name(self, name='Julius'):
        self.variety_name = name

        self.build_structure()
        return self.structure

    def replace_crop_name(self, name='winterwheat'):
        self.crop_name = name

        self.build_structure()
        return self.structure

    def start_sowing(self):
        if self.campaign_date.year == self.crop_end_date.year:
            self.campaign_date = datetime.date(self.crop_end_date.year - 1, 10, 1)
            self.crop_start_date = datetime.date(self.crop_end_date.year - 1, 10, 1)

        self.build_structure()

    def start_emergence(self):
        self.campaign_date = datetime.date(self.crop_end_date.year, 1, 1)
        self.crop_start_date = datetime.date(self.crop_end_date.year, 1, 1)

        self.build_structure()

    def get_start_type(self, start_type):
        self.start_emergence() if start_type == 'emergence' else self.start_sowing()

    @property
    def get_structure(self):
        return self.structure

    @property
    def get_start_date(self):
        return self.crop_start_date

    @property
    def get_end_date(self):
        return self.crop_end_date

    @property
    def get_campaign_date(self):
        return self.campaign_date

    @staticmethod
    def _yaml_value(val):
        return 'null' if val is None else val


def get_weather_data_provider(location: tuple,
                              wpo: str='openmeteo',
                              random_weather: bool =False,
                              seed = None) ->(
        pcse.input.NASAPowerWeatherDataProvider or pcse.input.OpenMeteoWeatherDataProvider or pcse.fileinput.CSVWeatherDataProvider):
    if random_weather:
        wdp = get_random_weather_provider(location)
    else:
        if wpo == 'openmeteo':
            wdp = get_openmeteo_provider(location, seed=seed)
        elif wpo == 'nasapower':
            wdp = get_nasapower_provider(location)
        else:
            wdp = get_openmeteo_provider(location, seed=seed)
    return wdp


@functools.cache
def get_excel_provider(file_dir: str, location):
    return pcse.input.ExcelWeatherDataProvider(os.path.join(file_dir, f'{location[0]}-{location[1]}.xlsx'))


def get_nasapower_provider(location):
    return pcse.input.NASAPowerWeatherDataProvider(*location)


def get_openmeteo_provider(location, seed=None, training=False, random_manager=None):
    from cropgymzoo.utils.domain_randomizer import NoisyOpenMeteo
    from cropgymzoo import _BASE_PATH
    api_key = None
    if os.path.exists(os.path.join(_BASE_PATH, "openmeteo_api")):
        with open(os.path.join(_BASE_PATH, "openmeteo_api", "api"), "r") as f:
            api_key = f.readline()
    if training and random_manager is not None:
        return NoisyOpenMeteo(*location, seed=seed, api_key=api_key) if random_manager.weather is True else pcse.input.OpenMeteoWeatherDataProvider(*location, api_key=api_key,)
    else:
        return pcse.input.OpenMeteoWeatherDataProvider(*location, api_key=api_key)


@functools.cache
def get_random_weather_provider(location) -> pcse.input.CSVWeatherDataProvider:
    path_to_file = os.path.dirname(os.path.realpath(__file__))
    lat, lon = location
    if '.' not in str(lat):
        lat = str(lat) + '.0'
    if '.' not in str(lon):
        lon = str(lon) + '.0'
    csv_name = f'{lat}-{lon}_random_weather.csv'
    filename = os.path.join(path_to_file[:-4], 'utils', 'weather_utils', 'random_weather_csv', csv_name)
    wdp = pcse.input.CSVWeatherDataProvider(filename)
    return wdp


class Engine(pcse.engine.Engine):
    """
    Wraps around the PCSE engine/crop model for correct rate updates after fertilization action and
    to set a flag when the simulation has terminated
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flag_terminated = False
        self._flag_irrigate = False

    def _run(self, action):
        """Make one time step of the simulation.
        """

        # Update timer
        self.day, delt = self.timer()

        # State integration
        self.integrate(self.day, delt)

        # Driving variables
        self.drv = self._get_driving_variables(self.day)

        # Agromanagement decisions
        self.agromanager(self.day, self.drv)

        # Do actions
        if action > 0:
            self._send_signal(signal=pcse.signals.apply_n_snomin,
                              amount=action,
                              application_depth=10.,
                              cnratio=0.,
                              f_orgmat=0.,
                              f_NH4N=0.5,
                              f_NO3N=0.5,
                              initial_age=0,
                              )
            self._send_signal(signal=pcse.signals.apply_n,
                              amount=action,
                              recovery=0.7,
                              N_amount=action,
                              N_recovery=0.7
                              )
        if self._flag_irrigate:
            self._send_signal(signal=pcse.signals.irrigate,
                              amount=20,
                              efficiency=0.7,
                              )
            self._flag_irrigate = False

        # Rate calculation
        self.calc_rates(self.day, self.drv)

        if self.flag_terminate is True:
            self._terminate_simulation(self.day)

    def run(self, days=1, action=0):
        """Advances the system state with given number of days"""

        # do action at end of time step
        days_counter = days
        days_done = 0
        while (days_done < days) and (self.flag_terminate is False):
            days_done += 1
            days_counter -= 1
            if days_counter > 0:
                self._run(0)
            else:
                self._run(action)

    @property
    def terminated(self):
        return self._flag_terminated

    @property
    def wdp(self):
        return self.weatherdataprovider

    def _terminate_simulation(self, day):
        super()._terminate_simulation(day)
        self._flag_terminated = True


class PCSEEnv(gym.Env):
    """
    Create a new PCSE-Gym environment

    :param model_config: PCSE config file name (must be available in the pcse/conf/ folder inside the pcse library)
    :param agro_config: file name of the yaml file specifying the agro-management configuration
    :param crop_parameters: Can be specified in two ways:
                                - A path to the crop parameter file
                                  Will be read by a `pcse.fileinput.PCSEFileReader`
                                - An object that is directly passed to the `pcse.base.ParameterProvider`
    :param site_parameters: Can be specified in two ways:
                                - A path to the site parameter file
                                  Will be read by a `pcse.fileinput.PCSEFileReader`
                                - An object that is directly passed to the `pcse.base.ParameterProvider`
    :param soil_parameters: Can be specified in two ways:
                                - A path to the soil parameter file
                                  Will be read by a `pcse.fileinput.PCSEFileReader`
                                - An object that is directly passed to the `pcse.base.ParameterProvider`
    :param years: A single year, or list of years to get weather data for. If not set use year from agro_config
    :param location: latitude, longitude to get weather data for
    :param seed: A seed for the random number generators used in PCSE-Gym
    :param timestep: Number of days that are simulated during a single time step
    """

    _PATH_TO_FILE = os.path.dirname(os.path.realpath(__file__))
    _PATH_TO_ENVS = os.path.dirname(_PATH_TO_FILE)
    _PATH_TO_SOURCE = os.path.dirname(_PATH_TO_ENVS)
    _CONFIG_PATH = os.path.join(_PATH_TO_SOURCE, 'configs')

    _DEFAULT_AGRO_FILE = 'generic_cropcalendar.yaml'
    _DEFAULT_CROP_FILE = 'winterwheat.yaml'
    _DEFAULT_SITE_FILE = 'default_site.yaml'
    _DEFAULT_SOIL_FILE = 'EC3-mediumfine_soil.yaml'

    _DEFAULT_AGRO_FILE_PATH = os.path.join(_CONFIG_PATH, 'agro', _DEFAULT_AGRO_FILE)
    _DEFAULT_CROP_FILE_PATH = os.path.join(_CONFIG_PATH, 'crop', _DEFAULT_CROP_FILE)
    _DEFAULT_SITE_FILE_PATH = os.path.join(_CONFIG_PATH, 'site', _DEFAULT_SITE_FILE)
    _DEFAULT_SOIL_FILE_PATH = os.path.join(_CONFIG_PATH, 'soil', _DEFAULT_SOIL_FILE)

    _DEFAULT_CONFIG = 'Wofost81_NWLP_MLWB_SNOMIN.conf'

    def __init__(self,
                 model_config: str = _DEFAULT_CONFIG,
                 agro_config: str = _DEFAULT_AGRO_FILE_PATH,
                 crop_parameters=_DEFAULT_CROP_FILE_PATH,
                 site_parameters=_DEFAULT_SITE_FILE_PATH,
                 soil_parameters=_DEFAULT_SOIL_FILE_PATH,
                 years=None,
                 location=None,
                 seed: int = None,
                 timestep: int = 7,
                 crop: str = 'winterwheat',
                 crop_info: dict = None,
                 training: bool = False,
                 wait_for_crop: bool = True,
                 **kwargs
                 ):

        assert timestep > 0

        # For skipping campaign date and starting simulation when sowing
        self._wait_for_crop = wait_for_crop
        self.training = training

        # Optionally set the seed
        super().reset(seed=seed)

        # If any parameter files are specified as path, convert them to a suitable object for pcse
        if isinstance(crop_parameters, str):
            crop_parameters = pcse.input.PCSEFileReader(crop_parameters)
        if isinstance(site_parameters, str):
            site_parameters = pcse.input.PCSEFileReader(site_parameters)
        if isinstance(soil_parameters, str):
            soil_parameters = pcse.input.PCSEFileReader(soil_parameters)

        # Set location
        if location is None:
            location = (52.0, 5.5)
        self._location = location
        self._timestep = timestep

        # Store the crop/soil/site parameters
        self._crop_params = crop_parameters
        self._site_params = site_parameters
        self._site_params_ = site_parameters
        self._soil_params = soil_parameters

        # Store the agro-management config
        with open(agro_config, 'r') as f:
            self._agro_management = yaml.load(f, Loader=yaml.SafeLoader)

        # Initialize Agromanagement Container Class
        self.agmt = AgroManagementContainer(self._agro_management, crop)

        # Crop infos
        self.crop = crop
        self.crop_info = crop_info

        if self.crop_info:
            self._agro_management = self.agmt.update_attributes(**self.crop_info[self.crop])

        # Store the PCSE Engine config
        self._model_config = model_config

        # Get the weather data source
        # self._weather_data_provider = get_openmeteo_provider(
        #     self._location,
        #     seed=self.seed,
        #     training=self.training,
        # )

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()

        # Use the config files to extract relevant settings
        model_config = pcse.base.ConfigurationLoader(model_config)
        self._output_variables = model_config.OUTPUT_VARS  # variables given by the PCSE model output
        self._summary_variables = model_config.SUMMARY_OUTPUT_VARS  # Summary variables are given at the end of a run
        self._weather_variables = list(pcse.base.weather.WeatherDataContainer.required)

        # Define action features for observation
        self.action_feature = self._get_action_features_space()
        # Define Gym observation space
        self.observation_space = self._get_observation_space()
        # Define Gym action space
        self.action_space = self._get_action_space()

    def new_wdp(self, options):
        # Reuse an existing weather provider when possible to avoid heavy re-initialization.
        rm = getattr(self, 'random_manager', None)

        # --- what we currently have / want ---
        # current desired "randomized weather" mode (NoisyOpenMeteo) if training+rm.weather
        curr_randomize_flag = bool(self.training and (rm is not None) and getattr(rm, "weather", False))

        # detect whether we must rebuild
        need_new_wdp = not hasattr(self, "_weather_data_provider") or (self._weather_data_provider is None)

        # derive the previous mode if we haven't recorded it yet (first run)
        provider_name = getattr(getattr(self, "_weather_data_provider", None), "__class__", object).__name__ \
            if hasattr(self, "_weather_data_provider") else ""
        prev_randomize_flag = getattr(self, "_wdp_randomize_flag", None)
        if prev_randomize_flag is None and provider_name:
            # heuristic: if class name is NoisyOpenMeteo, we were in randomized mode
            prev_randomize_flag = (provider_name == "NoisyOpenMeteo")

        # triggers for a rebuild
        location_changed = getattr(self, "_wdp_location", None) != self._location
        randomize_mode_changed = (prev_randomize_flag is not None) and (prev_randomize_flag != curr_randomize_flag)
        weather_reseed = bool(options.get('weather')) and ('weather_seed' in options)

        need_new_wdp = need_new_wdp or location_changed or randomize_mode_changed or weather_reseed

        return need_new_wdp, rm, curr_randomize_flag

    def _init_pcse_model(self, options={}, *args, **kwargs) -> Engine:

        if options is None:
            options = {}

        # Inject different initial condition every episode if it specified in args
        if 'site_params' in options:
            if 'NH4I' in options['site_params']:
                self._site_params['NH4I'] = options['site_params']['NH4I']
                self._site_params['NO3I'] = options['site_params']['NO3I']
            if 'NH4ConcR' in options['site_params']:
                self._site_params['NH4ConcR'] = options['site_params']['NH4ConcR']
                self._site_params['NO3ConcR'] = options['site_params']['NO3ConcR']
        if 'soil_params' in options:
            self._soil_params = options['soil_params']
        #
        # if bool(options.get("multi_crop", False)):
        #     crop_dir = os.path.join(self._CONFIG_PATH, "crop")
        #     crop_files = [
        #         os.path.join(crop_dir, fn)
        #         for fn in os.listdir(crop_dir)
        #         if fn.endswith(".yaml") or fn.endswith(".yml")
        #     ]
        #     merged_cropdata = {}
        #     for fp in sorted(crop_files):
        #         reader = pcse.input.PCSEFileReader(fp)
        #         merged_cropdata.update(dict(reader))
        #     if merged_cropdata:
        #         self._crop_params = merged_cropdata

        # Combine the config files in a single PCSE ParameterProvider object
        self._parameter_provider = pcse.base.ParameterProvider(cropdata=self._crop_params,
                                                               sitedata=self._site_params,
                                                               soildata=self._soil_params,
                                                               )

        # if need to reinitialize wdp
        # need_new_wdp, rm, curr_randomize_flag = self.new_wdp(options)
        #
        # if need_new_wdp:
        #     # Use an episode-specific seed only when provided; otherwise keep the env seed.
        #     seed = options.get('weather_seed', self.seed)
        #     self._weather_data_provider = get_openmeteo_provider(
        #         location=self._location,
        #         seed=seed,
        #         training=self.training,
        #         random_manager=rm,
        #     )
        #     # remember state for future resets
        #     self._wdp_location = self._location
        #     self._wdp_randomize_flag = curr_randomize_flag
        # else:
        self._weather_data_provider = get_openmeteo_provider(
            location=self._location,
            seed=self.seed,
            training=self.training,
            random_manager=getattr(self, 'random_manager', None),  # assumed that it's initialised
        )

        # Create a PCSE engine / crop growth model
        model = Engine(self._parameter_provider,
                       self._weather_data_provider,
                       self._agro_management,
                       config=self._model_config,
                       )

        # Let simulation run until crop start date
        if self._wait_for_crop:
            skip_days = max(0, (self.agmt.get_start_date - self.agmt.get_campaign_date).days) - (self._timestep -1)
            if skip_days:
                model.run(days=skip_days, action=0)

        # The model starts with output values for the initial date
        # The initial observation should contain output values for an entire timestep
        # If the timestep > 1, generate the remaining outputs by running the model
        if self._timestep > 1:
            model.run(days=self._timestep - 1)
        return model

    def _get_observation_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict({
            'crop_model': self._get_observation_space_crop_model(),
            'weather': self._get_observation_space_weather(),
            'actions': self._get_action_features_space(),
        })
        return space

    def _get_observation_space_weather(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                'IRRAD': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'TMIN': gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                'TMAX': gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                'VAP': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'RAIN': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'E0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'ES0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'ET0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'WIND': gym.spaces.Box(0, np.inf, (self._timestep,)),
            }
        )

    def _get_action_features_space(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                'action_history': gym.spaces.Box(0, np.inf, (self._timestep,)),
            }
        )

    def _get_observation_space_crop_model(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {var: gym.spaces.Box(-np.inf, np.inf, shape=(self._timestep,)) for var in self._output_variables}
        )

    def _get_action_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict(
            {
                'irrigation': gym.spaces.Box(0, np.inf, shape=()),
                'N': gym.spaces.Box(0, np.inf, shape=()),
            }
        )
        return space

    """
    Properties of the crop model config file
    """

    @property
    def output_variables(self) -> list:
        return list(self._output_variables)

    @property
    def summary_variables(self) -> list:
        return list(self._summary_variables)

    @property
    def weather_variables(self):
        return list(self._weather_variables)

    """
    Properties derived from the agro management config:
    """

    @property
    def _campaigns(self) -> dict:
        return self._agro_management[0]

    @property
    def _first_campaign(self) -> dict:
        return self._campaigns[min(self._campaigns.keys())]

    @property
    def _last_campaign(self) -> dict:
        return self._campaigns[max(self._campaigns.keys())]

    @property
    def start_date(self) -> datetime.date:
        return self._model.agromanager.start_date

    @property
    def end_date(self) -> datetime.date:
        return self._model.agromanager.end_date

    """
    Other properties
    """

    @property
    def date(self) -> datetime.date:
        return self._model.day

    @property
    def wdp(self) -> pcse.input.NASAPowerWeatherDataProvider or pcse.input.OpenMeteoWeatherDataProvider:
        return self._weather_data_provider

    """
    Gym functions
    """

    def step(self, action) -> tuple:
        """
        Perform a single step in the Gym environment. The provided action is performed and the environment transitions
        from state s_t to s_t+1. Based on s_t+1 an observation and reward are generated.

        :param action: an action that respects the action space definition as described by `self._get_action_space()`
        :return: a 4-tuple containing
            - an observation that respects the observation space definition as described by `self._get_observation_space()`
            - a scalar reward
            - a boolean flag indicating whether the environment/simulation has ended
            - a dict containing extra info about the environment and state transition
        """

        # Create a dict for storing info
        info = dict()

        # Apply action
        if isinstance(action, np.ndarray):
            action = action[0] if action.shape else action
        action = self._apply_action(action)  # is subclassed by sb3

        # Run the crop growth model
        self._model.run(days=self._timestep, action=action)
        # Get the model output
        output = self._model.get_output()[-self._timestep:]
        info['days'] = [day['day'] for day in output]

        # Construct an observation and reward from the new environment state
        o = self._get_observation(output)
        r = self._get_reward()
        # Check whether the environment has terminated
        done = self._model.terminated
        if done:
            info['output_history'] = self._model.get_output()
            info['summary_output'] = self._model.get_summary_output()
            info['terminal_output'] = self._model.get_terminal_output()
        truncated = False
        terminated = done
        # Return all values
        return o, r, terminated, truncated, info

    def _apply_action(self, action):

        irrigation = action.get('irrigation', 0)
        N = action.get('N', 0)

        self._model._send_signal(signal=pcse.signals.irrigate,
                                 amount=irrigation,
                                 efficiency=0.8,
                                 )

        self._model._send_signal(signal=pcse.signals.apply_n,
                                 N_amount=N,
                                 N_recovery=0.7,
                                 )

    def _get_observation(self, output) -> dict:
        """
        Generate an observation based on the current environment state

        :param output: the output of the model after the state transition
        :return: an observation. The default implementation returns a dict containing two dicts containing crop model
                 and weather data, respectively
        """

        # Get the datetime objects characterizing the specific days
        days = [day['day'] for day in output]

        # Get the output variables for each of the days
        crop_model_observation = {v: [day[v] for day in output] for v in self._output_variables}

        # Get the weather data of the passed days
        weather_data = [self._weather_data_provider(day) for day in days]
        # Cast the weather data into a dict
        weather_observation = {var: [getattr(weather_data[d], var) for d in range(len(days))] for var in
                               self._weather_variables}
        # Get action history through action features
        action_features = {}
            # {'ActionHistory': [day['RNH4AMTT'] / 1e-3 + day["RNO3AMTT"] / 1e-3 for day in output]}

        o = {
            'crop_model': crop_model_observation,
            'weather': weather_observation,
            'action_features': action_features,
        }

        return o

    def _get_reward(self, var='TWSO') -> float:
        """
        Generate a reward based on the current environment state

        :param var: the variable extracted from the model output
        :return: a scalar reward. The default implementation gives the increase in yield during the last state transition
                 if the environment is in its initial state, the initial yield is returned
        """

        output = self._model.get_output()
        # var = 'LAI'  # For debugging
        # Consider different cases:
        if len(output) == 0:  # The simulation has not started -> 0 reward
            return 0
        if len(output) <= self._timestep:  # Only one observation is made -> give initial yield as reward
            return output[-1][var] or 0
        else:  # Multiple observations are made -> give difference of yield of the last time steps
            last_index_previous_state = (np.ceil(len(output) / self._timestep).astype('int') - 1) * self._timestep - 1
            return (output[-1][var] or 0) - (output[last_index_previous_state][var] or 0)

    def reset(self,
              *,
              seed: int = None,
              return_info: bool = False,
              options: dict = None
              ):
        """
        Reset the PCSE-Gym environment to its initial state

        :param seed:
        :param return_info: flag indicating whether an info dict should be returned
        :param options: optional dict containing options for reinitialization
        :return: depending on the `return_info` flag, an initial observation is returned or a two-tuple of the initial
                 observation and the info dict
        """

        # Optionally set the seed
        super().reset(seed=seed)

        # Create an info dict
        info = dict()

        # Allow overriding wait_for_crop per reset (useful for evaluation-time preseason simulation)
        if options is None:
            options = {}
        if "wait_for_crop" in options:
            self._wait_for_crop = bool(options["wait_for_crop"])

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model(options)
        output = self._model.get_output()[-self._timestep:]
        o = self._get_observation(output)
        info['date'] = self.date

        return o, info if return_info else o

    def render(self, mode="human"):
        pass  # Nothing to see here
