import os
import yaml

import gymnasium as gym

from cropgymzoo.utils.defaults import get_wofost_default_crop_features, get_default_weather_features, get_default_action_features

_SOURCE_PATH = os.path.dirname(os.path.realpath(__file__))
_BASE_PATH = os.path.dirname(_SOURCE_PATH)
_CONFIG_PATH = os.path.join(_SOURCE_PATH, 'configs')

_CROPS_PATH = os.path.join(_CONFIG_PATH, 'crop')
_SITE_PATH = os.path.join(_CONFIG_PATH, 'sites')
_SOIL_PATH = os.path.join(_CONFIG_PATH, 'soil')
_SOILGRIDS_PATH = os.path.join(_SOIL_PATH, 'soilgrids')

_CROPS_LIST = os.path.join(_CROPS_PATH, 'crops.yaml')
_AGRO_CALENDAR_CONFIG = os.path.join(_CONFIG_PATH, 'agro', 'generic_cropcalendar.yaml')
_FIELDS_CONFIG = os.path.join(_CONFIG_PATH, 'fields.yaml')
_CROPS_CONFIG = os.path.join(_CONFIG_PATH, 'crop_info.yaml')
_WOFOST_CONFIG = os.path.join(_CONFIG_PATH, 'Wofost81_NWLP_MLWB_SNOMIN.conf')

# Initialize singular gymnasium envs
def register_predefined_cropgym_instances() -> None:

    with open(_FIELDS_CONFIG) as f:
        dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

    # importing modules...
    for key, field in dict_fields.items():
        gym.register(
            id=key,
            entry_point='cropgymzoo.envs.singular_env:ParcelEnv',
            kwargs={
                'crop_features': get_wofost_default_crop_features(),
                'weather_features': get_default_weather_features(),
                'action_features': get_default_action_features(),
                'location': (field['soil_lat'], field['soil_lon']),
                'crop': field['crop'],
                'year': 2000,
                'area': field['area'],
                'original': True,
                'training': True,
                'flatten_obs': False,
            },

        )

register_predefined_cropgym_instances()

from cropgymzoo.envs.pcse_env import PCSEEnv
from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.envs.worker_env import ParallelRLWorkers
from cropgymzoo.envs.allocation_env import AllocationBandit