import os
import yaml

from cropgymzoo import _FIELDS_CONFIG, ParcelEnv
from cropgymzoo.utils.defaults import get_wofost_default_crop_features, get_default_weather_features, get_default_action_features

import gymnasium as gym


def register_predefined_cropgym_instances() -> None:

    with open(_FIELDS_CONFIG) as f:
        dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

    for field in dict_fields:
        gym.register(
            id='field-1',
            entry_point='cropgymzoo.envs.singular_env:ParcelEnv',
            kwargs={
                'crop_features': get_wofost_default_crop_features(2),
                'weather_features': get_default_weather_features(),
                'action_features': get_default_action_features(),
                'locations': (field['soil_lat'], field['soil_lon']),
                'crop': field['crop'],
            },

        )

def register_eval_envs():
    ...

def register_test_envs():
    ...