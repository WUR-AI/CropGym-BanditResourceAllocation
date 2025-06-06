import unittest

import os
import yaml

from cropgymzoo import _FIELDS_CONFIG, get_default_action_features, get_default_weather_features, get_wofost_default_crop_features
from cropgymzoo.envs.singular_env import ParcelEnv


class TestCreationSingularEnv(unittest.TestCase):
    def test_create_singular_env(self):

        with open(_FIELDS_CONFIG) as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        env = ParcelEnv(
            crop_features=get_wofost_default_crop_features(),
            weather_features=get_default_weather_features(),
            action_features=get_default_action_features(),
            location=(dict_fields['field-1']['soil_lat'], dict_fields['field-1']['soil_lon']),
            crop=dict_fields['field-1']['crop'],
            year=2000,
            original=True,
            training=True,
        )

        self.assertEqual(isinstance(env, ParcelEnv), True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
