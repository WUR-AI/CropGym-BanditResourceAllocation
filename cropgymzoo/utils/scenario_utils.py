import os
import yaml
from cropgymzoo import _SCENARIO_PATH
from itertools import chain
import torch

def get_scenario_based_on_name(name: str):
    if '-s' in name:
        return 'zeeland'
    elif '-e' in name:
        return 'gelderland'
    elif '-n' in name:
        return 'groningen'
    else:
        return 'no_scenario'

def get_scenario_based_on_loc(location: tuple):
    if location[0] >= 53.0:
        return "groningen"
    elif location[0] <= 51.7:
        return "zeeland"
    elif 51.7 <= location[0] <= 53.0:
        return "gelderland"
    else:
        return "no_scenario"

def get_coords_for_soil(scenario):
    with open(os.path.join(_SCENARIO_PATH, "scenario_coords.yaml")) as f:
        dict_coords = yaml.load(f, Loader=yaml.SafeLoader)
    if 'gelderland' in scenario:
        return dict_coords['gelderland']
    elif 'groningen' in scenario:
        return dict_coords['groningen']
    elif 'zeeland' in scenario:
        return dict_coords['zeeland']
    else:
        return list(chain.from_iterable(dict_coords.values()))


def region_crop_picker(region, crop):
    region_suffix = {"groningen": "n", "zeeland": "s", "gelderland": "e"}
    crop_code = {"sugarbeet": "sb", "winterwheat": "ww", "potato": "pt"}
    return f"field-{crop_code[crop]}-{region_suffix[region]}"


def model_picker(model_file, dict_fields):
    orig_model_dict = torch.load(model_file, weights_only=False)

    assert isinstance(orig_model_dict, dict)

    new_model_dict = {}
    new_obs_rms_dict = {}
    for name, field in dict_fields.items():
        crop = field['crop']
        coor = (field['soil_lat'], field['soil_lon'])
        region = get_scenario_based_on_loc(coor)

        orig_agent_name = region_crop_picker(region, crop)
        new_model_dict[name] = orig_model_dict["models"][orig_agent_name]
        if isinstance(orig_model_dict["obs_rms"], dict):
            new_obs_rms_dict[name] = orig_model_dict["obs_rms"][orig_agent_name]

    orig_model_dict['models'] = new_model_dict
    if new_obs_rms_dict:
        orig_model_dict['obs_rms'] = new_obs_rms_dict

    return orig_model_dict
