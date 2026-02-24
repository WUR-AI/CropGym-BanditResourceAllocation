import os
import yaml
from itertools import chain
from copy import deepcopy
import torch
from pathlib import Path, PosixPath

from cropgymzoo import _SCENARIO_PATH

import hashlib
from typing import Tuple

def load_dict_fields(farm_id: int, region: str, year: int =None):
    if year is None:
        year = 2020

    with open(os.path.join(_SCENARIO_PATH, f"{region}",
                           f"farmer_{farm_id}.yaml"), 'rb') as f:
        dict_fields = yaml.safe_load(f)
    with open(os.path.join(_SCENARIO_PATH, f"{region}",
                           f"crop_rotations.yaml"), 'rb') as f:
        crops = yaml.safe_load(f)[f"farmer_{farm_id}"]

    for field_id, field_dict in dict_fields.items():
        dict_fields[field_id]['crop'] = crops[field_id][year]

    return dict_fields

def _stable_uniform_01(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    # Take first 8 bytes as an integer
    val = int.from_bytes(h[:8], "big")
    return (val % 10**8) / 10**8  # in [0, 1)


def choose_soil_type(crop: str, location: Tuple[float, float]) -> str:
    """
    Deterministic soil type based on region (via location)
    and crop/location key.
    """
    region = get_scenario_based_on_loc(location)

    if region == "groningen":
        options = ["clay", "silt"]
        probs = [0.4, 0.6]
    elif region == "gelderland":
        options = ["sand", "silt"]
        probs = [0.6, 0.4]
    elif region == "zeeland":
        options = ["silt", "sand", "clay"]
        probs = [0.4, 0.3, 0.3]
    else:
        options = ["clay", "silt", "sand"]
        probs = [1/3, 1/3, 1/3]

    # Build a deterministic key from crop + location
    lat, lon = location
    key = f"{crop}_{lat:.6f}_{lon:.6f}"
    r = _stable_uniform_01(key)  # in [0,1)

    cum = 0.0
    for opt, p in zip(options, probs):
        cum += p
        if r <= cum:
            return opt

    # Numerical safety net (shouldn't happen if probs sum to 1)
    return options[-1]

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
    # OK, make sure to deepcopy so there is no inplace stuff going on
    if isinstance(model_file, (str, Path, PosixPath)):
        model_dict = torch.load(model_file, weights_only=False)
        orig_model_dict = deepcopy(model_dict)
    else:
        orig_model_dict = deepcopy(model_file)

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
