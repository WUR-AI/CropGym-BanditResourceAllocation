import os
import yaml
from cropgymzoo import _SCENARIO_PATH
from itertools import chain

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