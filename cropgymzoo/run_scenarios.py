import os
from pathlib import Path
import torch
import argparse
import pickle

import yaml

from cropgymzoo import _SCENARIO_PATH, _DEFAULT_MODEL_DIR
from cropgymzoo.utils.scenario_utils import get_scenario_based_on_loc

from cropgymzoo.eval_policy import MultiRLAgent

from cropgymzoo.envs.multi_field_env import MultiFieldEnv



def run_region_year(
        region: str,
        year: int,
        agent: str = "baseline",
        scenario: str = "baseline"
):
    result_dict = {}

    _REGION_PATH = os.path.join(_SCENARIO_PATH, region)

    _YEAR_PATH = os.path.join(_REGION_PATH, str(year))

    # Loop through farmers
    for i in range(len(os.listdir(_YEAR_PATH)) - 1):
        info = None
        _FARMER_PATH = os.path.join(_YEAR_PATH, f"farmer_{i}.yaml")

        with open(_FARMER_PATH, 'r') as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        env = MultiFieldEnv(
            years=[year],
            training=False,
            render=True,
            farm_dict=dict_fields,
        )

        if agent == "MLP":
            # assume only one file in the folder
            model_path = Path(os.path.join(_DEFAULT_MODEL_DIR, agent))
            model_file = [p for p in model_path.iterdir() if p.is_file()][0]

            proper_model_file = model_picker(model_file, dict_fields)

            runner = MultiRLAgent(
                env=env,
                saved_model=proper_model_file,
                render=True,
            )
            print(f"Running farmer_{i} at {region} in year {year}")
            info = runner.run(years=[year])

        result_dict[f"farmer_{i}"] = info

        return result_dict

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


def region_crop_picker(region, crop):
    region_suffix = {"groningen": "n", "zeeland": "s", "gelderland": "e"}
    crop_code = {"sugarbeet": "sb", "winterwheat": "ww", "potato": "pt"}
    return f"field-{crop_code[crop]}-{region_suffix[region]}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, help="region name", default="all")
    parser.add_argument("--years", type=int, help="year", default=0)
    parser.add_argument("--agent", type=str, help="agent name", default="baseline")
    parser.add_argument("--scenario", type=str, help="scenario name", default="baseline")
    args = parser.parse_args()

    regions = args.regions
    years = args.years
    agent = args.agent
    scenario = args.scenario

    if regions == "all":
        regions = ["groningen", "zeeland", "gelderland"]
    else:
        regions = [regions]
    if years == 0:
        years = [2020, 2021, 2022, 2023, 2024]
    else:
        years = [years]

    results_dict = {}
    for region in regions:
        for year in years:
            info_dict = run_region_year(region, year, agent=agent, scenario=scenario)
            results_dict[f"{region}_{year}"] = info_dict