import os
from pathlib import Path
import torch
import argparse
import pickle
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from cropgymzoo import _SCENARIO_PATH, _DEFAULT_MODEL_DIR, _DEFAULT_RESULTSDIR
from cropgymzoo.utils.scenario_utils import get_scenario_based_on_loc

from cropgymzoo.eval_policy import MultiRLAgent, RoTAgent

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

    # Loop through farmers with a progress bar
    num_farmers = len(os.listdir(_YEAR_PATH)) - 1
    for i in tqdm(range(num_farmers), desc=f"{region}-{year} farmer"):
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
            # print(f"Running farmer_{i} at {region} in year {year}")
            info = runner.run(years=[year])

        elif agent == "ROT":
            runner = RoTAgent(
                env=env,
                render=True,
            )

            info = runner.run(years=[year])

        if info is None:
            print(f"No results for farmer_{i} at {region} in year {year}")

        del runner
        del env

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


# Helper for parallel execution
def _run_region_year_wrapper(args):
    region, year, agent, scenario = args
    info_dict = run_region_year(region, year, agent=agent, scenario=scenario)
    return region, year, info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, help="region name", default="all")
    parser.add_argument("--years", type=int, help="year", default=0)
    parser.add_argument("--agent", type=str, help="agent name", default="baseline")
    parser.add_argument("--scenario", type=str, help="scenario name", default="full_budget")
    parser.add_argument("--num_workers", type=int, help="number of parallel workers (1 = no parallelism)", default=1)
    args = parser.parse_args()

    regions = args.regions
    years = args.years
    agent = args.agent
    scenario = args.scenario
    num_workers = args.num_workers

    if regions == "all":
        regions = ["groningen", "zeeland", "gelderland"]
    else:
        regions = [regions]
    if years == 0:
        years = [2020, 2021, 2022, 2023, 2024]
    else:
        years = [years]

    results_dict = {}
    # Create list of (region, year, agent, scenario) jobs
    jobs = [(region, year, agent, scenario) for region in regions for year in years]

    if num_workers is None or num_workers <= 1:
        # Fallback to sequential execution
        for region, year, agent, scenario in tqdm(jobs, desc="Running scenarios"):
            info_dict = run_region_year(region, year, agent=agent, scenario=scenario)
            results_dict[f"{region}_{year}"] = info_dict
    else:
        # Parallel execution over regions/years
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_run_region_year_wrapper, job): job
                for job in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running scenarios"):
                region, year, info_dict = future.result()
                results_dict[f"{region}_{year}"] = info_dict

    with open(os.path.join(_DEFAULT_RESULTSDIR, f"results_{agent}_{scenario}.pkl"), "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Saved results to {os.path.join(_DEFAULT_RESULTSDIR, f'results_{agent}_{scenario}.pkl')}")