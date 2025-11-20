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


def _get_scenario_code(scenario):
    if scenario in ["full_budget", "full_budget_lp"]:
        return "max"
    elif scenario == "half_budget":
        return "low"
    else:
        return "max"


def run_region_year(
        region: str,
        year: int,
        agent: str = "baseline",
        scenario: str = "full_budget",
        render: bool = False,
):
    result_dict = {}

    _REGION_PATH = os.path.join(_SCENARIO_PATH, region)

    _YEAR_PATH = os.path.join(_REGION_PATH, str(year))

    year = year - 5 if "_lp" in scenario else year

    # Loop through farmers with a progress bar
    num_farmers = len(os.listdir(_YEAR_PATH)) - 1
    for i in tqdm(range(num_farmers), desc=f"{region}-{year} farmer"):
        info = None
        out_path = os.path.join(
            _DEFAULT_RESULTSDIR,
            agent,
            f"results_{scenario}_{region}_{year}_farmer_{i}.pkl",
        )
        # Skip if this farmer's results already exist
        if os.path.exists(out_path):
            # Optional: uncomment for debug logging
            print(f"Skipping {region}-{year} farmer_{i}; results already exist at {out_path}")
            continue
        _FARMER_PATH = os.path.join(_YEAR_PATH, f"farmer_{i}.yaml")

        with open(_FARMER_PATH, 'r') as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        env = MultiFieldEnv(
            years=[year],
            training=False,
            render=render,
            farm_dict=dict_fields,
        )

        if "MLP" in agent:
            # assume only one file in the folder
            model_path = Path(os.path.join(_DEFAULT_MODEL_DIR, agent))
            model_file = [p for p in model_path.iterdir() if p.is_file()][0]

            proper_model_file = model_picker(model_file, dict_fields)

            runner = MultiRLAgent(
                env=env,
                saved_model=proper_model_file,
                render=render,
            )
            # print(f"Running farmer_{i} at {region} in year {year}")
            info = runner.run(years=[year], scenario=_get_scenario_code(scenario))

        elif agent == "ROT":
            runner = RoTAgent(
                env=env,
                render=render,
            )

            info = runner.run(years=[year])

        if info is None:
            print(f"No results for farmer_{i} at {region} in year {year}")

        del runner
        del env

        with open(out_path, "wb") as f:
            pickle.dump(info, f)

        # result_dict[f"farmer_{i}"] = info

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
    region, year, agent, scenario, render = args
    info_dict = run_region_year(region, year, agent=agent, scenario=scenario, render=render)
    return region, year, info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, help="region name", default="all")
    parser.add_argument("--years", type=int, help="year", default=0)
    parser.add_argument("--agent", type=str, help="agent name", default="ROT")
    parser.add_argument("--scenario", type=str, help="scenario name", default="full_budget")
    parser.add_argument("--num_workers", type=int, help="number of parallel workers (1 = no parallelism)", default=1)
    parser.add_argument("--render", action='store_true', help="render", dest='render')
    parser.set_defaults(render=False)
    args = parser.parse_args()

    regions = args.regions
    years = args.years
    agent = args.agent
    scenario = args.scenario
    num_workers = args.num_workers

    # make subfolder
    os.makedirs(os.path.join(_DEFAULT_RESULTSDIR, args.agent), exist_ok=True)

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
    jobs = [(region, year, agent, scenario, args.render) for region in regions for year in years]

    if num_workers is None or num_workers <= 1:
        # Fallback to sequential execution
        for region, year, agent, scenario in tqdm(jobs, desc="Running scenarios"):
            info_dict = run_region_year(region, year, agent=agent, scenario=scenario, render=args.render)
            with open(os.path.join(_DEFAULT_RESULTSDIR, args.agent, f"results_{region}_{year}.pkl"), "wb") as f:
                pickle.dump(info_dict, f)
            # results_dict[f"{region}_{year}"] = info_dict
    else:
        # Parallel execution over regions/years
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_run_region_year_wrapper, job): job
                for job in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="-----Running scenarios------"):
                region, year, info_dict = future.result()
                # results_dict[f"{region}_{year}"] = info_dict

    # Aggregate all per-farmer pickle files into one big dictionary
    aggregated_results = {}
    base_dir = Path(_DEFAULT_RESULTSDIR) / agent

    for region in regions:
        for year in years:
            pattern = f"results_{scenario}_{region}_{year}_farmer_*.pkl"
            for pkl_file in base_dir.glob(pattern):
                # Example filename: results_full_budget_groningen_2020_farmer_0.pkl
                stem_parts = pkl_file.stem.split("_")
                # last part should be like "0" from "farmer_0"; keep the whole farmer tag for clarity
                farmer_tag = "_".join(stem_parts[-2:])  # e.g. "farmer_0"
                key = f"{region}_{year}_{farmer_tag}"
                with open(pkl_file, "rb") as f:
                    temp_dict = pickle.load(f)
                aggregated_results[key] = temp_dict.get(year, temp_dict)

    out_path = base_dir / f"results_{agent}_{scenario}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Saved aggregated results to {out_path}")