import os
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm

from dataclasses import dataclass
from typing import Hashable, List
import numpy as np

from cropgymzoo.utils.agent_helpers import model_picker


@dataclass
class FieldResponse:
    alpha: float
    beta: float
    r2: float
    n_points: int

@dataclass
class FarmAllocationResult:
    farm_id: Hashable
    year: int
    field_ids: List[Hashable]
    N_opt: np.ndarray          # kg per field
    alpha: np.ndarray          # slope per field
    area: np.ndarray           # ha per field
    N_per_ha: np.ndarray       # kg/ha per field (allocation scaled to area)
    frac_of_rate: np.ndarray   # N_per_ha / farm_rate (allocation scaled to rate)

from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from cropgymzoo import _SCENARIO_PATH, _DEFAULT_MODEL_DIR, _DEFAULT_RESULTSDIR

from cropgymzoo.eval_policy import MultiRLAgent, RoTAgent, RandomAgent

from cropgymzoo.envs.multi_field_env import MultiFieldEnv


def _get_scenario_code(scenario):
    if scenario in ["full_budget", "full_budget-lp"]:
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
        allocator: str = "None",
        render: bool = False,
):
    result_dict = {}

    _REGION_PATH = os.path.join(_SCENARIO_PATH, region)

    _YEAR_PATH = os.path.join(_REGION_PATH, str(year))

    year = year - 5 if "-lp" in scenario else year

    name_allocator = "" if allocator is None else f"_{allocator}"

    # Loop through farmers with a progress bar
    num_farmers = len(os.listdir(_YEAR_PATH)) - 1
    for i in tqdm(range(num_farmers), desc=f"{region}-{year} farmer"):
        info = None
        out_path = os.path.join(
            _DEFAULT_RESULTSDIR,
            agent,
            f"results_{scenario}_{region}_{year}" + name_allocator + f"_farmer_{i}.pkl",
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

        if "reduced" in scenario and allocator == "None":
            for ag in env.possible_agents:
                env.set_per_parcel_budget(ag, env.get_per_parcel_max_budget(ag) - 100)
                assert env.get_per_parcel_budget(ag) < env.get_per_parcel_max_budget(ag)

        if allocator is not None and "LP" in allocator:
            if year not in [2015, 2016, 2017, 2018, 2019]:
                year = year - 5
            if allocator == "LP_low":
                name = "lp_results_reduced.pkl"
            else:  # "LP_max"
                name = "lp_results.pkl"

            with open(os.path.join(_DEFAULT_RESULTSDIR, "LP", name), "rb") as f:
                lp = pickle.load(f)

            alloc_info = lp["lp_allocations"][f"{region}_farmer_{i}"][year]

            alloc_vec = alloc_info["N_per_ha"]
            alloc_fields = alloc_info["field_ids"]

            assert len(alloc_vec) == len(env.possible_agents) == len(alloc_fields)

            for (alloc, field_name) in zip(alloc_vec, alloc_fields):
                env.set_per_parcel_budget(field_name, alloc)
                assert env.get_per_parcel_budget_left(field_name) <= alloc

            if year in [2015, 2016, 2017, 2018, 2019]:
                year = year + 5


        if "MLP" in agent:
            # assume only one file in the folder
            model_path = Path(os.path.join(_DEFAULT_MODEL_DIR, agent))
            model_file = [p for p in model_path.iterdir() if p.is_file()][0]

            proper_model_file = model_picker(model_file, dict_fields)

            if proper_model_file["args"].get('special_action_space', False):
                env.override_action_space()

            runner = MultiRLAgent(
                env=env,
                saved_model=proper_model_file,
                render=render,
            )
            # print(f"Running farmer_{i} at {region} in year {year}")
            info = runner.run(years=[year])

        elif agent == "ROT":
            runner = RoTAgent(
                env=env,
                render=render,
            )
            info = runner.run(years=[year])

        elif agent == "random":
            runner = RandomAgent(env=env, render=render)
            info = runner.run(years=[year])

        if info is None:
            print(f"No results for farmer_{i} at {region} in year {year}")

        del runner
        del env

        with open(out_path, "wb") as f:
            pickle.dump(info, f)

        # result_dict[f"farmer_{i}"] = info

    return result_dict


def region_crop_picker(region, crop):
    region_suffix = {"groningen": "n", "zeeland": "s", "gelderland": "e"}
    crop_code = {"sugarbeet": "sb", "winterwheat": "ww", "potato": "pt"}
    return f"field-{crop_code[crop]}-{region_suffix[region]}"


# Helper for parallel execution
def _run_region_year_wrapper(args):
    region, year, agent, scenario, allocator, render = args
    info_dict = run_region_year(region, year, agent=agent, scenario=scenario, allocator=allocator, render=render)
    return region, year, info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, help="region name", default="all")
    parser.add_argument("--years", type=int, help="year", default=0)
    parser.add_argument("--agent", type=str, help="agent name", default="ROT")
    parser.add_argument("--scenario", type=str, help="scenario name", default="full_budget")
    parser.add_argument("--allocator", type=str, help="allocator name", default=None)
    parser.add_argument("--num_workers", type=int, help="number of parallel workers (1 = no parallelism)", default=1)
    parser.add_argument("--render", action='store_true', help="render", dest='render')
    parser.set_defaults(render=False)
    args = parser.parse_args()

    regions = args.regions
    years = args.years
    agent = args.agent
    scenario = args.scenario
    num_workers = args.num_workers
    allocator = args.allocator

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
    all_jobs = [(region, year, agent, scenario, allocator, args.render) for region in regions for year in years]
    sliced_jobs = [all_jobs[i:i+3] for i in range(0, len(all_jobs), 3)]

    if num_workers is None or num_workers <= 1:
        # Fallback to sequential execution
        for region, year, agent, scenario, allocator, args.render in tqdm(all_jobs, desc="Running scenarios"):
            info_dict = run_region_year(region, year, agent=agent, scenario=scenario, allocator=allocator, render=args.render)
            # results_dict[f"{region}_{year}"] = info_dict
    else:
        # Parallel execution over regions/years
        for jobs in tqdm(sliced_jobs, desc="Slicing jobs"):
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

    if "-lp" in scenario:
        years = [year - 5 for year in years]

    name_al = "" if allocator is None else f"_{allocator}"

    for region in regions:
        for year in years:
            pattern = f"results_{scenario}_{region}_{year}" + name_al + f"_farmer_*.pkl"
            for pkl_file in base_dir.glob(pattern):
                # Example filename: results_full_budget_groningen_2020_farmer_0.pkl
                stem_parts = pkl_file.stem.split("_")
                # last part should be like "0" from "farmer_0"; keep the whole farmer tag for clarity
                farmer_tag = "_".join(stem_parts[-2:])  # e.g. "farmer_0"
                key = f"{region}_{year}_{farmer_tag}"
                with open(pkl_file, "rb") as f:
                    temp_dict = pickle.load(f)
                aggregated_results[key] = temp_dict.get(year, temp_dict)

    out_name = f"results_{agent}_{scenario}" + name_al + ".pkl"
    out_path = base_dir / out_name
    with open(out_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Saved aggregated results to {out_path}")