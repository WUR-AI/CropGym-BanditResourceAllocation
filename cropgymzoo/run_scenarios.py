import os
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable

from cropgymzoo.utils.scenario_utils import model_picker


from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from cropgymzoo import _SCENARIO_PATH, _DEFAULT_MODEL_DIR, _DEFAULT_RESULTSDIR

from cropgymzoo.eval_policy import MultiRLAgent, RoTAgent, RandomAgent

from cropgymzoo.envs.multi_field_env import MultiFieldEnv

import numpy as np

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
    N_opt_ha: np.ndarray       # kg/ha per field (LP decision)
    N_opt_abs: np.ndarray      # kg per field (rate * area)
    alpha: np.ndarray          # slope per field (per +1 kg/ha)
    area: np.ndarray           # ha per field
    bmax_rate: np.ndarray      # kg/ha per field
    frac_of_rate: np.ndarray   # N_opt_ha / farm_rate

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
        subset: bool = False,
        render: bool = False,
):
    result_dict = {}

    _REGION_PATH = os.path.join(_SCENARIO_PATH, region)

    _YEAR_PATH = os.path.join(_REGION_PATH, str(year))

    year = year - 5 if "-lp" in scenario else year

    name_allocator = "" if allocator is None else f"_{allocator}"

    # Loop through farmers with a progress bar
    num_farmers = len(os.listdir(_YEAR_PATH)) - 1
    if subset:
        num_farmers = 2
    for i in tqdm(range(num_farmers), desc=f"{region}-{year} farmer"):
        info = None
        name = f"results_{scenario}_{region}_{year}" + name_allocator + f"_farmer_{i}.pkl"
        if subset:
            name = f"results_{scenario}_{region}_{year}" + name_allocator + f"_subset_farmer_{i}.pkl"
        out_path = os.path.join(
            _DEFAULT_RESULTSDIR,
            agent,
            name,
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

        if "reduced" in scenario and allocator is None:
            env.allocate_bandit_budgets([10 for _ in env.possible_agents])
            for ag in env.possible_agents:
            #     env.set_per_parcel_budget(ag, env.get_per_parcel_max_budget(ag) - 100)
            #     env.global_allocated_budget = env._get_global_budget_left()
                assert env.get_per_parcel_budget(ag) < env.get_per_parcel_max_budget(ag)

        if allocator is not None and "LP" in allocator:
            # LP allocations are precomputed and stored under results/LP
            # If scenario has -lp, the evaluation year is shifted earlier for realism.
            lp_year = year
            if lp_year not in [2015, 2016, 2017, 2018, 2019] and "-lp" in scenario:
                lp_year = lp_year - 5

            # Pick the correct LP file (normal vs reduced)
            lp_suffix = "reduced_" if allocator == "LP_low" else ""
            lp_name = f"lp_results_{lp_suffix}{agent}.pkl"

            with open(os.path.join(_DEFAULT_RESULTSDIR, "LP", lp_name), "rb") as f:
                lp = pickle.load(f)

            alloc_info = lp["lp_allocations"][f"{region}_farmer_{i}"][lp_year]

            # LP allocations are stored as kg/ha
            alloc_vec = alloc_info.get("N_opt_ha", alloc_info.get("N_per_ha"))
            alloc_fields = alloc_info["field_ids"]

            assert len(alloc_vec) == len(env.possible_agents) == len(alloc_fields)

            alloc_reductions = np.asarray([(env.get_per_parcel_max_budget(ag) - alloc_vec[i])/10 for i, ag in enumerate(env.possible_agents)])

            env.allocate_bandit_budgets(alloc_reductions)

            # for alloc, field_name in zip(alloc_vec, alloc_fields):
            #     alloc = float(alloc)
            #     env.set_per_parcel_budget(field_name, alloc)
            #     assert env.get_per_parcel_budget_left(field_name) <= alloc


        if "MLP" in agent:
            # assume only one file in the folder
            model_path = Path(os.path.join(_DEFAULT_MODEL_DIR, agent))
            model_file = [p for p in model_path.iterdir() if p.is_file()][0]

            proper_model_file = model_picker(model_file, dict_fields)

            if getattr(proper_model_file["args"], 'special_action_space', False):
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


# Helper for parallel execution
def _run_region_year_wrapper(args):
    region, year, agent, scenario, allocator, subset, render = args
    info_dict = run_region_year(region, year, agent=agent, scenario=scenario, allocator=allocator, subset=subset, render=render)
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
    parser.add_argument("--subset", action='store_true', dest='subset')
    parser.set_defaults(render=False, subset=False)
    args = parser.parse_args()

    regions = args.regions
    years = args.years
    agent = args.agent
    scenario = args.scenario
    num_workers = args.num_workers
    allocator = args.allocator
    subset = args.subset

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

    if subset:
        years = [2020]

    results_dict = {}
    # Create list of (region, year, agent, scenario) jobs
    all_jobs = [(region, year, agent, scenario, allocator, subset, args.render) for region in regions for year in years]
    sliced_jobs = [all_jobs[i:i+3] for i in range(0, len(all_jobs), 3)]

    if num_workers is None or num_workers <= 1:
        # Fallback to sequential execution
        for region, year, agent, scenario, allocator, subset_job, render_job in tqdm(all_jobs, desc="Running scenarios"):
            info_dict = run_region_year(
                region,
                year,
                agent=agent,
                scenario=scenario,
                allocator=allocator,
                subset=subset_job,
                render=render_job,
            )
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
            if subset:
                pattern = f"results_{scenario}_{region}_{year}" + name_al + "_subset_farmer_*.pkl"
            for pkl_file in base_dir.glob(pattern):
                # Example filename: results_full_budget_groningen_2020_farmer_0.pkl
                stem_parts = pkl_file.stem.split("_")
                # last part should be like "0" from "farmer_0"; keep the whole farmer tag for clarity
                farmer_tag = "_".join(stem_parts[-2:])  # e.g. "farmer_0"
                key = f"{region}_{year}_{farmer_tag}"
                if subset:
                    key = f"{region}_{year}_subset_{farmer_tag}"
                with open(pkl_file, "rb") as f:
                    temp_dict = pickle.load(f)
                aggregated_results[key] = temp_dict.get(year, temp_dict)

    out_name = f"results_{agent}_{scenario}" + name_al + ".pkl"
    if subset:
        out_name = f"results_{agent}_{scenario}" + name_al + "_subset.pkl"
    out_path = base_dir / out_name
    with open(out_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Saved aggregated results to {out_path}")