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


# ------------------------------------------------------------
# Helper: map a global farm integer (0–52) to (region, farmer_id)
# ------------------------------------------------------------
def farm_int_mapper(x: int):
    """
    Maps a global farm index (0–52) to (region, farmer_id).

    Regions:
        Gelderland: 0–11  (12 farms)
        Groningen:  12–24 (13 farms)
        Zeeland:    25–51 (27 farms)

    Returns:
        (region_name: str, farmer_id: int)
    """
    if not (0 <= x <= 51):
        raise ValueError(f"farm index must be between 0 and 51, got {x}")

    # Gelderland (0–11)
    if x < 12:
        return "gelderland", x

    # Groningen (12–24)
    if x < 12 + 13:
        return "groningen", x - 12

    # Zeeland (25–51)
    return "zeeland", x - (12 + 13)


def run_region_year(
        year: int,
        agent: str = "baseline",
        scenario: str = "full_budget",
        allocator: str = "None",
        subset: bool = False,
        render: bool = False,
        farm_id: int | None = None,
        budget_reduction_kg_ha: int = 0,
):
    result_dict = {}

    # farm_id is a GLOBAL farm index (0–51). Map it to (region, farmer_idx_within_region)
    if farm_id is None:
        raise ValueError("farm_id must be provided when running run_region_year")
    region, farmer_idx = farm_int_mapper(int(farm_id))

    _REGION_PATH = os.path.join(_SCENARIO_PATH, region)
    _YEAR_PATH = os.path.join(_REGION_PATH, str(year))

    year = year - 5 if "-lp" in scenario else year
    name_allocator = "" if allocator is None else f"_{allocator}"

    global_tag = f"farm{int(farm_id)}"
    red_tag = f"red{int(budget_reduction_kg_ha)}"
    name = f"results_{scenario}_{region}_{year}_{red_tag}_{global_tag}" + name_allocator + f"_farmer_{farmer_idx}.pkl"
    if subset:
        name = f"results_{scenario}_{region}_{year}_{red_tag}_{global_tag}" + name_allocator + f"_subset_farmer_{farmer_idx}.pkl"
    out_path = os.path.join(
        _DEFAULT_RESULTSDIR,
        agent,
        name,
    )
    # Skip if this farmer's results already exist
    if os.path.exists(out_path):
        print(f"Skipping {region}-{year} farmer_{farmer_idx}; results already exist at {out_path}")
        return result_dict
    _FARMER_PATH = os.path.join(_YEAR_PATH, f"farmer_{farmer_idx}.yaml")

    with open(_FARMER_PATH, 'r') as f:
        dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

    env = MultiFieldEnv(
        years=[year],
        training=False,
        render=render,
        farm_dict=dict_fields,
        reward='NSU'
    )

    # ------------------------------------------------------------
    # Apply budget reduction (kg N/ha) as a per-field reduction.
    # allocate_bandit_budgets expects "reductions" in 10 kg/ha units.
    # So: 0, 50, 100 kg/ha -> 0, 5, 10.
    # ------------------------------------------------------------
    reduction_units = int(budget_reduction_kg_ha) / 10.0
    if reduction_units > 0:
        env.allocate_bandit_budgets([reduction_units for _ in env.possible_agents])
        for ag in env.possible_agents:
            assert env.get_per_parcel_budget(ag) < env.get_per_parcel_max_budget(ag)
    runner = RoTAgent(
        env=env,
        render=render,
    )
    info = runner.run(years=[year])

    if info is None:
        print(f"No results for farmer_{farmer_idx} at {region} in year {year}")

    del runner
    del env

    with open(out_path, "wb") as f:
        pickle.dump(info, f)

    return result_dict


# Helper for parallel execution
def _run_region_year_wrapper(args):
    year, agent, scenario, allocator, subset, render, farm_id, budget_reduction_kg_ha = args
    info_dict = run_region_year(
        year,
        agent=agent,
        scenario=scenario,
        allocator=allocator,
        subset=subset,
        render=render,
        farm_id=farm_id,
        budget_reduction_kg_ha=budget_reduction_kg_ha,
    )
    return year, info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, help="region name", default="all")
    parser.add_argument("--years", type=int, help="year", default=0)
    parser.add_argument("--agent", type=str, help="agent name", default="ROT")
    parser.add_argument("--scenario", type=str, help="scenario name", default="full_budget-lp")
    parser.add_argument("--allocator", type=str, help="allocator name", default=None)
    parser.add_argument("--num_workers", type=int, help="number of parallel workers (1 = no parallelism)", default=1)
    parser.add_argument("--render", action='store_true', help="render", dest='render')
    parser.add_argument("--farm", type=int, default=1)
    parser.set_defaults(render=False, subset=False)
    args = parser.parse_args()

    # We run a single farm only (selected by --farm)
    agent = args.agent
    scenario = args.scenario
    num_workers = args.num_workers
    allocator = args.allocator
    subset = args.subset

    # make subfolder
    os.makedirs(os.path.join(_DEFAULT_RESULTSDIR, args.agent), exist_ok=True)

    # Global farm id (0–51)
    farm_global = int(args.farm)
    region, farmer_idx = farm_int_mapper(farm_global)
    print(f"Running global farm {farm_global} -> {region}_farmer_{farmer_idx}")

    # Run 5 years: if args.years==0 -> [2020..2024], else interpret args.years as the end-year
    if args.years == 0:
        years = [2020, 2021, 2022, 2023, 2024]
    else:
        end_year = int(args.years)
        years = list(range(end_year - 4, end_year + 1))
        # Keep only supported scenario years
        years = [y for y in years if y in [2020, 2021, 2022, 2023, 2024]]

    if subset:
        years = [years[0]]

    # Budget reductions in kg N/ha
    budget_reductions = [0, 50, 100]

    results_dict = {}
    # Create list of jobs: (year, agent, scenario, allocator, subset, render, farm_global, reduction)
    all_jobs = [
        (year, agent, scenario, allocator, subset, args.render, farm_global, red)
        for year in years
        for red in budget_reductions
    ]
    sliced_jobs = [all_jobs[i:i+3] for i in range(0, len(all_jobs), 3)]

    if num_workers is None or num_workers <= 1:
        # Fallback to sequential execution
        for year, agent, scenario, allocator, subset_job, render_job, farm_id, red in tqdm(all_jobs, desc="Running scenarios"):
            info_dict = run_region_year(
                year,
                agent=agent,
                scenario=scenario,
                allocator=allocator,
                subset=subset_job,
                render=render_job,
                farm_id=farm_id,
                budget_reduction_kg_ha=red,
            )
            # results_dict[f"{year}"] = info_dict
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
                    year, info_dict = future.result()
                    # results_dict[f"{year}"] = info_dict

    # Aggregate only the single selected farm's files
    aggregated_results = {}
    base_dir = Path(_DEFAULT_RESULTSDIR) / agent

    if "-lp" in scenario:
        years = [year - 5 for year in years]

    name_al = "" if allocator is None else f"_{allocator}"

    # Recompute mapping for aggregation (same as run)
    farm_global = int(args.farm)
    region, farmer_idx = farm_int_mapper(farm_global)
    global_tag = f"farm{farm_global}"

    for year in years:
        for red in budget_reductions:
            red_tag = f"red{int(red)}"
            # IMPORTANT: match exactly the single farmer file
            pattern = f"results_{scenario}_{region}_{year}_{red_tag}_{global_tag}" + name_al + f"_farmer_{farmer_idx}.pkl"
            if subset:
                pattern = f"results_{scenario}_{region}_{year}_{red_tag}_{global_tag}" + name_al + f"_subset_farmer_{farmer_idx}.pkl"

            for pkl_file in base_dir.glob(pattern):
                key = f"{region}_{year}_{red_tag}_{global_tag}_farmer_{farmer_idx}"
                if subset:
                    key = f"{region}_{year}_{red_tag}_{global_tag}_subset_farmer_{farmer_idx}"
                with open(pkl_file, "rb") as f:
                    temp_dict = pickle.load(f)
                aggregated_results[key] = temp_dict.get(year, temp_dict)

    out_name = f"results_{agent}_{scenario}_farm_{int(args.farm)}" + name_al + ".pkl"
    if subset:
        out_name = f"results_{agent}_{scenario}_farm_{int(args.farm)}" + name_al + "_subset.pkl"
    out_path = base_dir / out_name
    with open(out_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Saved aggregated results to {out_path}")