import os
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable

from cropgym.baselines import make_baseline_runner, resolve_baseline


from concurrent.futures import ProcessPoolExecutor, as_completed

from cropgym import _DEFAULT_RESULTSDIR

from cropgym.utils.scenario_utils import load_dict_fields

from cropgym.envs.multi_field_env import MultiFieldEnv

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
        years: int | list[int],
        agent: str = "ROT",
        scenario: str = "full_budget",
        allocator: str | None = None,
        subset: bool = False,
        render: bool = False,
        farm_id: int | None = None,
        budget_reduction_kg_ha: int = 0,
        days_before_sowing: int = 7,
):
    """Run one GLOBAL farm over multiple season-years using the multi-campaign daisy-chain reset."""

    # Normalize years input
    if isinstance(years, int):
        season_years = [int(years)]
    else:
        season_years = sorted(set(int(y) for y in years))

    # Keep only supported scenario years (on disk)
    season_years = [y for y in season_years if y in [2020, 2021, 2022, 2023, 2024]]
    if not season_years:
        raise ValueError("No valid season years after filtering to [2020..2024].")

    if farm_id is None:
        raise ValueError("farm_id must be provided when running run_region_year")

    # farm_id is a GLOBAL farm index (0–51). Map it to (region, farmer_idx_within_region)
    region, farmer_idx = farm_int_mapper(int(farm_id))

    year_tag = f"{season_years[0]}-{season_years[-1]}" if len(season_years) > 1 else str(season_years[0])
    name_allocator = "" if allocator is None else f"_{allocator}"

    global_tag = f"farm{int(farm_id)}"
    red_tag = f"red{int(budget_reduction_kg_ha)}"

    name = f"results_{scenario}_{region}_{year_tag}_{red_tag}_{global_tag}" + name_allocator + f"_farmer_{farmer_idx}.pkl"
    if subset:
        name = f"results_{scenario}_{region}_{year_tag}_{red_tag}_{global_tag}" + name_allocator + f"_subset_farmer_{farmer_idx}.pkl"

    out_path = os.path.join(
        _DEFAULT_RESULTSDIR,
        agent,
        name,
    )

    # Skip if this farmer's results already exist
    if os.path.exists(out_path):
        print(f"Skipping {region}-{year_tag} farmer_{farmer_idx}; results already exist at {out_path}")
        return {}

    # Build farm_dict_by_year (daisy-chain reset uses this)
    farm_dict_by_year: dict[int, dict] = {}
    for sy in season_years:
        farm_dict_by_year[int(sy)] = load_dict_fields(farmer_idx, region, int(sy))

    # Initialize env once per farmer using first season's dict
    env = MultiFieldEnv(
        training=False,
        render=render,
        farm_dict=farm_dict_by_year[int(season_years[0])],
        reward='NSU'
    )

    if hasattr(env, "set_new_fields"):
        env.set_new_fields(farm_dict_by_year[int(season_years[0])])

    runner = make_baseline_runner(agent, env=env, render=render)

    # Reset env with multi-season campaign
    env.reset(options={
        "year": int(season_years[0]),
        "eval_horizon_years": season_years,
        "farm_dict_by_year": farm_dict_by_year,
        "preseason_allocation": True,
        "days_before_sowing": int(days_before_sowing),
    })

    info_dict: dict[int, dict] = {}

    # Apply budget reduction (kg N/ha) as a per-field reduction.
    # allocate_bandit_budgets expects "reductions" in 10 kg/ha units.
    reduction_units = float(int(budget_reduction_kg_ha)) / 10.0

    for sy in season_years:
        # Advance fields to allocation date for this season
        if hasattr(env, "advance_fields_to_allocation_dates"):
            env.advance_fields_to_allocation_dates(
                days_before_sowing=int(days_before_sowing),
                season_year=int(sy),
                farm_dict_by_year=farm_dict_by_year,
            )

        # Apply reduction (if any)
        if reduction_units > 0:
            env.allocate_bandit_budgets([min(reduction_units, (env.get_per_parcel_max_budget(ag)/10.0)) for ag in env.possible_agents])
            for ag in env.possible_agents:
                assert env.get_per_parcel_budget(ag) < env.get_per_parcel_max_budget(ag)

        # Run until season completes (daisy-chain in the same env instance)
        if hasattr(env, "run_til_past_season_year"):
            env.run_til_past_season_year(season_year=int(sy))
        else:
            runner.run(years=[int(sy)])

        # Collect per-season infos
        if hasattr(env, "collect_agent_infos_for_season"):
            info_dict[int(sy)] = env.collect_agent_infos_for_season(int(sy))
        else:
            info_dict[int(sy)] = {}

    # Save results (one file per reduction, containing all seasons)
    os.makedirs(os.path.join(_DEFAULT_RESULTSDIR, agent), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(info_dict, f)

    # Cleanup
    try:
        del runner
    except Exception:
        pass
    try:
        if env is not None:
            env.close()
    except Exception:
        pass

    return info_dict


# Helper for parallel execution
def _run_region_year_wrapper(args):
    years, agent, scenario, allocator, subset, render, farm_id, budget_reduction_kg_ha = args
    info_dict = run_region_year(
        years,
        agent=agent,
        scenario=scenario,
        allocator=allocator,
        subset=subset,
        render=render,
        farm_id=farm_id,
        budget_reduction_kg_ha=budget_reduction_kg_ha,
    )
    return years, info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, help="region name", default="all")
    parser.add_argument("--years", type=int, help="year", default=0)
    parser.add_argument("--baseline", type=str, choices=["ROT", "random"], default=None)
    parser.add_argument("--agent", type=str, help="deprecated alias for --baseline", default=None)
    parser.add_argument("--scenario", type=str, help="scenario name", default="full_budget-lp")
    parser.add_argument(
        "--allocator",
        type=str,
        choices=["LP", "LP_reduced", "LP_online", "LP_online_reduced"],
        help="allocation baseline name",
        default=None,
    )
    parser.add_argument("--num_workers", type=int, help="number of parallel workers (1 = no parallelism)", default=1)
    parser.add_argument("--render", action='store_true', help="render", dest='render')
    parser.add_argument("--subset", action='store_true', dest='subset')
    parser.add_argument("--farm", type=int, default=1)
    parser.set_defaults(render=False, subset=False)
    args = parser.parse_args()

    # We run a single farm only (selected by --farm)
    agent = resolve_baseline(
        baseline=args.baseline,
        deprecated_value=args.agent,
        deprecated_name="agent",
    )
    scenario = args.scenario
    num_workers = args.num_workers
    allocator = args.allocator
    subset = args.subset

    # make subfolder
    os.makedirs(os.path.join(_DEFAULT_RESULTSDIR, agent), exist_ok=True)

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
    budget_reductions = [
        0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
                         220, 240]

    # One daisy-chain run per budget reduction level (each run contains all seasons)
    all_jobs = [
        (years, agent, scenario, allocator, subset, args.render, farm_global, red)
        for red in budget_reductions
    ]

    if num_workers is None or num_workers <= 1:
        for yrs, agent_job, scenario_job, allocator_job, subset_job, render_job, farm_id_job, red in tqdm(all_jobs, desc="Running scenarios"):
            run_region_year(
                yrs,
                agent=agent_job,
                scenario=scenario_job,
                allocator=allocator_job,
                subset=subset_job,
                render=render_job,
                farm_id=farm_id_job,
                budget_reduction_kg_ha=red,
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_region_year_wrapper, job): job for job in all_jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="-----Running scenarios------"):
                future.result()

    # Aggregate only the single selected farm's files (one per reduction)
    aggregated_results = {}
    base_dir = Path(_DEFAULT_RESULTSDIR) / agent

    name_al = "" if allocator is None else f"_{allocator}"
    year_tag = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])

    farm_global = int(args.farm)
    region, farmer_idx = farm_int_mapper(farm_global)
    global_tag = f"farm{farm_global}"

    for red in budget_reductions:
        red_tag = f"red{int(red)}"
        pattern = f"results_{scenario}_{region}_{year_tag}_{red_tag}_{global_tag}" + name_al + f"_farmer_{farmer_idx}.pkl"
        if subset:
            pattern = f"results_{scenario}_{region}_{year_tag}_{red_tag}_{global_tag}" + name_al + f"_subset_farmer_{farmer_idx}.pkl"

        for pkl_file in base_dir.glob(pattern):
            key = f"{region}_{year_tag}_{red_tag}_{global_tag}_farmer_{farmer_idx}"
            if subset:
                key = f"{region}_{year_tag}_{red_tag}_{global_tag}_subset_farmer_{farmer_idx}"
            with open(pkl_file, "rb") as f:
                temp_dict = pickle.load(f)
            aggregated_results[key] = temp_dict

    out_name = f"results_{agent}_{scenario}_farm_{int(args.farm)}" + name_al + ".pkl"
    if subset:
        out_name = f"results_{agent}_{scenario}_farm_{int(args.farm)}" + name_al + "_subset.pkl"
    out_path = base_dir / out_name
    with open(out_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Saved aggregated results to {out_path}")
