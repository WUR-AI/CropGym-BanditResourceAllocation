import os
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable

from cropgymzoo.utils.scenario_utils import model_picker, load_dict_fields
from cropgymzoo.fit_nue_response import solve_discrete_lp_for_env, make_df_nue_response


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


def farm_region_mapper(region: str, farmer_id: int) -> int:
    """
    Maps (region, farmer_id) back to global farm index (0–52).

    Regions:
        Gelderland: farmer_id 0–11  -> global 0–11
        Groningen:  farmer_id 0–12  -> global 12–24
        Zeeland:    farmer_id 0–26  -> global 25–51

    Returns:
        global_farm_index: int
    """

    region = region.lower()

    if region == "gelderland":
        if not (0 <= farmer_id < 12):
            raise ValueError("Gelderland farmer_id must be in range 0–11")
        return farmer_id

    elif region == "groningen":
        if not (0 <= farmer_id < 13):
            raise ValueError("Groningen farmer_id must be in range 0–12")
        return 12 + farmer_id

    elif region == "zeeland":
        if not (0 <= farmer_id < 27):
            raise ValueError("Zeeland farmer_id must be in range 0–26")
        return 12 + 13 + farmer_id

    else:
        raise ValueError(f"Unknown region: {region}")


def run_region_year(
        region: str,
        years: list,
        agent: str = "baseline",
        scenario: str = "full_budget",
        allocator: str = "None",
        subset: bool = False,
        render: bool = False,
):
    """
    Runs a multi-season evaluation for a region over all requested years.
    """
    # Accept years as int or list[int]
    if isinstance(years, int):
        season_years = [years]
    else:
        season_years = list(sorted(set(int(y) for y in years)))
    # Defensive: ensure unique, sorted
    season_years = sorted(set(season_years))

    # For each season_year, determine the data_year for loading YAML files
    def get_data_year(season_year):
        return season_year - 5 if "-lp" in scenario else season_year

    _REGION_PATH = os.path.join(_SCENARIO_PATH, region)
    # Use the first season's data_year for farmer count
    name_allocator = "" if allocator is None else f"_{allocator}"


    # Loop through farmers with a progress bar
    num_farmers = len([name for name in os.listdir(_REGION_PATH) if "farmer" in name]) - 1


    if subset:
        num_farmers = 2
    year_tag = f"{season_years[0]}-{season_years[-1]}" if len(season_years) > 1 else str(season_years[0])


    for i in tqdm(range(num_farmers), desc=f"{region}-{year_tag} farmer"):
        info = None
        # Output file for this farmer, all seasons
        name = f"results_{scenario}_{region}_{year_tag}{name_allocator}_farmer_{i}.pkl"
        if subset:
            name = f"results_{scenario}_{region}_{year_tag}{name_allocator}_subset_farmer_{i}.pkl"
        out_path = os.path.join(
            _DEFAULT_RESULTSDIR,
            agent,
            name,
        )
        # Skip if this farmer's results already exist
        if os.path.exists(out_path):
            print(f"Skipping {region}-{year_tag} farmer_{i}; results already exist at {out_path}")
            continue
        # Build farm_dict_by_year for this farmer
        farm_dict_by_year = {}
        for sy in season_years:
            year_path = os.path.join(_REGION_PATH, str(sy))
            farmer_path = os.path.join(year_path, f"farmer_{i}.yaml")

            farm_dict_by_year[int(sy)] = load_dict_fields(i, region, sy)

            # with open(farmer_path, 'r') as f:
            #     farm_dict_by_year[int(sy)] = yaml.load(f, Loader=yaml.SafeLoader)


        # Initialize env once per farmer using first season's dict
        env = MultiFieldEnv(
            training=False,
            render=render,
            farm_dict=farm_dict_by_year[season_years[0]],
            reward='NSU'
        )

        # Optionally set new fields if available
        if hasattr(env, "set_new_fields"):
            env.set_new_fields(farm_dict_by_year[season_years[0]])
        # Allocation strategies before each season
        # Prepare LP allocation if needed (loaded once for all seasons)

        lp_df = None
        global_farm_id = None
        global_farm_idx = None

        if allocator is not None and "LP" in allocator:
            # Path-B benchmark: use precomputed grid metrics (0,20,40,...) for this GLOBAL farm.
            global_farm_idx = int(farm_region_mapper(region, i))
            global_farm_id = f"farm{global_farm_idx}"

            # The precomputed grid run (aggregated) is written by `run_for_response.py` as:
            #   results/{agent}/results_{agent}_{scenario}_farm_{global_farm_idx}.pkl
            metrics_path = os.path.join(
                _DEFAULT_RESULTSDIR,
                str(agent),
                f"results_{agent}_{scenario}-lp_farm_{global_farm_idx}.pkl"
                if scenario == "full_budget"
                else f"results_{agent}_full_budget-lp_farm_{global_farm_idx}.pkl",
            )
            if not os.path.exists(metrics_path):
                raise FileNotFoundError(
                    f"Precomputed LP grid metrics not found for {global_farm_id}. "
                    f"Expected at: {metrics_path}. "
                    "Run run_for_response.py first to generate the grid pickles."
                )

            with open(metrics_path, "rb") as f:
                metrics_data = pickle.load(f)

            # Convert aggregated grid metrics pickle -> tidy DataFrame (expects columns like farm_id, field_id, year, red_level, N, NUE, Nsurp)
            lp_df = make_df_nue_response(metrics_data)


        # Policy runner object (created once)
        if "MLP" in agent:
            model_path = Path(os.path.join(_DEFAULT_MODEL_DIR, agent))
            model_file = [p for p in model_path.iterdir() if p.is_file()][0]
            proper_model_file = model_picker(model_file, farm_dict_by_year[season_years[0]])
            if getattr(proper_model_file["args"], 'special_action_space', False):
                env.override_action_space()
            runner = MultiRLAgent(
                env=env,
                saved_model=proper_model_file,
                render=render,
            )
        elif agent == "ROT":
            runner = RoTAgent(
                env=env,
                render=render,
            )
        elif agent == "random":
            runner = RandomAgent(env=env, render=render)
        else:
            raise ValueError(f"Unknown agent: {agent}")

        # Reset env with multi-season campaign
        env.reset(options={
            "year": int(season_years[0]),
            "eval_horizon_years": season_years,
            "farm_dict_by_year": farm_dict_by_year,
            "preseason_allocation": True,
            "days_before_sowing": 7,
        })

        info_dict = {}

        next_states = None
        for sy in season_years:
            # Advance fields to allocation date for this season
            env.advance_fields_to_allocation_dates(
                days_before_sowing=7,
                season_year=int(sy),
                farm_dict_by_year=farm_dict_by_year,
            )
            # Allocation logic
            if allocator is None and "reduced" in scenario:
                reductions = [
                    (env.get_per_parcel_max_budget(a) - (env.get_per_parcel_max_budget(a) * 0.7)) / 10.0
                    for a in env.possible_agents
                ]
                env.allocate_bandit_budgets(reductions)
                for ag in env.possible_agents:
                    assert env.get_per_parcel_budget(ag) < env.get_per_parcel_max_budget(ag)


            if allocator is not None and "LP" in allocator:
                if lp_df is None or global_farm_id is None:
                    raise RuntimeError("LP allocator requested but lp_df/global_farm_id not initialized")

                farm_budget = float(getattr(env, "global_budget", env._get_global_max_budget()))
                if allocator == "LP_reduced" or ("reduced" in scenario):
                    farm_budget = 0.7 * farm_budget

                # Path-B discrete benchmark: pick the minimum-N feasible grid point per field
                # subject to NUE/Nsurp bands. Returns reductions in 10 kg/ha units.
                reductions_vec, lp_diag = solve_discrete_lp_for_env(
                    env,
                    df_metrics=lp_df,
                    farm_id=global_farm_id,
                    year=int(sy),
                    total_budget=farm_budget,
                    nue_range=(0.5, 0.9),
                    nsurp_range=(0.0, 80.0),
                )
                env.allocate_bandit_budgets(reductions_vec)

                # Optional debug print
                print(f"[LP-grid] {global_farm_id} year={sy} feasible={lp_diag.feasible} reason={lp_diag.reason} totalN={lp_diag.total_N:.2f}")
            # Step env until season completes
            if runner.__class__.__name__ == "MultiRLAgent":
                next_states = env.run_until_past_season_year(
                    season_year=int(sy),
                    env_agent=runner,
                    next_states=next_states,
                )
            else:
                env.run_til_past_season_year(
                    season_year=int(sy),
                )

            env.set_print_season_year(sy)
            print(env)

            # Collect per-season infos
            info_dict[int(sy)] = env.collect_agent_infos_for_season(int(sy))
        info = info_dict


        if info is None:
            print(f"No results for farmer_{i} at {region} in years {year_tag}")
        del runner
        with open(out_path, "wb") as f:
            pickle.dump(info, f)
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
    return {}


# Helper for parallel execution
def _run_region_year_wrapper(args):
    region, years, agent, scenario, allocator, subset, render = args
    info_dict = run_region_year(region, years, agent=agent, scenario=scenario, allocator=allocator, subset=subset, render=render)
    return region, years, info_dict

# Optional aggregation controlled by --aggregate/--aggregate_only
# Aggregation helper for multi-season pickles
def _aggregate_multi_season_pickles(*, agent: str, scenario: str, allocator: str | None, regions: list[str], year_tag: str, subset: bool) -> Path:
    base_dir = Path(_DEFAULT_RESULTSDIR) / agent
    name_al = "" if allocator is None else f"_{allocator}"

    aggregated_results = {}
    for region in regions:
        if subset:
            pattern = f"results_{scenario}_{region}_{year_tag}{name_al}_subset_farmer_*.pkl"
        else:
            pattern = f"results_{scenario}_{region}_{year_tag}{name_al}_farmer_*.pkl"

        for pkl_file in base_dir.glob(pattern):
            # keep farmer tag for clarity: farmer_0
            stem_parts = pkl_file.stem.split("_")
            farmer_tag = "_".join(stem_parts[-2:])  # e.g. farmer_0
            # include allocator tag in the aggregated key so different allocators don't collide
            # `name_al` is "" or f"_{allocator}" and matches the per-farmer filename convention
            key = f"{region}_{year_tag}{name_al}_{farmer_tag}"
            if subset:
                key = f"{region}_{year_tag}{name_al}_subset_{farmer_tag}"

            with open(pkl_file, "rb") as f:
                temp_dict = pickle.load(f)

            aggregated_results[key] = temp_dict

    out_name = f"results_{agent}_{scenario}{name_al}_{year_tag}_AGG.pkl"
    if subset:
        out_name = f"results_{agent}_{scenario}{name_al}_{year_tag}_subset_AGG.pkl"

    out_path = base_dir / out_name
    with open(out_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Saved aggregated results to {out_path}")
    return out_path

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
    parser.add_argument("--aggregate", action="store_true", help="combine per-farmer multi-season pickles into one file")
    parser.add_argument("--aggregate_only", action="store_true", help="only aggregate existing per-farmer pickles and exit")
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
        years_list = [2020, 2021, 2022, 2023, 2024]
    else:
        years_list = [years]

    if subset:
        years_list = [2020]

    # Compute year_tag as in run_region_year
    year_tag = f"{years_list[0]}-{years_list[-1]}" if len(years_list) > 1 else str(years_list[0])

    # Each job runs a full multi-season horizon for the region
    all_jobs = [(region, years_list, agent, scenario, allocator, subset, args.render) for region in regions]
    sliced_jobs = [all_jobs[i:i+3] for i in range(0, len(all_jobs), 3)]

    # Early exit for --aggregate_only
    if args.aggregate_only:
        _aggregate_multi_season_pickles(
            agent=agent,
            scenario=scenario,
            allocator=allocator,
            regions=regions,
            year_tag=year_tag,
            subset=subset,
        )
        raise SystemExit(0)

    if num_workers is None or num_workers <= 1:
        # Fallback to sequential execution
        for region, years_job, agent, scenario, allocator, subset_job, render_job in tqdm(all_jobs, desc="Running scenarios"):
            run_region_year(
                region,
                years_job,
                agent=agent,
                scenario=scenario,
                allocator=allocator,
                subset=subset_job,
                render=render_job,
            )
    else:
        # Parallel execution over regions/multi-years
        for jobs in tqdm(sliced_jobs, desc="Slicing jobs"):
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_run_region_year_wrapper, job): job
                    for job in jobs
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="-----Running scenarios------"):
                    region, years_job, info_dict = future.result()

    if args.aggregate:
        _aggregate_multi_season_pickles(
            agent=agent,
            scenario=scenario,
            allocator=allocator,
            regions=regions,
            year_tag=year_tag,
            subset=subset,
        )
