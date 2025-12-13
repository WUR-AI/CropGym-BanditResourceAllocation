import os

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable

from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse

import pickle
import re

from pprint import pprint

from scipy.optimize import linprog

from pathlib import Path

from cropgymzoo import _DEFAULT_RESULTSDIR, _DEFAULT_MODEL_DIR, _SCENARIO_PATH
from cropgymzoo.eval_policy import MultiRLAgent, RoTAgent, RandomAgent
from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.scenario_utils import model_picker

import hashlib
import yaml
from tqdm import tqdm


def _precompute_lp_spsa_for_farmer_worker(kwargs: dict) -> dict:
    return precompute_lp_spsa_for_farmer(**kwargs)


def _spsa_cache_dir(results_dir: str) -> Path:
    d = Path(results_dir) / "LP" / "spsa_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _spsa_cache_path(
    results_dir: str,
    region: str,
    farmer_idx: int,
    eval_year: int,
    sim_year: int,
    scenario: str,
    agent: str,
    delta: float,
) -> Path:
    raw = f"{region}|{farmer_idx}|{eval_year}|{sim_year}|{scenario}|{agent}|{float(delta):.6f}"
    key = hashlib.md5(raw.encode("utf-8")).hexdigest()
    fname = f"spsa_{region}_farmer_{farmer_idx}_eval{eval_year}_sim{sim_year}_{scenario}_{agent}_{key}.pkl"
    fname = fname.replace(os.sep, "_")
    return _spsa_cache_dir(results_dir) / fname


# Precompute SPSA-based LP allocation for a single farmer and cache the result
def precompute_lp_spsa_for_farmer(
    *,
    region: str,
    eval_year: int,
    sim_year: int,
    farmer_idx: int,
    scenario: str,
    agent: str,
    delta: float = 10.0,
    render: bool = False,
    results_dir: str = _DEFAULT_RESULTSDIR,
) -> dict:
    """Compute (or load cached) SPSA alpha_hat and LP allocation for one farmer."""

    cache_path = _spsa_cache_path(
        results_dir,
        region,
        int(farmer_idx),
        int(eval_year),
        int(sim_year),
        str(scenario),
        str(agent),
        float(delta),
    )

    # If cached, return it
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached
        except Exception as e:
            print(f"[LP_SPSA] Cache read failed ({cache_path}): {e}. Recomputing.")

    # Load farmer yaml
    farmer_path = Path(_SCENARIO_PATH) / region / str(eval_year) / f"farmer_{farmer_idx}.yaml"
    with open(farmer_path, "r") as f:
        dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

    # Build env at the simulation year
    env = MultiFieldEnv(years=[sim_year], training=False, render=render, farm_dict=dict_fields)
    field_order = list(env.possible_agents)

    base_budgets = np.array([env.get_per_parcel_budget(ag) for ag in field_order], dtype=float)
    max_budgets = np.array([env.get_per_parcel_max_budget(ag) for ag in field_order], dtype=float)
    areas = np.array([env.get_per_field_area()[ag] for ag in field_order], dtype=float)

    B_abs = float(np.sum(areas * base_budgets))
    spsa_seed = 12345 + int(farmer_idx) + int(sim_year) * 1000

    alpha_hat = _estimate_alpha_spsa(
        agent=agent,
        dict_fields=dict_fields,
        year=sim_year,
        base_budgets=base_budgets,
        max_budgets=max_budgets,
        areas=areas,
        delta=float(delta),
        seed=int(spsa_seed),
        render=render,
    )

    N_opt_ha = _solve_lp_from_alpha(alpha_hat, areas=areas, bmax_rate=max_budgets, B_abs=B_abs)

    payload = {
        "region": region,
        "farmer_idx": int(farmer_idx),
        "eval_year": int(eval_year),
        "sim_year": int(sim_year),
        "scenario": str(scenario),
        "agent": str(agent),
        "delta": float(delta),
        "field_order": list(field_order),
        "alpha_hat": np.asarray(alpha_hat, dtype=float),
        "N_opt_ha": np.asarray(N_opt_ha, dtype=float),
        "B_abs": float(B_abs),
        "base_budgets": np.asarray(base_budgets, dtype=float),
        "max_budgets": np.asarray(max_budgets, dtype=float),
        "areas": np.asarray(areas, dtype=float),
    }

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
    except Exception as e:
        print(f"[LP_SPSA] Cache write failed ({cache_path}): {e}")

    return payload


# Precompute SPSA-based LP allocations for all farmers in regions/years and write to lp_results_{agent}.pkl
def precompute_lp_spsa(
    *,
    agent: str,
    regions: list,
    years: list,
    scenario: str = "full_budget",
    delta: float = 10.0,
    subset: bool = False,
    render: bool = False,
    results_dir: str = _DEFAULT_RESULTSDIR,
    num_workers: int = 1,
) -> dict:
    """Precompute SPSA-based LP allocations and store them in the same format as save_lp_results()."""

    info = {
        "meta": {
            "method": "LP_SPSA",
            "agent": str(agent),
            "scenario": str(scenario),
            "delta": float(delta),
            "years": list(years),
            "regions": list(regions),
        },
        "alphas": {},
        "lp_allocations": {},
    }

    for region in regions:
        for eval_year in years:
            sim_year = eval_year - 5 if "-lp" in scenario else eval_year

            year_path = Path(_SCENARIO_PATH) / region / str(eval_year)
            if not year_path.exists():
                print(f"[LP_SPSA] Missing scenario folder: {year_path}")
                continue

            farmer_files = sorted([p for p in year_path.glob("farmer_*.yaml")])
            if subset:
                farmer_files = farmer_files[:2]

            farmer_idxs = [int(fp.stem.split("_")[-1]) for fp in farmer_files]

            worker_kwargs = [
                dict(
                    region=region,
                    eval_year=int(eval_year),
                    sim_year=int(sim_year),
                    farmer_idx=int(i),
                    scenario=str(scenario),
                    agent=str(agent),
                    delta=float(delta),
                    render=bool(render),
                    results_dir=str(results_dir),
                )
                for i in farmer_idxs
            ]

            desc = f"LP_SPSA {region}-eval{eval_year} (sim:{sim_year})"

            if num_workers is None or int(num_workers) <= 1:
                iterator = tqdm(worker_kwargs, desc=desc)
                payload_iter = (precompute_lp_spsa_for_farmer(**kw) for kw in iterator)
            else:
                max_workers = int(num_workers)
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_precompute_lp_spsa_for_farmer_worker, kw) for kw in worker_kwargs]
                    done_iter = tqdm(as_completed(futures), total=len(futures), desc=desc)
                    payload_iter = (f.result() for f in done_iter)

            for payload in payload_iter:
                farmer_idx = int(payload["farmer_idx"])
                farm_id = f"{region}_farmer_{farmer_idx}"
                info["lp_allocations"].setdefault(farm_id, {})

                field_ids = payload["field_order"]
                alpha_hat = np.asarray(payload["alpha_hat"], dtype=float)
                N_opt_ha = np.asarray(payload["N_opt_ha"], dtype=float)
                areas = np.asarray(payload["areas"], dtype=float)
                bmax_rate = np.asarray(payload["max_budgets"], dtype=float)

                for fid, a in zip(field_ids, alpha_hat):
                    info["alphas"][(farm_id, fid)] = FieldResponse(
                        alpha=float(a), beta=float("nan"), r2=float("nan"), n_points=2
                    )

                info["lp_allocations"][farm_id][int(eval_year)] = {
                    "field_ids": list(field_ids),
                    "N_opt_ha": N_opt_ha,
                    "N_opt_abs": N_opt_ha * areas,
                    "frac_of_rate": (
                            N_opt_ha / (
                        float(np.nanmax(payload["base_budgets"])) if len(payload["base_budgets"]) else 1.0)
                    ),
                    "alpha": alpha_hat,
                    "area": areas,
                    "bmax_rate": bmax_rate,
                }

    # Save
    out_dir = Path(results_dir) / "LP"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"lp_results_{agent}.pkl"
    if "reduced" in scenario:
        out_name = f"lp_results_reduced_{agent}.pkl"

    out_path = out_dir / out_name
    with open(out_path, "wb") as f:
        pickle.dump(info, f)

    print(f"[saved] LP_SPSA allocations stored in {out_path}")
    return info


@dataclass
class FieldResponse:
    alpha: float
    beta: float
    r2: float
    n_points: int


def _total_profit_from_info(info: dict, year: int) -> float:
    """Robustly extract farm total profit from runner output."""
    if info is None:
        return float("nan")

    # info sometimes is {year: {field_id: infos}} or directly {field_id: infos}
    year_blob = info.get(year, info)
    if not isinstance(year_blob, dict):
        return float("nan")

    total = 0.0
    for _, field_infos in year_blob.items():
        try:
            prof = field_infos.get("Profit", None)
            if prof is None:
                continue
            if isinstance(prof, (list, tuple, np.ndarray)) and len(prof) > 0:
                total += float(prof[-1])
            else:
                total += float(prof)
        except Exception:
            continue
    return float(total)


def _solve_lp_from_alpha(alpha: np.ndarray, areas: np.ndarray, bmax_rate: np.ndarray, B_abs: float) -> np.ndarray:
    """LP in kg/ha (rate): max alpha^T N s.t. sum area*N <= B_abs, 0<=N<=bmax."""
    alpha = np.asarray(alpha, dtype=float)
    areas = np.asarray(areas, dtype=float)
    bmax_rate = np.asarray(bmax_rate, dtype=float)

    n = len(alpha)
    c = -alpha  # linprog minimizes
    A_ub = areas.reshape(1, n)
    b_ub = np.array([float(B_abs)], dtype=float)
    bounds = [(0.0, float(bi)) for bi in bmax_rate]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x


def _make_env_for_farmer(dict_fields: dict, year: int, render: bool = False) -> MultiFieldEnv:
    env = MultiFieldEnv(
        years=[year],
        training=False,
        render=render,
        farm_dict=dict_fields,
    )
    return env


def _make_runner_for_agent(agent: str, env: MultiFieldEnv, dict_fields: dict, render: bool = False):
    """Create the correct runner object for the provided agent."""
    if "MLP" in agent:
        model_path = Path(os.path.join(_DEFAULT_MODEL_DIR, agent))
        model_file = [p for p in model_path.iterdir() if p.is_file()][0]
        proper_model_file = model_picker(model_file, dict_fields)

        if getattr(proper_model_file["args"], 'special_action_space', False):
            env.override_action_space()

        return MultiRLAgent(env=env, saved_model=proper_model_file, render=render)

    if agent == "ROT":
        return RoTAgent(env=env, render=render)

    if agent == "random":
        return RandomAgent(env=env, render=render)

    raise ValueError(f"Unknown agent {agent}")


def _estimate_alpha_spsa(
    agent: str,
    dict_fields: dict,
    year: int,
    base_budgets: np.ndarray,
    max_budgets: np.ndarray,
    areas: np.ndarray,
    delta: float,
    seed: int,
    render: bool = False,
) -> np.ndarray:
    """Two-rollout SPSA estimate of per-field marginal values (alpha)."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=len(base_budgets))

    b_plus = np.clip(base_budgets + delta * s, 0.0, max_budgets)
    b_minus = np.clip(base_budgets - delta * s, 0.0, max_budgets)

    # Run +
    env_p = _make_env_for_farmer(dict_fields, year=year, render=render)
    for ag, b in zip(env_p.possible_agents, b_plus):
        env_p.set_per_parcel_budget(ag, float(b))
    runner_p = _make_runner_for_agent(agent, env_p, dict_fields, render=render)
    info_p = runner_p.run(years=[year])
    Jp = _total_profit_from_info(info_p, year)
    del runner_p
    del env_p

    # Run -
    env_m = _make_env_for_farmer(dict_fields, year=year, render=render)
    for ag, b in zip(env_m.possible_agents, b_minus):
        env_m.set_per_parcel_budget(ag, float(b))
    runner_m = _make_runner_for_agent(agent, env_m, dict_fields, render=render)
    info_m = runner_m.run(years=[year])
    Jm = _total_profit_from_info(info_m, year)
    del runner_m
    del env_m

    # SPSA gradient estimate
    # alpha_i ~= ((J+ - J-) / (2*delta)) * s_i
    ghat = ((Jp - Jm) / (2.0 * float(delta))) * s
    return ghat.astype(float)


def fit_alpha_for_fields(
    sim_df: pd.DataFrame,
    train_years: List[int] = None,
    min_points: int = 2,
) -> Dict[Tuple[Hashable, Hashable], FieldResponse]:
    """
    Fit linear response R_i(N) ~= alpha_i * N + beta_i for each (farm_id, field_id).

    Parameters
    ----------
    sim_df : DataFrame
        Must contain columns: ["farm_id", "field_id", "N", "reward", "year"].
        Each row = one simulation: a given field with total seasonal N and resulting reward.
    train_years : list of int, optional
        If provided, we restrict to these years for fitting.
    min_points : int
        Minimum number of distinct N values required to fit a line.

    Returns
    -------
    dict[(farm_id, field_id) -> FieldResponse]
    """
    df = sim_df.copy()

    if train_years is not None:
        df = df[df["year"].isin(train_years)]

    responses: Dict[Tuple[Hashable, Hashable], FieldResponse] = {}

    grouped = df.groupby(["farm_id", "field_id"])
    for (farm_id, field_id), g in grouped:
        # We want at least some spread in N
        g = g.sort_values("N")
        # Need at least 2 distinct N values (and some spread) to fit a line reliably
        if g["N"].nunique() < min_points:
            continue

        N = g["N"].to_numpy(dtype=float)
        R = g["reward"].to_numpy(dtype=float)

        # Guard against degenerate fits (all N nearly identical)
        if np.nanmax(N) - np.nanmin(N) < 1e-6:
            continue

        # Fit R ~ alpha * N + beta
        # np.polyfit returns [slope, intercept]
        alpha, beta = np.polyfit(N, R, deg=1)

        # Compute a simple R^2 for diagnostics
        R_pred = alpha * N + beta
        ss_res = np.sum((R - R_pred) ** 2)
        ss_tot = np.sum((R - R.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        responses[(farm_id, field_id)] = FieldResponse(
            alpha=float(alpha),
            beta=float(beta),
            r2=float(r2),
            n_points=len(g),
        )

    return responses


def lp_allocate_for_farm(
    alpha: np.ndarray,
    area: np.ndarray,
    bmax_rate: np.ndarray,
    B_abs: float,
) -> np.ndarray:
    """
    Solve in kg/ha (rate) to match the fitted response R_i(N_ha).

      max   sum_i alpha_i * N_i_ha
      s.t.  sum_i area_i * N_i_ha <= B_abs
            0 <= N_i_ha <= bmax_rate_i

    Parameters
    ----------
    alpha : slope per field (profit change per +1 kg/ha)
    area : area per field in ha
    bmax_rate : per-field max rate in kg/ha
    B_abs : farm total budget in kg (absolute)

    Returns
    -------
    N_opt_ha : np.ndarray
        Optimal seasonal N rate (kg/ha) per field.
    """
    alpha = np.asarray(alpha, dtype=float)
    area = np.asarray(area, dtype=float)
    bmax_rate = np.asarray(bmax_rate, dtype=float)

    n_fields = len(alpha)
    assert area.shape == alpha.shape
    assert bmax_rate.shape == alpha.shape

    # linprog minimizes, so minimize -alpha^T N_ha
    c = -alpha

    # Constraint: sum_i area_i * N_i_ha <= B_abs
    A_ub = area.reshape(1, n_fields)
    b_ub = np.array([float(B_abs)], dtype=float)

    # Bounds: 0 <= N_i_ha <= bmax_rate_i
    bounds = [(0.0, float(bi)) for bi in bmax_rate]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    return res.x

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


def lp_allocate_for_all_farms(
    responses: Dict[Tuple[Hashable, Hashable], FieldResponse],
    farm_info: pd.DataFrame,
) -> Dict[Tuple[Hashable, int], FarmAllocationResult]:
    """
    For each farm, solve an LP based on the fitted alpha_i and bmax_i.

    Parameters
    ----------
    responses : dict
        Mapping (farm_id, field_id) -> FieldResponse (alpha, beta, etc.)
    farm_info : DataFrame
        Must contain columns: ["farm_id", "field_id", "year", "area", "bmax", "budget", "rate_budget"].
        budget and rate_budget are per farm-year (same value repeated per field is fine).

    Returns
    -------
    dict[(farm_id, year) -> FarmAllocationResult]
    """
    results: Dict[Tuple[Hashable, int], FarmAllocationResult] = {}

    # group by farm and year
    for (farm_id, year), g in farm_info.groupby(["farm_id", "year"]):
        field_ids: list[Hashable] = []
        alphas: list[float] = []
        bmax_rate_list: list[float] = []
        areas_list: list[float] = []

        # farm-level budget: assume same for all rows in this group
        B_vals = g["budget"].unique()
        if len(B_vals) != 1:
            raise ValueError(
                f"Farm {farm_id} in year {year} has multiple budget values in farm_info."
            )
        B = float(B_vals[0])

        # farm-level rate: assume same for all rows in this group
        rate_vals = g["rate_budget"].unique()
        if len(rate_vals) != 1:
            raise ValueError(
                f"Farm {farm_id} in year {year} has multiple rate_budget values in farm_info."
            )
        farm_rate = float(rate_vals[0])  # kg/ha

        for _, row in g.iterrows():
            key = (farm_id, row["field_id"])
            if key not in responses:
                # This field has no fitted response; skip it
                continue

            fr = responses[key]
            field_ids.append(row["field_id"])
            alphas.append(fr.alpha)
            bmax_rate_list.append(row["bmax_rate"])
            areas_list.append(row["area"])

        if not field_ids:
            # no usable fields for this farm-year
            continue

        alpha_vec = np.array(alphas, dtype=float)
        bmax_rate_vec = np.array(bmax_rate_list, dtype=float)
        area_vec = np.array(areas_list, dtype=float)

        # Solve LP in kg/ha (rate), constrained by absolute farm budget in kg
        N_opt_ha = lp_allocate_for_farm(alpha_vec, area_vec, bmax_rate_vec, B_abs=B)

        # Convert to absolute kg per field
        N_opt_abs = N_opt_ha * area_vec

        # Allocation scaled to rate: fraction of nominal farm_rate
        frac_of_rate = N_opt_ha / farm_rate

        results[(farm_id, year)] = FarmAllocationResult(
            farm_id=farm_id,
            year=year,
            field_ids=list(field_ids),
            N_opt_ha=N_opt_ha,
            N_opt_abs=N_opt_abs,
            alpha=alpha_vec,
            area=area_vec,
            bmax_rate=bmax_rate_vec,
            frac_of_rate=frac_of_rate,
        )

    return results

def make_farm_info(data: dict):
    farm_rows = []

    def parse_farm_key(key: str):
        # Example: "groningen_2019_farmer_0"
        m = re.match(r"(.+?)_(\d+)_farmer_(\d+)", key)
        if m:
            region, year, farmer = m.groups()
            return region, int(year), int(farmer)
        return None, None, None

    for farm_key, farm_dict in data.items():
        region, year, farmer_id = parse_farm_key(farm_key)
        farm_id = f"{region}_farmer_{farmer_id}"

        per_field = {}
        areas = []
        rates = []

        # ---------- FIRST PASS ----------
        for field_id, field_data in farm_dict.items():
            # extract area
            area_list = field_data.get("area", None)
            if area_list is None or len(area_list) == 0:
                continue
            area_ha = float(area_list[0])  # area is constant over time
            areas.append(area_ha)

            # extract BudgetTotal (rate in kg/ha)
            budget_rate = field_data.get("BudgetTotal", None)
            if isinstance(budget_rate, (list, np.ndarray)):
                if len(budget_rate) > 0:
                    budget_rate = float(budget_rate[-1])
                else:
                    budget_rate = None
            if budget_rate is None:
                continue

            rates.append(budget_rate)

            # field-level maximum N (rate in kg/ha)
            per_field[field_id] = {
                "area": area_ha,
                "bmax_rate": float(budget_rate),
            }

        if len(per_field) == 0:
            continue

        # ---------- FARM-LEVEL BUDGET ----------
        if len(set(rates)) != 1:
            print(f"Warning: farm {farm_id} year {year} has multiple budget rates: {set(rates)}")

        farm_rate = max(rates)          # kg/ha
        total_area = sum(areas)
        farm_budget = farm_rate * total_area   # kg total N allowed for farm

        # ---------- SECOND PASS ----------
        for field_id, info in per_field.items():
            farm_rows.append({
                "farm_id": farm_id,
                "field_id": field_id,
                "region": region,
                "year": year,
                "area": info["area"],
                "bmax_rate": info["bmax_rate"],   # kg/ha allowed per field
                "budget": farm_budget,             # absolute kg allowed per farm
                "rate_budget": farm_rate,          # kg/ha farm budget rate
            })

    farm_info = pd.DataFrame(farm_rows)
    return farm_info

def make_df_lp(data: dict):
    rows = []

    def parse_farm_key(key: str):
        # Example: "groningen_2020_farmer_8"
        m = re.match(r"(.+?)_(\d+)_farmer_(\d+)", key)
        if m:
            region, year, farmer = m.groups()
            return region, int(year), int(farmer)
        return None, None, None

    for farm_key, farm_dict in data.items():
        region, year, farmer_id = parse_farm_key(farm_key)
        farm_id = f"{region}_farmer_{farmer_id}"

        for field_id, field_data in farm_dict.items():

            N_values = field_data["Naction"]
            if len(N_values) == 0:
                continue
            N_season = float(N_values[-1])

            # Extract final reward (from RL)
            reward_values = field_data["Profit"]
            reward_final = float(reward_values[-1]) if len(reward_values) > 0 else np.nan

            # Crop (optional)
            crop = field_data.get("CropName", ["unknown"])[-1]

            rows.append({
                "farm_id": farm_id,
                "field_id": field_id,
                "year": year,
                "N": N_season,
                "reward": reward_final,
                "crop": crop,
                "region": region
            })

    df_lp = pd.DataFrame(rows)

    return df_lp


def save_lp_results(alpha_dict, lp_results, model_name, red: bool = False):
    """
    Save alpha parameters and LP allocation results to disk.
    alpha_dict:  {(farm_id, field_id): alpha}
    lp_results:  output of lp_allocate_for_all_farms()
    """
    info = {
        "alphas": alpha_dict,
        "lp_allocations": {},
    }

    # Structure LP results as farm → year → data
    for (farm_id, year), res in lp_results.items():
        if farm_id not in info["lp_allocations"]:
            info["lp_allocations"][farm_id] = {}

        info["lp_allocations"][farm_id][year] = {
            "field_ids": res.field_ids,
            "N_opt_ha": res.N_opt_ha,
            "N_opt_abs": res.N_opt_abs,
            "frac_of_rate": res.frac_of_rate,
            "alpha": res.alpha,
            "area": res.area,
            "bmax_rate": res.bmax_rate,
        }

    path = os.path.join(_DEFAULT_RESULTSDIR, "LP")
    os.makedirs(os.path.join(path), exist_ok=True)
    if red:
        name = f"lp_results_reduced_{model_name}.pkl"
    else:
        name = f"lp_results_{model_name}.pkl"
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump(info, f)

    print(f"[saved] LP baseline stored in {path}")

def get_model_name(file_path: str) -> str:
    p = os.path.basename(file_path).upper()  # or use full path if you want
    candidates = ["MLP_FOCOPS", "MLP_PCPO", "MLP_LAGPPO", "ROT"]  # specific -> broad
    for c in candidates:
        if c in p:
            return c
    raise ValueError(f"Unknown model name for file {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Old mode (offline fit from a simulation pickle)
    parser.add_argument("--file_path", type=str, default=None, help="path to simulation pickle file")

    # New mode (SPSA precompute)
    parser.add_argument("--mode", type=str, default="offline_fit", choices=["offline_fit", "spsa_precompute"], help="which LP fitting mode to run")
    parser.add_argument("--agent", type=str, default="MLP_FOCOPS", help="agent name (e.g., MLP_FOCOPS)")
    parser.add_argument("--regions", type=str, default="all", help="region name or 'all'")
    parser.add_argument("--years", type=int, default=0, help="year or 0 for default test years")
    parser.add_argument("--scenario", type=str, default="full_budget", help="scenario name (supports '-lp')")
    parser.add_argument("--delta", type=float, default=10.0, help="SPSA perturbation (kg/ha)")
    parser.add_argument("--subset", action='store_true', dest='subset')
    parser.add_argument("--render", action='store_true', dest='render')
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Processes for SPSA precompute (1 = serial)")
    parser.set_defaults(render=False, subset=False)

    args = parser.parse_args()

    if args.mode == "offline_fit":
        if args.file_path is None:
            raise ValueError("--file_path is required in offline_fit mode")

        model_name = get_model_name(args.file_path)

        with open(args.file_path, "rb") as f:
            data = pickle.load(f)

        df_lp = make_df_lp(data)
        responses = fit_alpha_for_fields(df_lp, train_years=[2015, 2016, 2017, 2018, 2019])
        farm_info = make_farm_info(data)
        lp_results = lp_allocate_for_all_farms(responses=responses, farm_info=farm_info)

        reduced = True if "reduced" in args.file_path else False
        save_lp_results(responses, lp_results, model_name, reduced)

    else:
        regions = args.regions
        years = args.years

        if regions == "all":
            regions = ["groningen", "zeeland", "gelderland"]
        else:
            regions = [regions]

        if years == 0:
            years = [2020, 2021, 2022, 2023, 2024]
        else:
            years = [years]

        if args.subset:
            years = [years[0]]

        precompute_lp_spsa(
            agent=args.agent,
            regions=regions,
            years=years,
            scenario=args.scenario,
            delta=float(args.delta),
            subset=bool(args.subset),
            render=bool(args.render),
            results_dir=_DEFAULT_RESULTSDIR,
            num_workers=args.num_workers
        )
