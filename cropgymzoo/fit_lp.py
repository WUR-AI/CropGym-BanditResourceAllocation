import os.path

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable

import argparse

import pickle
import re

from pprint import pprint

from scipy.optimize import linprog

from cropgymzoo import _DEFAULT_RESULTSDIR


@dataclass
class FieldResponse:
    alpha: float
    beta: float
    r2: float
    n_points: int


def fit_alpha_for_fields(
    sim_df: pd.DataFrame,
    train_years: List[int] = None,
    min_points: int = 1,
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
        if g["N"].nunique() < min_points:
            continue

        N = g["N"].to_numpy()
        R = g["reward"].to_numpy()

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
    bmax: np.ndarray,
    B: float,
) -> np.ndarray:
    """
    Solve: max sum_i alpha_i * N_i
           s.t. sum_i N_i <= B, 0 <= N_i <= bmax_i

    Returns
    -------
    N_opt : np.ndarray
        Optimal seasonal N per field (same length as alpha).
    """
    alpha = np.asarray(alpha, dtype=float)
    bmax = np.asarray(bmax, dtype=float)

    n_fields = len(alpha)
    assert bmax.shape == alpha.shape

    # linprog minimizes, so we minimize -alpha^T N
    c = -alpha

    # Constraint: sum_i N_i <= B
    A_ub = np.ones((1, n_fields))
    b_ub = np.array([B], dtype=float)

    # Bounds: 0 <= N_i <= bmax_i
    bounds = [(0.0, float(bi)) for bi in bmax]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    N_opt = res.x
    return N_opt

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
        bmax_list: list[float] = []
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
            bmax_list.append(row["bmax"])
            areas_list.append(row["area"])

        if not field_ids:
            # no usable fields for this farm-year
            continue

        alpha_vec = np.array(alphas, dtype=float)
        bmax_vec = np.array(bmax_list, dtype=float)
        area_vec = np.array(areas_list, dtype=float)

        # Solve LP in absolute kg
        N_opt = lp_allocate_for_farm(alpha_vec, bmax_vec, B)

        # Allocation scaled to area: kg/ha
        N_per_ha = N_opt / area_vec

        # Allocation scaled to rate: fraction of nominal farm_rate
        frac_of_rate = N_per_ha / farm_rate

        results[(farm_id, year)] = FarmAllocationResult(
            farm_id=farm_id,
            year=year,
            field_ids=list(field_ids),
            N_opt=N_opt,
            alpha=alpha_vec,
            area=area_vec,
            N_per_ha=N_per_ha,
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

            # field-level maximum N (kg)
            bmax_abs = area_ha * budget_rate  # kg
            per_field[field_id] = {
                "area": area_ha,
                "bmax": bmax_abs,
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
                "bmax": info["bmax"],          # absolute kg allowed per field
                "budget": farm_budget,         # absolute kg allowed per farm
                "rate_budget": farm_rate,      # kg/ha budget rate
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


def save_lp_results(alpha_dict, lp_results, red: bool = False):
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
            "N_opt": res.N_opt,
            "N_per_ha": res.N_per_ha,
            "frac_of_rate": res.frac_of_rate,
            "alpha": res.alpha,
            "area": res.area,
        }

    path = os.path.join(_DEFAULT_RESULTSDIR, "LP")
    os.makedirs(os.path.join(path), exist_ok=True)
    if red:
        name = "lp_results_reduced.pkl"
    else:
        name = "lp_results.pkl"
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump(info, f)

    print(f"[saved] LP baseline stored in {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, help="path to simulation pickle file")

    args = parser.parse_args()

    with open(args.file_path, "rb") as f:
        data = pickle.load(f)

    df_lp = make_df_lp(data)

    # get alpha per field
    responses = fit_alpha_for_fields(df_lp, train_years=[2015, 2016, 2017, 2018, 2019])

    farm_info = make_farm_info(data)

    lp_results = lp_allocate_for_all_farms(
        responses=responses,
        farm_info=farm_info,  # has farm_id, field_id, bmax, budget
    )

    reduced = True if "reduced" in args.file_path else False
    save_lp_results(responses, lp_results, reduced)

    print(lp_results)
    # df = pd.DataFrame(lp_results, )
    #
    # os.makedirs(os.path.join(_DEFAULT_RESULTSDIR, "LP"), exist_ok=True)
    #
    # df.to_csv(os.path.join(_DEFAULT_RESULTSDIR, "LP", "lp_results_all.csv"), index=False)
