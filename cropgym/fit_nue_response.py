import os
import re
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Hashable, Optional, Any
import copy

import gc
import tracemalloc
import resource

from concurrent.futures import ProcessPoolExecutor, as_completed

import pulp

import numpy as np
import pandas as pd

from cropgym import _DEFAULT_RESULTSDIR


# ======================================================================================
# NUE / Nsurp response fitting utilities
# ======================================================================================


@dataclass
class LinearMetricResponse:
    """Simple linear response model: metric ~= slope * N + intercept."""

    metric: str
    slope: float
    intercept: float
    r2: float
    n_points: int

    def predict(self, N: float | np.ndarray) -> float | np.ndarray:
        return self.slope * N + self.intercept


@dataclass
class FieldMetricResponses:
    """Container for multiple metric response models for one field."""

    farm_id: Hashable
    field_id: Hashable
    crop: str
    region: str
    models: Dict[str, LinearMetricResponse]  # metric -> model


# --------------------------------------------------------------------------------------
# Parsing + dataframe construction
# --------------------------------------------------------------------------------------


def _parse_farm_key(key: str):
    """Parse farm keys.

    Supported formats:

    1) 'groningen_2020_farmer_8'
        -> (region='groningen', year=2020, farm_global_id=None, farmer_id=8, red_level=None)

    2) 'gelderland_2015_red0_farmer_11'
        -> (region='gelderland', year=2015, farm_global_id=None, farmer_id=11, red_level=0.0)

    3) 'zeeland_2015_red0_farm32_farmer_7'
        -> (region='zeeland', year=2015, farm_global_id=32, farmer_id=7, red_level=0.0)

    Notes
    -----
    - `redX` is the global reduction level applied to all fields (typically kg/ha).
    - `farmK` is a global farm identifier across regions.
    - `farmer_J` is the local farm number within the region.
    """
    s = str(key)

    # Daisy-chain format: region_YYYY-YYYY_redX_farmK_farmer_J
    # Return start year (YYYY); caller can override per-season later.
    m = re.match(r"(.+?)_(\d+)-(\d+)_red(-?\d+(?:\.\d+)?)_farm(\d+)_farmer_(\d+)$", s)
    if m:
        region, year0, year1, red_level, farm_global_id, farmer_id = m.groups()
        return str(region), int(year0), int(farm_global_id), int(farmer_id), float(red_level)

    # Daisy-chain format without global farm id: region_YYYY-YYYY_redX_farmer_J
    m = re.match(r"(.+?)_(\d+)-(\d+)_red(-?\d+(?:\.\d+)?)_farmer_(\d+)$", s)
    if m:
        region, year0, year1, red_level, farmer_id = m.groups()
        return str(region), int(year0), None, int(farmer_id), float(red_level)

    # Newest format: region_year_redX_farmK_farmer_J
    m = re.match(r"(.+?)_(\d+)_red(-?\d+(?:\.\d+)?)_farm(\d+)_farmer_(\d+)$", s)
    if m:
        region, year, red_level, farm_global_id, farmer_id = m.groups()
        return str(region), int(year), int(farm_global_id), int(farmer_id), float(red_level)

    # Mid format: region_year_redX_farmer_J
    m = re.match(r"(.+?)_(\d+)_red(-?\d+(?:\.\d+)?)_farmer_(\d+)$", s)
    if m:
        region, year, red_level, farmer_id = m.groups()
        return str(region), int(year), None, int(farmer_id), float(red_level)

    # Old format: region_year_farmer_J
    m = re.match(r"(.+?)_(\d+)_farmer_(\d+)$", s)
    if m:
        region, year, farmer_id = m.groups()
        return str(region), int(year), None, int(farmer_id), None

    return None, None, None, None, None


def make_df_nue_response(data: dict) -> pd.DataFrame:
    """Create a tidy df with per-field seasonal N and end-season NUE/Nsurp.

    Expected (best-effort) fields in the pickle per field:
      - Naction: list/array (we take last)
      - NUE: list/array (we take last)
      - Nsurp: list/array (we take last)
      - CropName (optional)
      - BudgetTotal (optional, rate kg/ha)

    Supported pickle shapes
    ----------------------
    1) Non-daisy-chain:
        data[farm_key] -> {field_id -> field_data}

    2) Daisy-chain aggregated:
        data[farm_key] -> {season_year(int) -> {field_id -> field_data}}

    Notes
    -----
    - If the key contains a year-range (e.g. '2020-2024'), we still parse a start-year,
      but for daisy-chain aggregated pickles we ALWAYS use the inner `season_year` as
      the row's `year`.

    Returns columns:
      [farm_id, farm_global_id, farmer_id, field_id, year, red_level, region, crop, N, NUE, Nsurp, budget_rate]
    """
    rows: list[dict] = []

    def _last_finite(seq):
        """Return last finite value in seq (or None)."""
        if seq is None:
            return None
        try:
            arr = np.asarray(seq, dtype=float)
        except Exception:
            return None
        if arr.size == 0:
            return None
        mask = np.isfinite(arr)
        if not np.any(mask):
            return None
        return float(arr[np.where(mask)[0][-1]])

    if not isinstance(data, dict):
        return pd.DataFrame(rows)

    for farm_key, farm_dict in data.items():
        region, year_from_key, farm_global_id, farmer_id, red_level = _parse_farm_key(farm_key)
        if region is None:
            continue

        if farm_global_id is not None:
            farm_id = f"farm{int(farm_global_id)}"
        else:
            farm_id = f"{region}_farmer_{int(farmer_id)}"

        if not isinstance(farm_dict, dict) or not farm_dict:
            continue

        # Detect daisy-chain aggregated dict: keys are season years (int)
        if all(isinstance(k, (int, np.integer)) for k in farm_dict.keys()):
            season_items = list(farm_dict.items())
        else:
            season_items = [(int(year_from_key), farm_dict)]

        for season_year, season_fields in season_items:
            if not isinstance(season_fields, dict):
                continue

            for field_id, field_data in season_fields.items():
                if not isinstance(field_data, dict):
                    continue

                # --- seasonal N ---
                N_values = field_data.get("Naction", None)
                N_season = _last_finite(N_values)
                if N_season is None:
                    continue

                # --- NUE ---
                nue_values = field_data.get("NUE", None)
                if nue_values is None:
                    nue_values = field_data.get("Nue", None)
                NUE_final = _last_finite(nue_values)
                if NUE_final is None:
                    NUE_final = np.nan

                # --- Nsurp ---
                nsurp_values = field_data.get("Nsurp", None)
                Nsurp_final = _last_finite(nsurp_values)
                if Nsurp_final is None:
                    Nsurp_final = np.nan

                # --- crop (optional) ---
                crop = "unknown"
                crop_values = field_data.get("CropName", None)
                if crop_values is not None and len(crop_values):
                    crop = str(crop_values[-1])

                # --- optional direct outputs for proxy accounting ---
                n_out = _last_finite(field_data.get("NamountSO", None))
                if n_out is None:
                    n_out = np.nan

                no3_depo = _last_finite(field_data.get("RNO3DEPOSTT", None))
                nh4_depo = _last_finite(field_data.get("RNH4DEPOSTT", None))
                n_depo = float((0.0 if no3_depo is None or not np.isfinite(no3_depo) else no3_depo) +
                               (0.0 if nh4_depo is None or not np.isfinite(nh4_depo) else nh4_depo))

                # --- budget rate (optional) ---
                budget_rate = field_data.get("BudgetTotal", None)
                if isinstance(budget_rate, (list, tuple, np.ndarray)):
                    v = _last_finite(budget_rate)
                    budget_rate = float(v) if v is not None else np.nan
                elif budget_rate is None:
                    budget_rate = np.nan
                else:
                    try:
                        budget_rate = float(budget_rate)
                    except Exception:
                        budget_rate = np.nan

                rows.append(
                    {
                        "farm_id": farm_id,
                        "farm_global_id": float(farm_global_id) if farm_global_id is not None else np.nan,
                        "farmer_id": int(farmer_id),
                        "field_id": field_id,
                        "year": int(season_year),
                        "red_level": float(red_level) if red_level is not None else np.nan,
                        "region": region,
                        "crop": crop,
                        "N": float(N_season),
                        "NUE": float(NUE_final),
                        "Nsurp": float(Nsurp_final),
                        "budget_rate": float(budget_rate),
                        "N_out": float(n_out) if np.isfinite(n_out) else np.nan,
                        "N_depo": float(n_depo),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["N_seed"] = df["crop"].map(_crop_seed_n).astype(float)
    df["N_tot_in"] = (
        pd.to_numeric(df["N"], errors="coerce").fillna(0.0)
        + df["N_seed"].fillna(0.0)
        + df["N_depo"].fillna(0.0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        df["NUE_proxy"] = df["N_out"] / df["N_tot_in"].replace(0.0, np.nan)

    df["Nsurp_proxy"] = df["N_tot_in"] - df["N_out"]

    # Keep legacy names only if missing
    if "NUE" not in df.columns:
        df["NUE"] = df["NUE_proxy"]
    if "Nsurp" not in df.columns:
        df["Nsurp"] = df["Nsurp_proxy"]

    return df


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def _fit_linear(N: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y ~= slope*N + intercept and return (slope, intercept, r2)."""
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)

    # remove NaNs
    mask = np.isfinite(N) & np.isfinite(y)
    N = N[mask]
    y = y[mask]

    if N.size < 2:
        raise ValueError("Need at least 2 points")

    if float(np.nanmax(N) - np.nanmin(N)) < 1e-9:
        raise ValueError("Degenerate N range")

    slope, intercept = np.polyfit(N, y, deg=1)
    y_hat = slope * N + intercept

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return float(slope), float(intercept), float(r2)


def fit_field_metric_responses(
    df: pd.DataFrame,
    train_years: Optional[List[int]] = None,
    min_points: int = 2,
    metrics: Optional[List[str]] = None,
) -> Dict[Tuple[Hashable, Hashable], FieldMetricResponses]:
    """Fit per-field linear response for metrics as a function of N.

    By default fits for ["NUE", "Nsurp"].

    Returns:
      dict[(farm_id, field_id)] -> FieldMetricResponses
    """
    if metrics is None:
        metrics = ["NUE", "Nsurp"]

    dff = df.copy()
    if train_years is not None:
        dff = dff[dff["year"].isin(train_years)]

    out: Dict[Tuple[Hashable, Hashable], FieldMetricResponses] = {}

    for (farm_id, field_id, crop), g in dff.groupby(["farm_id", "field_id", "crop"], sort=False):
        # basic group identity
        region = str(g["region"].iloc[0]) if "region" in g.columns and len(g) else "unknown"
        crop = str(g["crop"].iloc[0]) if "crop" in g.columns and len(g) else "unknown"

        models: Dict[str, LinearMetricResponse] = {}

        # We want to fit each metric vs N using all points in the group
        N = g["N"].to_numpy(dtype=float)

        # If we don't even have enough distinct N values, skip
        if pd.Series(N).nunique() < min_points:
            continue

        for metric in metrics:
            if metric not in g.columns:
                continue
            y = g[metric].to_numpy(dtype=float)

            # Must have enough finite points
            finite = np.isfinite(N) & np.isfinite(y)
            if int(np.sum(finite)) < min_points:
                continue

            try:
                slope, intercept, r2 = _fit_linear(N[finite], y[finite])
            except Exception:
                continue

            models[metric] = LinearMetricResponse(
                metric=str(metric),
                slope=float(slope),
                intercept=float(intercept),
                r2=float(r2),
                n_points=int(np.sum(finite)),
            )

        if not models:
            continue

        out[(farm_id, field_id, crop)] = FieldMetricResponses(
            farm_id=farm_id,
            field_id=field_id,
            crop=crop,
            region=region,
            models=models,
        )

    return out


# --------------------------------------------------------------------------------------
# Constraint helper (optional but handy for LP feasibility)
# --------------------------------------------------------------------------------------


def feasible_N_bounds_from_constraints(
    *,
    nue_model: Optional[LinearMetricResponse],
    nsurp_model: Optional[LinearMetricResponse],
    bmax_rate: float,
    nue_min: Optional[float] = None,
    nue_max: Optional[float] = None,
    nsurp_min: Optional[float] = None,
    nsurp_max: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute a conservative feasible [lb, ub] for N (kg/ha) given linear constraints.

    - If NUE model exists and you specify nue_min/nue_max, enforce:
          nue_min <= a*N + b <= nue_max
    - If Nsurp model exists and you specify nsurp_min/nsurp_max, enforce:
          nsurp_min <= c*N + d <= nsurp_max

    We always intersect with the physical bounds [0, bmax_rate].

    Returns:
      (lb, ub)

    Notes:
    - If constraints are impossible under the fitted model, lb may exceed ub.
    """
    lb, ub = 0.0, float(bmax_rate)

    # NUE band constraint
    if nue_model is not None and (nue_min is not None or nue_max is not None):
        a, b = float(nue_model["slope"]), float(nue_model["intercept"])

        # Helper to intersect with (a*N + b >= v)
        def _intersect_ge(v: float):
            nonlocal lb, ub
            # a*N >= v-b
            rhs = float(v) - b
            if abs(a) < 1e-12:
                # constant model
                if b < float(v):
                    lb, ub = 1.0, 0.0  # infeasible
                return
            x = rhs / a
            if a > 0:
                lb = max(lb, x)
            else:
                ub = min(ub, x)

        # Helper to intersect with (a*N + b <= v)
        def _intersect_le(v: float):
            nonlocal lb, ub
            rhs = float(v) - b
            if abs(a) < 1e-12:
                if b > float(v):
                    lb, ub = 1.0, 0.0  # infeasible
                return
            x = rhs / a
            if a > 0:
                ub = min(ub, x)
            else:
                lb = max(lb, x)

        if nue_min is not None:
            _intersect_ge(float(nue_min))
        if nue_max is not None:
            _intersect_le(float(nue_max))

    # Nsurp band constraint (lower/upper)
    if nsurp_model is not None and (nsurp_min is not None or nsurp_max is not None):
        c, d = float(nsurp_model["slope"]), float(nsurp_model["intercept"])

        def _intersect_nsurp_ge(v: float):
            nonlocal lb, ub
            rhs = float(v) - d
            if abs(c) < 1e-12:
                # constant model
                if d < float(v):
                    lb, ub = 1.0, 0.0
                return
            x = rhs / c
            if c > 0:
                lb = max(lb, x)
            else:
                ub = min(ub, x)

        def _intersect_nsurp_le(v: float):
            nonlocal lb, ub
            rhs = float(v) - d
            if abs(c) < 1e-12:
                if d > float(v):
                    lb, ub = 1.0, 0.0
                return
            x = rhs / c
            if c > 0:
                ub = min(ub, x)
            else:
                lb = max(lb, x)

        if nsurp_min is not None:
            _intersect_nsurp_ge(float(nsurp_min))
        if nsurp_max is not None:
            _intersect_nsurp_le(float(nsurp_max))

    # clip / sanitize
    lb = float(np.clip(lb, 0.0, bmax_rate))
    ub = float(np.clip(ub, 0.0, bmax_rate))

    return lb, ub


# --------------------------------------------------------------------------------------
# LP-style feasible allocation under per-field bounds + optional global budget
# --------------------------------------------------------------------------------------


# def compute_feasible_bounds_per_field(
#     *,
#     responses: Dict[Tuple[Hashable, Hashable], FieldMetricResponses],
#     max_crop_soil: Dict[Tuple[Hashable, Hashable], float],
#     nue_range: Tuple[float, float] = (0.5, 0.9),
#     nsurp_range: Tuple[float, float] = (0.0, 40.0),
# ) -> Dict[Tuple[Hashable, Hashable], Tuple[float, float]]:
#     """Compute feasible (lb, ub) for each (farm_id, field_id).
#
#     This converts your *metric* constraints into a simple bound on applied nitrogen N.
#
#     Constraints enforced (per field):
#       - NUE in [nue_range[0], nue_range[1]] (if NUE model exists)
#       - Nsurp in [nsurp_range[0], nsurp_range[1]] (if Nsurp model exists)
#       - 0 <= N <= max_crop_soil[(farm_id, field_id)]
#
#     Parameters
#     ----------
#     responses:
#         Output from `fit_field_metric_responses(...)` or `load_field_metric_responses(...)`.
#     max_crop_soil:
#         Dict mapping (farm_id, field_id) -> MAX_CROP_SOIL (kg/ha).
#         You said you'll fill this externally.
#
#     Returns
#     -------
#     Dict[(farm_id, field_id)] -> (lb, ub)
#         If a field is infeasible under the fitted model, lb may exceed ub.
#     """
#     out: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = {}
#
#     nue_min, nue_max = float(nue_range[0]), float(nue_range[1])
#     ns_min, ns_max = float(nsurp_range[0]), float(nsurp_range[1])
#
#     for key, fr in responses.items():
#         bmax = max_crop_soil.get(key)
#         if bmax is None:
#             # If user didn't provide a max for this field, skip it.
#             continue
#
#         nue_model = fr.models.get("NUE")
#         nsurp_model = fr.models.get("Nsurp")
#
#         lb, ub = feasible_N_bounds_from_constraints(
#             nue_model=nue_model,
#             nsurp_model=nsurp_model,
#             bmax_rate=float(bmax),
#             nue_min=nue_min,
#             nue_max=nue_max,
#             nsurp_min=ns_min,
#             nsurp_max=ns_max,
#         )
#
#         out[key] = (float(lb), float(ub))
#
#     return out


def allocate_feasible_N(
    *,
    bounds: Dict[Tuple[Hashable, Hashable], Tuple[float, float]],
    total_budget: Optional[float] = None,
    mode: str = "max_sumN",
    repair_infeasible_budget: bool = True,
) -> Dict[Tuple[Hashable, Hashable], float]:
    """Find a feasible N (kg/ha) per field given bound constraints.

    This is a tiny LP-style allocator, but because constraints are only box bounds
    and an optional global sum constraint, it has a closed-form greedy solution.

    Variables: N_i for each field i

    Constraints:
      lb_i <= N_i <= ub_i
      (optional) sum_i N_i <= total_budget

    Objective (mode):
      - 'max_sumN' : use as much budget as possible (typical for 'allocate N')
      - 'min_sumN' : satisfy constraints with the least N (conservative)

    Returns
    -------
    Dict[(farm_id, field_id)] -> N_i

    Notes
    -----
    - If total_budget is None, we simply return N_i = ub_i (max_sumN) or lb_i (min_sumN).
    - If total_budget is too small to satisfy all lower bounds and repair_infeasible_budget=True, we return a repaired allocation that satisfies the global budget by allowing per-field lower-bound violations (N_i < lb_i).
    """
    keys = list(bounds.keys())
    lbs = np.array([bounds[k][0] for k in keys], dtype=float)
    ubs = np.array([bounds[k][1] for k in keys], dtype=float)

    # Basic sanity: if any field has lb > ub, it is infeasible
    infeasible = lbs > ubs
    if np.any(infeasible):
        # Still return something safe-ish: clamp to ub (or lb) per field
        alloc = np.where(mode == "min_sumN", lbs, ubs)
        return {k: float(v) for k, v in zip(keys, alloc)}

    if total_budget is None:
        alloc = ubs if mode == "max_sumN" else lbs
        return {k: float(v) for k, v in zip(keys, alloc)}

    B = float(total_budget)

    # Start from the minimum feasible point
    alloc = lbs.copy()
    used = float(np.sum(alloc))

    # If we already exceed the global budget, we cannot satisfy all per-field lower bounds.
    # Option C repair: enforce sum_i N_i <= B by allowing N_i < lb_i (violating constraints).
    if used > B + 1e-9:
        if not repair_infeasible_budget:
            # Return lower bounds (signals infeasible for the global budget)
            return {k: float(v) for k, v in zip(keys, alloc)}

        # Amount we must remove from allocations to meet the farm budget
        deficit = used - B

        # Greedy reduction from fields with the largest current allocation first.
        # With uniform lambda=1, any distribution of violations is equivalent in objective.
        order = np.argsort(-alloc)  # descending lb
        for idx in order:
            if deficit <= 1e-9:
                break
            # We can reduce this field down to 0
            reducible = float(alloc[idx])
            if reducible <= 1e-12:
                continue
            take = min(reducible, float(deficit))
            alloc[idx] -= take
            deficit -= take

        # Ensure numerical safety
        alloc = np.clip(alloc, 0.0, ubs)
        return {k: float(v) for k, v in zip(keys, alloc)}

    # Remaining budget to distribute
    rem = B - used

    if mode == "min_sumN":
        # Already minimal
        return {k: float(v) for k, v in zip(keys, alloc)}

    # mode == 'max_sumN': push each field up to its upper bound until budget is used
    headroom = ubs - alloc

    # Greedy fill: equal priority across fields
    for i in range(len(keys)):
        if rem <= 1e-9:
            break
        add = min(float(headroom[i]), float(rem))
        alloc[i] += add
        rem -= add

    return {k: float(v) for k, v in zip(keys, alloc)}


# --------------------------------------------------------------------------------------
# Saving / loading fitted functions (pickled dict)
# --------------------------------------------------------------------------------------


def default_out_path(results_dir: str, tag: str = "default") -> Path:
    out_dir = Path(results_dir) / "LP" / "nue_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"nue_response_{tag}.pkl"


def save_field_metric_responses(
    responses: Dict[Tuple[Hashable, Hashable], FieldMetricResponses],
    out_path: str | Path,
    meta: Optional[dict] = None,
) -> Path:
    payload = {
        "meta": meta or {},
        "responses": {},
    }

    # Make it pickle-friendly (dataclasses -> dict)
    for key, fr in responses.items():
        payload["responses"][key] = {
            "farm_id": fr.farm_id,
            "field_id": fr.field_id,
            "crop": fr.crop,
            "region": fr.region,
            "models": {m: asdict(model) for m, model in fr.models.items()},
        }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    return out_path


def load_field_metric_responses(path: str | Path) -> Dict[Tuple[Hashable, Hashable], FieldMetricResponses]:
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    raw = payload.get("responses", {})
    out: Dict[Tuple[Hashable, Hashable], FieldMetricResponses] = {}

    for key, fr in raw.items():
        models = {}
        for metric, md in fr.get("models", {}).items():
            models[metric] = LinearMetricResponse(
                metric=str(md.get("metric", metric)),
                slope=float(md.get("slope", np.nan)),
                intercept=float(md.get("intercept", np.nan)),
                r2=float(md.get("r2", np.nan)),
                n_points=int(md.get("n_points", 0)),
            )

        out[key] = FieldMetricResponses(
            farm_id=fr.get("farm_id"),
            field_id=fr.get("field_id"),
            crop=str(fr.get("crop", "unknown")),
            region=str(fr.get("region", "unknown")),
            models=models,
        )

    return out


# ======================================================================================
# CLI entrypoint
# ======================================================================================


def _parse_years_arg(s: str) -> Optional[List[int]]:
    """Parse e.g. '2015,2016,2017' -> [2015,2016,2017]."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    years = []
    for p in parts:
        if not p:
            continue
        years.append(int(p))
    return years or None

# ======================================================================================
# Solvers
# ======================================================================================

def get_field_response_model(
    responses: Dict[Tuple[Hashable, ...], "FieldMetricResponses"],
    *,
    farm_id: Hashable,
    field_id: Hashable,
    crop: Optional[str] = None,
) -> Optional["FieldMetricResponses"]:
    """
    Fetch a fitted response model for a field, optionally crop-specific.

    Supports both:
      - (farm_id, field_id)
      - (farm_id, field_id, crop)

    If crop is provided and crop-specific model exists, prefer it.
    """
    if crop is not None:
        k3 = (farm_id, field_id, str(crop))
        if k3 in responses:
            return responses[k3]

    k2 = (farm_id, field_id)
    if k2 in responses:
        return responses[k2]

    # Last resort: any (farm_id, field_id, *) entry
    for k, v in responses.items():
        if len(k) >= 2 and k[0] == farm_id and k[1] == field_id:
            return v

    return None


def compute_feasible_bounds_per_field(
    *,
    responses: Dict[Tuple[Hashable, ...], "FieldMetricResponses"],
    max_crop_soil: Dict[Tuple[Hashable, Hashable], float],
    field_crop: Optional[Dict[Tuple[Hashable, Hashable], str]] = None,
    nue_range: Tuple[float, float] = (0.5, 0.9),
    nsurp_range: Tuple[float, float] = (0.0, 40.0),
) -> Dict[Tuple[Hashable, Hashable], Tuple[float, float]]:
    """
    Convert NUE/Nsurp constraints + MAX_CROP_SOIL into per-field feasible (lb, ub) on N_applied.
    Returns bounds keyed by (farm_id, field_id).
    """
    out: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = {}

    nue_min, nue_max = float(nue_range[0]), float(nue_range[1])
    ns_min, ns_max = float(nsurp_range[0]), float(nsurp_range[1])

    for (farm_id, field_id), bmax in max_crop_soil.items():
        crop = None if field_crop is None else field_crop.get((farm_id, field_id))
        fr = get_field_response_model(responses, farm_id=farm_id, field_id=field_id, crop=crop)
        if fr is None:
            continue

        nue_model = fr["models"].get("NUE")
        nsurp_model = fr["models"].get("Nsurp")

        lb, ub = feasible_N_bounds_from_constraints(
            nue_model=nue_model,
            nsurp_model=nsurp_model,
            bmax_rate=float(bmax),
            nue_min=nue_min,
            nue_max=nue_max,
            nsurp_min=ns_min,
            nsurp_max=ns_max,
        )

        out[(farm_id, field_id)] = (float(lb), float(ub))

    return out


def solve_lp_allocation(
    *,
    responses: Dict[Tuple[Hashable, ...], "FieldMetricResponses"],
    max_crop_soil: Dict[Tuple[Hashable, Hashable], float],
    field_crop: Optional[Dict[Tuple[Hashable, Hashable], str]] = None,
    total_budget: Optional[float] = None,
    nue_range: Tuple[float, float] = (0.5, 0.9),
    nsurp_range: Tuple[float, float] = (0.0, 60.0),
    mode: str = "max_sumN",
) -> Tuple[
    Dict[Tuple[Hashable, Hashable], float],
    Dict[Tuple[Hashable, Hashable], Tuple[float, float]],
    Dict[str, Any],
]:
    """
    Convenience wrapper:
      1) build bounds per field
      2) allocate N_i under optional global budget

    Returns:
      alloc:  (farm_id, field_id) -> Napplied
      bounds: (farm_id, field_id) -> (lb, ub)
      info: diagnostics
    """

    bounds = compute_feasible_bounds_per_field(
        responses=responses,
        max_crop_soil=max_crop_soil,
        field_crop=field_crop,
        nue_range=nue_range,
        nsurp_range=nsurp_range,
    )

    infeasible_fields = [k for k, (lb, ub) in bounds.items() if lb > ub]
    sum_lb = float(np.sum([lb for lb, _ in bounds.values()])) if bounds else 0.0
    sum_ub = float(np.sum([ub for _, ub in bounds.values()])) if bounds else 0.0

    feasible_under_budget = True
    if total_budget is not None:
        feasible_under_budget = sum_lb <= float(total_budget) + 1e-9

    alloc = allocate_feasible_N(
        bounds=bounds,
        total_budget=total_budget,
        mode=mode,
        repair_infeasible_budget=True,
    )

    # Compute total lower-bound violation after allocation (how much we went below lb)
    total_lb_violation = 0.0
    if bounds:
        for k, (lb, _ub) in bounds.items():
            n_i = float(alloc.get(k, float(lb)))
            if n_i < float(lb):
                total_lb_violation += float(lb) - n_i

    info = {
        "infeasible_fields": infeasible_fields,
        "sum_lb": sum_lb,
        "sum_ub": sum_ub,
        "total_budget": None if total_budget is None else float(total_budget),
        "feasible_under_global_budget": bool(feasible_under_budget),
        "mode": str(mode),
        "n_fields": int(len(bounds)),
        "total_lb_violation": float(total_lb_violation),
        "repaired_to_meet_budget": bool((total_budget is not None) and (sum_lb > float(total_budget) + 1e-9) and (total_lb_violation > 1e-9)),
    }
    return alloc, bounds, info


def allocation_dict_to_agent_vector(
    alloc: Dict[Tuple[Hashable, Hashable], float],
    *,
    farm_id: Hashable,
    possible_agents: List[Hashable],
    default: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """
    Convert {(farm_id, field_id) -> value} into a vector ordered by `possible_agents`.
    """
    vec = np.empty(len(possible_agents), dtype=dtype)
    for i, field_id in enumerate(possible_agents):
        vec[i] = float(alloc.get((farm_id, field_id), default))
    return vec


def napplied_to_reduction_units(
    n_applied: float,
    *,
    max_budget_n: float,
    unit_kg: float = 10.0,
    snap: Optional[float] = 0.5,
) -> float:
    """
    Convert N_applied (kg/ha) into reduction-units expected by allocate_bandit_budgets.

    reduction_units = (max_budget_n - n_applied) / 10
    Example: max=200, n=150 => reduction=(50/10)=5.0
    """
    red = (float(max_budget_n) - float(n_applied)) / float(unit_kg)
    red = max(0.0, red)  # cannot reduce negative

    # snap to same grid your bandit uses (0.5 -> 5 kg/ha)
    if snap is not None and snap > 0:
        red = round(red / float(snap)) * float(snap)

    return float(red)


def lp_alloc_to_env_reduction_vector(
    alloc_n: Dict[Tuple[Hashable, Hashable], float],
    *,
    farm_id: Hashable,
    possible_agents: List[Hashable],
    max_crop_soil: Dict[Tuple[Hashable, Hashable], float],
    infeasible_fields: Optional[set[Tuple[Hashable, Hashable]]] = None,
    unit_kg: float = 10.0,
    snap: Optional[float] = 0.5,
    default_n: Optional[float] = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Convert LP solution (N_applied per field) into env reduction-units vector.
    Output aligns with env.possible_agents.
    """
    reds = np.empty(len(possible_agents), dtype=dtype)

    for i, field_id in enumerate(possible_agents):
        bmax = float(max_crop_soil[(farm_id, field_id)])

        # If this field has unsatisfiable bounds (lb > ub), apply 0 reductions:
        # i.e., keep full budget (Napplied = bmax => reduction_units = 0).
        if infeasible_fields is not None and (farm_id, field_id) in infeasible_fields:
            reds[i] = 0.0
            continue

        # choose Napplied for this field
        if (farm_id, field_id) in alloc_n:
            n_applied = float(alloc_n[(farm_id, field_id)])
        else:
            n_applied = float(default_n) if default_n is not None else bmax

        # always clip to [0, bmax]
        n_applied = float(np.clip(n_applied, 0.0, bmax))

        reds[i] = napplied_to_reduction_units(
            n_applied,
            max_budget_n=bmax,
            unit_kg=unit_kg,
            snap=snap,
        )

    return reds


def solve_lp_for_env(
    env,
    *,
    responses: Dict[Tuple[Hashable, ...], "FieldMetricResponses"],
    farm_id: Hashable,
    total_budget: Optional[float] = None,
    nue_range: Tuple[float, float] = (0.5, 0.9),
    nsurp_range: Tuple[float, float] = (0.0, 60.0),
    mode: str = "max_sumN",
    unit_kg: float = 10.0,
    snap: Optional[float] = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve LP allocation and return reductions vector aligned with env.possible_agents.
    """

    # Max per-field budgets from env (kg/ha)
    max_crop_soil = {(farm_id, a): float(env.get_per_parcel_max_budget(a)) for a in env.possible_agents}

    # Crop mapping for this evaluation year (optional but recommended with Option A)
    field_crop = None
    if hasattr(env, "get_per_field_crop_name"):
        fc = env.get_per_field_crop_name()
        field_crop = {(farm_id, a): str(fc[a]) for a in env.possible_agents if a in fc}

    # Run LP wrapper you already have
    alloc_n, bounds, lp_info = solve_lp_allocation(
        responses=responses,
        max_crop_soil=max_crop_soil,
        field_crop=field_crop,
        total_budget=total_budget,
        nue_range=nue_range,
        nsurp_range=nsurp_range,
        mode=mode,
    )

    # Compute sum of Napplied before discretization (continuous alloc)
    applied_sum_continuous = float(np.sum(list(alloc_n.values()))) if alloc_n else 0.0

    # Convert Napplied -> env reduction units vector
    reductions_vec = lp_alloc_to_env_reduction_vector(
        alloc_n,
        farm_id=farm_id,
        possible_agents=list(env.possible_agents),
        max_crop_soil=max_crop_soil,
        infeasible_fields=set(lp_info.get("infeasible_fields", [])),
        unit_kg=unit_kg,
        snap=snap,
    )

    # ------------------------------------------------------------------
    # Enforce STRICT farm-level budget after discretization/snapping.
    # Rationale: converting Napplied -> reduction units (with snap) can
    # introduce rounding that pushes sum(Napplied) above total_budget.
    # Here we greedily increase reductions (decrease Napplied) until the
    # strict budget is satisfied.
    # ------------------------------------------------------------------
    applied_sum = 0.0
    bmax_vec = np.array([float(env.get_per_parcel_max_budget(a)) for a in env.possible_agents], dtype=float)
    red_vec = reductions_vec.astype(float).copy()

    applied_vec = bmax_vec - red_vec * float(unit_kg)
    applied_vec = np.clip(applied_vec, 0.0, bmax_vec)
    applied_sum = float(np.sum(applied_vec))

    strict_budget_repaired = False
    if total_budget is not None:
        B = float(total_budget)
        tol = 1e-6

        # Max possible reduction-units per field (so Napplied >= 0)
        max_red_units = bmax_vec / float(unit_kg)

        step = float(snap) if (snap is not None and snap > 0) else 1.0
        delta_applied = float(unit_kg) * step

        # If rounding caused budget violation, increase reductions greedily
        if applied_sum > B + tol:
            strict_budget_repaired = True
            # Greedy: always reduce the field with the largest current Napplied
            # (uniform lambda=1 => any distribution is acceptable)
            # Safety cap on iterations to avoid infinite loops
            max_iter = int(1e6)
            it = 0
            while applied_sum > B + tol and it < max_iter:
                it += 1
                # candidates that can still be reduced
                can_reduce = applied_vec > tol
                if not np.any(can_reduce):
                    break

                # pick index with largest applied among reducible
                idx = int(np.argmax(np.where(can_reduce, applied_vec, -np.inf)))

                # compute the maximum step we can take without exceeding max reduction
                remaining_units = max_red_units[idx] - red_vec[idx]
                if remaining_units <= tol:
                    # cannot reduce this one anymore; mark as not reducible and continue
                    applied_vec[idx] = 0.0
                    continue

                take_units = min(step, remaining_units)

                red_vec[idx] += take_units
                applied_vec[idx] = max(0.0, bmax_vec[idx] - red_vec[idx] * float(unit_kg))

                # update sum (fast incremental)
                applied_sum = float(np.sum(applied_vec))

            # write back
            reductions_vec = red_vec.astype(np.float32)

    info = {
        **lp_info,
        "alloc_n": alloc_n,
        "bounds": bounds,
        "reductions_vec": reductions_vec,
        "applied_sum_continuous": float(applied_sum_continuous),
        "applied_sum_post_discretization": float(applied_sum),
        "strict_budget_repaired": bool(strict_budget_repaired),
    }
    return reductions_vec, info


@dataclass
class CandidatePoint:
    red_level: float
    N: float
    NUE: float
    Nsurp: float
    N_out: float = np.nan
    N_depo: float = np.nan
    N_seed: float = np.nan
    score_nue: float = np.nan
    score_nsurp: float = np.nan
    score_total: float = np.nan


@dataclass
class DiscreteLPResult:
    farm_id: Hashable
    year: int
    field_ids: list[Hashable]
    chosen_red_level: np.ndarray   # kg/ha per field
    chosen_N: np.ndarray           # kg/ha per field
    total_N: float
    feasible: bool
    reason: str


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _crop_seed_n(crop: str) -> float:
    c = str(crop).strip().lower().replace("_", " ")
    if "potato" in c:
        return 10.0
    if "winter" in c and "wheat" in c:
        return 3.5
    if "sugar" in c and "beet" in c:
        return 0.0
    return 0.0


def _bounded_range_score(x: float, low: float, high: float, max_dev: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(x):
        return 0.0
    if low <= x <= high:
        return 1.0
    if x < low:
        dist = low - x
    else:
        dist = x - high
    if max_dev <= 0:
        return 0.0
    return float(max(0.0, 1.0 - dist / max_dev))


def _target_peak_score(x: float, target: float, width: float, clip: bool = True) -> float:
    """
    Triangular peak score centered at `target`.

    Score is 1 at the target and decreases linearly with absolute distance.
    `width` controls how far from the target the score decays to 0.
    """
    try:
        x = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(x):
        return 0.0
    if width <= 0:
        return 1.0 if abs(x - target) <= 1e-12 else 0.0

    score = 1.0 - abs(x - float(target)) / float(width)
    return float(max(score, 0.0)) if clip else float(score)


def _priority_weights(
    field_ids: list[Hashable],
    areas: dict[Hashable, float],
    field_crop: dict[Hashable, str],
) -> dict[Hashable, float]:
    """
    Priority order:
      1) largest-area field
      2) potato fields
      3) others
    Encoded as weights in the objective.
    """
    weights = {fid: 1.0 for fid in field_ids}
    if not field_ids:
        return weights

    largest_area_fid = max(field_ids, key=lambda f: areas.get(f, 1.0))
    weights[largest_area_fid] = 3.0

    for fid in field_ids:
        crop = str(field_crop.get(fid, "")).lower()
        if "potato" in crop:
            weights[fid] = max(weights.get(fid, 1.0), 2.0)

    return weights


def _last_finite_value(seq):
    if seq is None:
        return None
    try:
        arr = np.asarray(seq, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    mask = np.isfinite(arr)
    if not np.any(mask):
        return None
    return float(arr[np.where(mask)[0][-1]])


def _candidate_point_from_field_info(field_info: dict, *, red_level: float) -> CandidatePoint | None:
    if not isinstance(field_info, dict):
        return None

    n_action = _last_finite_value(field_info.get("Naction", None))
    if n_action is None:
        return None

    nue_val = _last_finite_value(field_info.get("NUE", None))
    if nue_val is None:
        nue_val = _last_finite_value(field_info.get("Nue", None))

    nsurp_val = _last_finite_value(field_info.get("Nsurp", None))
    n_out = _last_finite_value(field_info.get("NamountSO", None))
    no3_depo = _last_finite_value(field_info.get("RNO3DEPOSTT", None))
    nh4_depo = _last_finite_value(field_info.get("RNH4DEPOSTT", None))

    crop_val = "unknown"
    crop_name = field_info.get("CropName", None)
    if crop_name is not None:
        try:
            if isinstance(crop_name, (list, tuple, np.ndarray)) and len(crop_name) > 0:
                crop_val = str(crop_name[-1])
            else:
                crop_val = str(crop_name)
        except Exception:
            crop_val = "unknown"

    n_seed = _crop_seed_n(crop_val)
    n_depo = float(
        (0.0 if no3_depo is None or not np.isfinite(no3_depo) else no3_depo)
        + (0.0 if nh4_depo is None or not np.isfinite(nh4_depo) else nh4_depo)
    )

    if n_out is not None and np.isfinite(n_out):
        n_tot_in = float(n_action) + float(n_seed) + float(n_depo)
        nue_proxy = float(n_out) / n_tot_in if n_tot_in > 0 else np.nan
        nsurp_proxy = float(n_tot_in) - float(n_out)
    else:
        nue_proxy = float(nue_val) if nue_val is not None and np.isfinite(nue_val) else np.nan
        nsurp_proxy = float(nsurp_val) if nsurp_val is not None and np.isfinite(nsurp_val) else np.nan

    return CandidatePoint(
        red_level=float(red_level),
        N=float(n_action),
        NUE=float(nue_proxy) if np.isfinite(nue_proxy) else np.nan,
        Nsurp=float(nsurp_proxy) if np.isfinite(nsurp_proxy) else np.nan,
        N_out=float(n_out) if n_out is not None and np.isfinite(n_out) else np.nan,
        N_depo=float(n_depo),
        N_seed=float(n_seed),
    )


def _candidate_reduction_grid(max_budget: float, step: float = 20.0) -> list[float]:
    if not np.isfinite(max_budget) or max_budget <= 0:
        return [0.0]
    vals = list(np.arange(0.0, float(max_budget) + 1e-9, float(step)))
    if abs(vals[-1] - float(max_budget)) > 1e-9:
        vals.append(float(max_budget))
    return [float(v) for v in vals]


def _serialize_alloc_history(history: dict[int, dict[Hashable, float]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for year, allocs in history.items():
        out[int(year)] = {str(fid): float(val) for fid, val in allocs.items()}
    return out


def _get_rss_mb() -> float:
    """Best-effort resident-set size in MB."""
    try:
        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS usually reports bytes, Linux usually reports KB
        if rss_raw > 10_000_000:
            return float(rss_raw) / (1024.0 * 1024.0)
        return float(rss_raw) / 1024.0
    except Exception:
        return float("nan")


def _maybe_start_tracemalloc(enabled: bool) -> None:
    if not enabled:
        return
    try:
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)
    except Exception:
        pass


def _format_top_allocations(limit: int = 5) -> str:
    try:
        if not tracemalloc.is_tracing():
            return "tracemalloc_off"
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("lineno")[:limit]
        return " | ".join(
            f"{s.traceback[0]}: {s.size / (1024.0 * 1024.0):.2f} MB"
            for s in stats
        )
    except Exception as e:
        return f"tracemalloc_error:{e}"


def _memory_log(label: str, *, enabled: bool = False) -> None:
    if not enabled:
        return
    gc.collect()
    rss = _get_rss_mb()
    top = _format_top_allocations(limit=5)
    print(f"[mem] {label} rss_mb={rss:.2f} top={top}")



# Helper to reset and replay allocations in an existing env instance.
def _reset_env_for_replay(
    env,
    *,
    farm_dict_by_year: dict[int, dict],
    season_years: list[int],
    allocation_history: dict[int, dict[Hashable, float]],
    target_year: int,
    days_before_sowing: int = 7,
    stop_at_target_preseason: bool = True,
):
    """
    Reuse an existing MultiFieldEnv instance by resetting/reconfiguring it and replaying
    accepted allocations up to the target year.

    This avoids constructing a brand-new MultiFieldEnv object for every replay.
    """
    farm0 = farm_dict_by_year[int(season_years[0])]

    if hasattr(env, "set_new_fields"):
        env.set_new_fields(farm0)

    env.reset(options={
        "year": int(season_years[0]),
        "eval_horizon_years": list(season_years),
        "farm_dict_by_year": farm_dict_by_year,
        "preseason_allocation": True,
        "days_before_sowing": int(days_before_sowing),
    })

    for sy in season_years:
        sy = int(sy)
        env.advance_fields_to_allocation_dates(
            days_before_sowing=int(days_before_sowing),
            season_year=sy,
            farm_dict_by_year=farm_dict_by_year,
        )

        if sy == int(target_year) and bool(stop_at_target_preseason):
            break

        hist = allocation_history.get(sy, None)
        if hist is None:
            raise RuntimeError(
                f"Missing replay allocation history for season {sy} while rebuilding target year {target_year}."
            )

        reductions_vec = []
        for fid in env.possible_agents:
            val = hist.get(fid, hist.get(str(fid), None))
            if val is None:
                raise RuntimeError(f"Missing replay allocation for field {fid} in season {sy}.")
            reductions_vec.append(float(val))

        env.allocate_bandit_budgets(reductions_vec)
        env.run_til_past_season_year(season_year=sy)
        gc.collect()

    return env

def _rebuild_env_with_history(
    env_template_or_cls,
    *,
    farm_dict_by_year: dict[int, dict],
    season_years: list[int],
    allocation_history: dict[int, dict[Hashable, float]],
    target_year: int,
    days_before_sowing: int = 7,
    stop_at_target_preseason: bool = True,
):
    """
    Rebuild a fresh MultiFieldEnv from scratch and replay accepted allocations up to
    the preseason allocation point of `target_year`.
    """
    env_cls = env_template_or_cls if isinstance(env_template_or_cls, type) else env_template_or_cls.__class__
    farm0 = farm_dict_by_year[int(season_years[0])]

    new_env = env_cls(
        training=False,
        render=False,
        farm_dict=farm0,
        reward='NSU',
    )

    return _reset_env_for_replay(
        new_env,
        farm_dict_by_year=farm_dict_by_year,
        season_years=season_years,
        allocation_history=allocation_history,
        target_year=target_year,
        days_before_sowing=days_before_sowing,
        stop_at_target_preseason=stop_at_target_preseason,
    )


def _save_online_oracle_progress(
    *,
    checkpoint_path: str | os.PathLike,
    farm_key: str,
    season_year: int,
    reductions_vec_units10: np.ndarray,
    lp_diag,
    allocation_history: dict[int, dict[Hashable, float]],
) -> None:
    payload = {
        "farm_key": str(farm_key),
        "season_year": int(season_year),
        "reductions_vec_units10": np.asarray(reductions_vec_units10, dtype=float),
        "chosen_red_level": np.asarray(getattr(lp_diag, "chosen_red_level", []), dtype=float),
        "chosen_N": np.asarray(getattr(lp_diag, "chosen_N", []), dtype=float),
        "total_N": float(getattr(lp_diag, "total_N", np.nan)),
        "feasible": bool(getattr(lp_diag, "feasible", False)),
        "reason": str(getattr(lp_diag, "reason", "unknown")),
        "allocation_history": _serialize_alloc_history(allocation_history),
    }

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(payload, f)


def _generate_online_candidates_for_field_worker(
    *,
    env_cls,
    farm_dict_by_year: dict[int, dict],
    season_years: list[int],
    allocation_history: dict[int, dict[Hashable, float]],
    season_year: int,
    field_id: Hashable,
    days_before_sowing: int,
    reduction_step: float,
    debug_memory: bool = False,
) -> tuple[Hashable, list[CandidatePoint]]:
    """
    Worker for online oracle candidate generation.

    Reuses a single env object for one field and repeatedly resets/replays it,
    instead of constructing a brand-new MultiFieldEnv for every candidate.
    """
    _maybe_start_tracemalloc(debug_memory)
    _memory_log(f"worker_start field={field_id} season={season_year}", enabled=debug_memory)

    farm0 = farm_dict_by_year[int(season_years[0])]
    live_env = env_cls(
        training=False,
        render=False,
        farm_dict=farm0,
        reward='NSU',
    )

    try:
        live_env = _reset_env_for_replay(
            live_env,
            farm_dict_by_year=farm_dict_by_year,
            season_years=season_years,
            allocation_history=allocation_history,
            target_year=int(season_year),
            days_before_sowing=int(days_before_sowing),
            stop_at_target_preseason=True,
        )

        try:
            max_budget = float(live_env.get_per_parcel_max_budget(field_id))
        except Exception:
            max_budget = np.nan

        candidates_for_field: list[CandidatePoint] = []
        gc.collect()

        red_grid = _candidate_reduction_grid(max_budget, step=reduction_step)

        for red_level in red_grid:
            try:
                live_env = _reset_env_for_replay(
                    live_env,
                    farm_dict_by_year=farm_dict_by_year,
                    season_years=season_years,
                    allocation_history=allocation_history,
                    target_year=int(season_year),
                    days_before_sowing=int(days_before_sowing),
                    stop_at_target_preseason=True,
                )

                allocation = max(0.0, float(max_budget) - float(red_level))
                live_env.set_per_parcel_budget(field_id, allocation)

                if hasattr(live_env, "_get_global_budget_left"):
                    live_env.global_budget_left = live_env._get_global_budget_left()
                    live_env.global_allocated_budget = live_env.global_budget_left

                live_env.run_til_past_season_year(season_year=int(season_year))
                season_info = live_env.collect_agent_infos_for_season(int(season_year))
                point = _candidate_point_from_field_info(
                    season_info.get(field_id, {}),
                    red_level=float(red_level),
                )
                if point is not None:
                    candidates_for_field.append(point)
            except Exception:
                continue
            finally:
                gc.collect()

        _memory_log(
            f"worker_end field={field_id} season={season_year} n_candidates={len(candidates_for_field)}",
            enabled=debug_memory,
        )
        candidates_for_field.sort(key=lambda p: (p.N, p.red_level))
        return field_id, candidates_for_field
    finally:
        try:
            live_env.close()
        except Exception:
            pass
        finally:
            try:
                del live_env
            except Exception:
                pass
            gc.collect()


def build_candidate_grid(
    df: pd.DataFrame,
    *,
    farm_id: Hashable,
    year: int,
    field_ids: list[Hashable] | None = None,
    nue_range: tuple[float, float] = (0.5, 0.8),
    nsurp_range: tuple[float, float] = (20.0, 80.0),
    require_finite: bool = True,
    filter_feasible: bool = True,
    keep_closest_if_infeasible: bool = True,
) -> dict[Hashable, list[CandidatePoint]]:
    """
    Build per-field candidate lists from a precomputed response DataFrame.

    Expected minimum columns:
      ['farm_id', 'field_id', 'year', 'red_level', 'N']

    Optional proxy columns:
      ['N_out', 'N_depo', 'N_seed', 'NUE_proxy', 'Nsurp_proxy']

    Returns dict[field_id] -> list[CandidatePoint] sorted by N ascending.
    """
    lo_nue, hi_nue = float(nue_range[0]), float(nue_range[1])
    lo_ns, hi_ns = float(nsurp_range[0]), float(nsurp_range[1])

    dff = df[(df["farm_id"] == farm_id) & (df["year"] == int(year))].copy()
    if dff.empty:
        return {}

    if field_ids is not None:
        wanted = {str(x) for x in field_ids}
        dff = dff[dff["field_id"].astype(str).isin(wanted)]

    if "red_level" in dff.columns:
        dff = dff[np.isfinite(pd.to_numeric(dff["red_level"], errors="coerce").to_numpy(dtype=float))]

    # Prefer proxy metrics if available
    if "NUE_proxy" in dff.columns:
        dff["NUE_eff"] = pd.to_numeric(dff["NUE_proxy"], errors="coerce")
    else:
        dff["NUE_eff"] = pd.to_numeric(dff.get("NUE", np.nan), errors="coerce")

    if "Nsurp_proxy" in dff.columns:
        dff["Nsurp_eff"] = pd.to_numeric(dff["Nsurp_proxy"], errors="coerce")
    else:
        dff["Nsurp_eff"] = pd.to_numeric(dff.get("Nsurp", np.nan), errors="coerce")

    if require_finite:
        for col in ("N", "NUE_eff", "Nsurp_eff"):
            if col in dff.columns:
                dff = dff[np.isfinite(pd.to_numeric(dff[col], errors="coerce").to_numpy(dtype=float))]

    if filter_feasible:
        feas = (
            dff["NUE_eff"].between(lo_nue, hi_nue, inclusive="both")
            & dff["Nsurp_eff"].between(lo_ns, hi_ns, inclusive="both")
        )

        if keep_closest_if_infeasible:
            nue = dff["NUE_eff"].to_numpy(dtype=float)
            ns = dff["Nsurp_eff"].to_numpy(dtype=float)

            nue_v = np.maximum(0.0, lo_nue - nue) + np.maximum(0.0, nue - hi_nue)
            ns_v = np.maximum(0.0, lo_ns - ns) + np.maximum(0.0, ns - hi_ns)

            dff = dff.copy()
            dff["__feas__"] = feas.to_numpy(dtype=bool)
            dff["__penalty__"] = nue_v + ns_v

            kept = []
            for fid, g in dff.groupby("field_id", sort=False):
                gf = g[g["__feas__"]]
                if len(gf) > 0:
                    kept.append(gf)
                else:
                    kept.append(g.sort_values(["__penalty__", "N"], ascending=[True, True]).head(1))

            dff = pd.concat(kept, axis=0, ignore_index=True) if kept else dff.iloc[0:0]
            dff = dff.drop(columns=["__feas__", "__penalty__"], errors="ignore")
        else:
            dff = dff[feas]

    out: dict[Hashable, list[CandidatePoint]] = {}
    for fid, g in dff.groupby("field_id", sort=False):
        pts = []
        for _, row in g.iterrows():
            pts.append(
                CandidatePoint(
                    red_level=_safe_float(row.get("red_level", np.nan)),
                    N=_safe_float(row.get("N", np.nan)),
                    NUE=_safe_float(row.get("NUE_eff", np.nan)),
                    Nsurp=_safe_float(row.get("Nsurp_eff", np.nan)),
                    N_out=_safe_float(row.get("N_out", np.nan)),
                    N_depo=_safe_float(row.get("N_depo", np.nan)),
                    N_seed=_safe_float(row.get("N_seed", np.nan)),
                )
            )
        pts.sort(key=lambda p: (p.N, p.red_level))
        if pts:
            out[fid] = pts

    if field_ids is not None and out:
        ordered: dict[Hashable, list[CandidatePoint]] = {}
        for fid in field_ids:
            if fid in out:
                ordered[fid] = out[fid]
            else:
                for k in list(out.keys()):
                    if str(k) == str(fid):
                        ordered[fid] = out[k]
                        break
        for k, v in out.items():
            if k not in ordered:
                ordered[k] = v
        out = ordered

    return out


def solve_discrete_score_oracle(
    candidates: dict[Hashable, list[CandidatePoint]],
    *,
    field_priority_weight: Optional[dict[Hashable, float]] = None,
    total_budget: float | None = None,
    nue_range: tuple[float, float] = (0.5, 0.8),
    nsurp_range: tuple[float, float] = (20.0, 80.0),
    nue_target: float = 0.8,
    nsurp_target: float = 30.0,
    nue_width: float = 0.2,
    nsurp_width: float = 20.0,
    score_mode: str = "peak",
    tie_break_eps: float = 1e-9,
) -> tuple[dict[Hashable, CandidatePoint], bool, str]:
    """
    Lexicographic discrete oracle over precomputed candidates.

    Stage 1:
        maximize weighted agronomic score
    Stage 2:
        among all score-optimal solutions, minimize total N

    This avoids the degenerate pure-min-N solution while still giving a true
    "minimum-N subject to best achievable score" oracle.
    """
    if not candidates:
        return {}, False, "no_candidates"

    try:
        import pulp
    except Exception as e:
        return {}, False, f"missing_pulp:{e}"

    lo_nue, hi_nue = float(nue_range[0]), float(nue_range[1])
    lo_ns, hi_ns = float(nsurp_range[0]), float(nsurp_range[1])

    weights = field_priority_weight or {fid: 1.0 for fid in candidates.keys()}

    # Precompute candidate scores once
    raw_score_map: dict[tuple[Hashable, int], float] = {}
    score_map: dict[tuple[Hashable, int], float] = {}
    for fid, pts in candidates.items():
        if not pts:
            return {}, False, f"no_candidates_for_field:{fid}"
        for j, p in enumerate(pts):
            if score_mode == "peak":
                s_nue = _target_peak_score(
                    p.NUE,
                    target=float(nue_target),
                    width=float(nue_width),
                )
                s_ns = _target_peak_score(
                    p.Nsurp,
                    target=float(nsurp_target),
                    width=float(nsurp_width),
                )
            elif score_mode == "range":
                s_nue = _bounded_range_score(p.NUE, lo_nue, hi_nue, nue_width)
                s_ns = _bounded_range_score(p.Nsurp, lo_ns, hi_ns, nsurp_width)
            else:
                raise ValueError(f"Unknown score_mode: {score_mode}")

            s_total = s_nue * s_ns

            pts[j].score_nue = float(s_nue)
            pts[j].score_nsurp = float(s_ns)
            pts[j].score_total = float(s_total)
            score_map[(fid, j)] = float(weights.get(fid, 1.0)) * float(s_total)
            raw_score_map[(fid, j)] = float(s_total)

    def _build_problem(
            objective: str,
            target_weighted_score: float | None = None,
            target_raw_score: float | None = None,
    ):
        prob = pulp.LpProblem(
            f"discrete_score_oracle_{objective}",
            pulp.LpMaximize if objective in {"score", "raw_score"} else pulp.LpMinimize,
        )
        x = {}

        for fid, pts in candidates.items():
            for j, _ in enumerate(pts):
                x[(fid, j)] = pulp.LpVariable(
                    f"x_{str(fid).replace('-', '_')}_{j}",
                    cat="Binary",
                )

        # exactly one candidate per field
        for fid, pts in candidates.items():
            prob += pulp.lpSum(x[(fid, j)] for j in range(len(pts))) == 1, f"one_choice_{str(fid)}"

        # optional farm budget
        if total_budget is not None:
            prob += (
                    pulp.lpSum(
                        float(candidates[fid][j].N) * x[(fid, j)]
                        for fid in candidates
                        for j in range(len(candidates[fid]))
                    )
                    <= float(total_budget)
            ), "farm_budget"

        total_weighted_score_expr = pulp.lpSum(
            score_map[(fid, j)] * x[(fid, j)]
            for fid in candidates
            for j in range(len(candidates[fid]))
        )

        total_raw_score_expr = pulp.lpSum(
            raw_score_map[(fid, j)] * x[(fid, j)]
            for fid in candidates
            for j in range(len(candidates[fid]))
        )

        total_n_expr = pulp.lpSum(
            float(candidates[fid][j].N) * x[(fid, j)]
            for fid in candidates
            for j in range(len(candidates[fid]))
        )

        if target_weighted_score is not None:
            prob += (
                    total_weighted_score_expr >= float(target_weighted_score) - float(tie_break_eps)
            ), "fix_optimal_weighted_score"

        if target_raw_score is not None:
            prob += (
                    total_raw_score_expr >= float(target_raw_score) - float(tie_break_eps)
            ), "fix_optimal_raw_score"

        if objective == "score":
            prob += total_weighted_score_expr
        elif objective == "raw_score":
            prob += total_raw_score_expr
        elif objective == "min_n":
            prob += total_n_expr
        else:
            raise ValueError(f"Unknown objective stage: {objective}")

        return prob, x, total_weighted_score_expr, total_raw_score_expr, total_n_expr

    # Stage 1: maximize priority-weighted score
    prob1, x1, weighted_score_expr1, _, _ = _build_problem("score")
    status1 = prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    status1_str = pulp.LpStatus.get(status1, str(status1))
    if status1_str != "Optimal":
        return {}, False, f"solver_status_stage1:{status1_str}"

    best_weighted_score = pulp.value(weighted_score_expr1)
    if best_weighted_score is None:
        return {}, False, "stage1_no_score"

    # Stage 2: among Stage-1-optimal solutions, maximize total unweighted score
    prob2, x2, _, raw_score_expr2, _ = _build_problem(
        "raw_score",
        target_weighted_score=float(best_weighted_score),
    )
    status2 = prob2.solve(pulp.PULP_CBC_CMD(msg=False))
    status2_str = pulp.LpStatus.get(status2, str(status2))
    if status2_str != "Optimal":
        return {}, False, f"solver_status_stage2:{status2_str}"

    best_raw_score = pulp.value(raw_score_expr2)
    if best_raw_score is None:
        return {}, False, "stage2_no_score"

    # Stage 3: among Stage-1/2-optimal solutions, minimize total N
    prob3, x3, _, _, _ = _build_problem(
        "min_n",
        target_weighted_score=float(best_weighted_score),
        target_raw_score=float(best_raw_score),
    )
    status3 = prob3.solve(pulp.PULP_CBC_CMD(msg=False))
    status3_str = pulp.LpStatus.get(status3, str(status3))
    if status3_str != "Optimal":
        return {}, False, f"solver_status_stage3:{status3_str}"

    chosen: dict[Hashable, CandidatePoint] = {}
    for fid, pts in candidates.items():
        picked = None
        for j, p in enumerate(pts):
            val = pulp.value(x3[(fid, j)])
            if val is not None and float(val) > 0.5:
                picked = p
                break
        if picked is None:
            return chosen, False, f"no_selected_candidate:{fid}"
        chosen[fid] = picked

    return chosen, True, "ok"


def solve_discrete_minN(
    candidates: dict[Hashable, list[CandidatePoint]],
    *,
    total_budget: float | None = None,
) -> tuple[dict[Hashable, CandidatePoint], bool, str]:
    """
    Select one candidate per field to minimize total N.

    With objective min(sum N) and only coupling constraint sum(N) <= total_budget,
    the optimal policy is to pick each field’s minimum-N feasible point.
    """
    if not candidates:
        return {}, False, "no_candidates"

    chosen: dict[Hashable, CandidatePoint] = {}
    for fid, pts in candidates.items():
        if not pts:
            return {}, False, f"no_feasible_points_for_field:{fid}"
        chosen[fid] = pts[0]

    if total_budget is not None:
        totalN = float(sum(p.N for p in chosen.values()))
        if totalN > float(total_budget) + 1e-9:
            return chosen, False, "min_solution_exceeds_total_budget"

    return chosen, True, "ok"


def solve_discrete_lp_online_for_env(
    env,
    *,
    farm_dict_by_year: dict[int, dict],
    season_years: list[int],
    season_year: int,
    total_budget: float | None = None,
    nue_range: tuple[float, float] = (0.5, 0.8),
    nsurp_range: tuple[float, float] = (10.0, 80.0),
    reduction_step: float = 20.0,
    allocation_history: dict[int, dict[Hashable, float]] | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    farm_key: str = "online",
    days_before_sowing: int = 7,
    n_jobs: int | None = None,
    debug_memory: bool = False,
) -> tuple[np.ndarray, DiscreteLPResult, dict[Hashable, float]]:
    """
    This is slower than the offline grid, but avoids the deepcopy instability and the
    trajectory-stitching mismatch from precomputed candidate tables. The farm-level
    allocation uses a three-stage lexicographic objective that first protects
    higher-priority fields, then improves total closeness to target across all fields,
    and only then minimizes total N.
    """
    season_year = int(season_year)
    season_years = [int(y) for y in season_years]
    allocation_history = {} if allocation_history is None else dict(allocation_history)

    _maybe_start_tracemalloc(debug_memory)
    _memory_log(f"online_solver_start farm={farm_key} season={season_year}", enabled=debug_memory)

    live_env = _rebuild_env_with_history(
        env,
        farm_dict_by_year=farm_dict_by_year,
        season_years=season_years,
        allocation_history=allocation_history,
        target_year=season_year,
        days_before_sowing=days_before_sowing,
        stop_at_target_preseason=True,
    )

    field_ids = list(getattr(live_env, "possible_agents", []))
    if not field_ids:
        field_ids = list(getattr(live_env, "parcel_meta_infos", {}).keys())

    def _field_area(fid) -> float:
        if hasattr(live_env, "get_per_parcel_area"):
            try:
                return float(live_env.get_per_parcel_area(fid))
            except Exception:
                pass
        try:
            if hasattr(live_env, "fields") and fid in live_env.fields:
                fenv = live_env.fields[fid]
                fenv = getattr(fenv, "unwrapped", fenv)
                for attr in ("area", "AREA", "parcel_area"):
                    if hasattr(fenv, attr):
                        return float(getattr(fenv, attr))
        except Exception:
            pass
        return 1.0

    def _field_crop(fid) -> str:
        try:
            if hasattr(live_env, "fields") and fid in live_env.fields:
                fenv = live_env.fields[fid]
                fenv = getattr(fenv, "unwrapped", fenv)
                if hasattr(fenv, "crop"):
                    return str(getattr(fenv, "crop"))
        except Exception:
            pass
        return "unknown"

    areas = {fid: _field_area(fid) for fid in field_ids}
    field_crop = {fid: _field_crop(fid) for fid in field_ids}
    priority_weight = _priority_weights(field_ids, areas, field_crop)

    # Cache the preseason target-year state once. Candidate evaluation then starts from
    # this cached state instead of replaying all past seasons for every field × level.
    preseason_template = {
        "agent_infos": {},
        "budget_left": {},
        "budget_total": {},
        "max_budget": {},
    }
    for fid in field_ids:
        try:
            preseason_template["agent_infos"][fid] = copy.deepcopy(live_env.fields[fid].unwrapped.infos)
        except Exception:
            preseason_template["agent_infos"][fid] = None
        try:
            preseason_template["budget_left"][fid] = float(live_env.get_per_parcel_budget_left(fid))
        except Exception:
            preseason_template["budget_left"][fid] = None
        try:
            preseason_template["budget_total"][fid] = float(live_env.get_per_parcel_budget(fid))
        except Exception:
            preseason_template["budget_total"][fid] = None
        try:
            preseason_template["max_budget"][fid] = float(live_env.get_per_parcel_max_budget(fid))
        except Exception:
            preseason_template["max_budget"][fid] = None

    candidates: dict[Hashable, list[CandidatePoint]] = {fid: [] for fid in field_ids}
    env_cls = env.__class__
    n_jobs_eff = int(n_jobs) if n_jobs is not None else max(1, min(len(field_ids), (os.cpu_count() or 1)))

    if n_jobs_eff <= 1:
        for fid in field_ids:
            try:
                fid_out, pts = _generate_online_candidates_for_field_worker(
                    env_cls=env_cls,
                    farm_dict_by_year=farm_dict_by_year,
                    season_years=season_years,
                    allocation_history=allocation_history,
                    season_year=int(season_year),
                    field_id=fid,
                    days_before_sowing=int(days_before_sowing),
                    reduction_step=float(reduction_step),
                    debug_memory=debug_memory,
                )
                candidates[fid_out] = pts
            except Exception:
                candidates[fid] = []
    else:
        futures = {}
        try:
            with ProcessPoolExecutor(max_workers=n_jobs_eff) as ex:
                for fid in field_ids:
                    fut = ex.submit(
                        _generate_online_candidates_for_field_worker,
                        env_cls=env_cls,
                        farm_dict_by_year=farm_dict_by_year,
                        season_years=season_years,
                        allocation_history=allocation_history,
                        season_year=int(season_year),
                        field_id=fid,
                        days_before_sowing=int(days_before_sowing),
                        reduction_step=float(reduction_step),
                        debug_memory=debug_memory,
                    )
                    futures[fut] = fid

                for fut in as_completed(futures):
                    fid = futures[fut]
                    try:
                        fid_out, pts = fut.result()
                        candidates[fid_out] = pts
                    except Exception:
                        candidates[fid] = []
        except Exception:
            # Fallback to sequential if multiprocessing fails on this platform/env.
            for fid in field_ids:
                try:
                    fid_out, pts = _generate_online_candidates_for_field_worker(
                        env_cls=env_cls,
                        farm_dict_by_year=farm_dict_by_year,
                        season_years=season_years,
                        allocation_history=allocation_history,
                        season_year=int(season_year),
                        field_id=fid,
                        days_before_sowing=int(days_before_sowing),
                        reduction_step=float(reduction_step),
                        debug_memory=debug_memory,
                    )
                    candidates[fid_out] = pts
                except Exception:
                    candidates[fid] = []

    _memory_log(f"online_solver_after_candidates farm={farm_key} season={season_year}", enabled=debug_memory)

    chosen, feasible, reason = solve_discrete_score_oracle(
        candidates,
        field_priority_weight=priority_weight,
        total_budget=total_budget,
        nue_range=nue_range,
        nsurp_range=nsurp_range,
        nue_target=0.7,
        nsurp_target=40.0,
        nue_width=0.2,
        nsurp_width=20.0,
        score_mode="range",
    )

    chosen_red = []
    chosen_N = []
    reductions_history_for_year: dict[Hashable, float] = {}
    for fid in field_ids:
        p = chosen.get(fid, None)
        if p is None:
            chosen_red.append(0.0)
            chosen_N.append(np.nan)
            reductions_history_for_year[fid] = 0.0
            feasible = False
            if reason == "ok":
                reason = f"missing_field_in_solution:{fid}"
        else:
            chosen_red.append(float(p.red_level))
            chosen_N.append(float(p.N))
            reductions_history_for_year[fid] = float(p.red_level) / 10.0

    chosen_red = np.asarray(chosen_red, dtype=float)
    chosen_N = np.asarray(chosen_N, dtype=float)
    reductions_vec_units10 = chosen_red / 10.0

    res = DiscreteLPResult(
        farm_id=str(farm_key),
        year=season_year,
        field_ids=list(field_ids),
        chosen_red_level=chosen_red,
        chosen_N=chosen_N,
        total_N=float(np.nansum(chosen_N)),
        feasible=bool(feasible),
        reason=str(reason),
    )

    allocation_history[season_year] = dict(reductions_history_for_year)
    if checkpoint_path is not None:
        _save_online_oracle_progress(
            checkpoint_path=checkpoint_path,
            farm_key=str(farm_key),
            season_year=season_year,
            reductions_vec_units10=reductions_vec_units10,
            lp_diag=res,
            allocation_history=allocation_history,
        )

    try:
        live_env.close()
    except Exception:
        pass
    finally:
        try:
            del live_env
        except Exception:
            pass
        gc.collect()

    return reductions_vec_units10, res, reductions_history_for_year


def solve_discrete_lp_for_env(
    env,
    *,
    df_metrics: pd.DataFrame,
    farm_id: Hashable,
    year: int,
    total_budget: float | None = None,
    nue_range: tuple[float, float] = (0.5, 0.8),
    nsurp_range: tuple[float, float] = (20.0, 80.0),
) -> tuple[np.ndarray, DiscreteLPResult]:
    """
       The optimizer uses a three-stage lexicographic objective under an optional farm-level
    N budget:
      1) maximize a priority-weighted agronomic score
      2) among Stage-1-optimal solutions, maximize total unweighted agronomic score
      3) among Stage-1/2-optimal solutions, minimize total N
    Priority order is encoded as:
    """
    field_ids = list(getattr(env, "possible_agents", []))
    if not field_ids:
        field_ids = list(getattr(env, "parcel_meta_infos", {}).keys())

    def _field_area(fid) -> float:
        if hasattr(env, "get_per_parcel_area"):
            try:
                return float(env.get_per_parcel_area(fid))
            except Exception:
                pass
        try:
            if hasattr(env, "fields") and fid in env.fields:
                fenv = env.fields[fid]
                fenv = getattr(fenv, "unwrapped", fenv)
                for attr in ("area", "AREA", "parcel_area"):
                    if hasattr(fenv, attr):
                        return float(getattr(fenv, attr))
        except Exception:
            pass
        return 1.0

    areas = {fid: _field_area(fid) for fid in field_ids}

    # infer crop per field
    dff_crop = df_metrics[(df_metrics["farm_id"] == farm_id) & (df_metrics["year"] == int(year))].copy()
    field_crop: dict[Hashable, str] = {}
    if not dff_crop.empty:
        crop_col = "crop" if "crop" in dff_crop.columns else ("CropName" if "CropName" in dff_crop.columns else None)
        if crop_col is not None:
            for fid, g in dff_crop.groupby("field_id"):
                try:
                    c = g[crop_col].dropna().astype(str)
                    field_crop[fid] = str(c.mode().iloc[0]) if len(c) else "unknown"
                except Exception:
                    field_crop[fid] = "unknown"

    priority_weight = _priority_weights(field_ids, areas, field_crop)

    cand_all = build_candidate_grid(
        df_metrics,
        farm_id=farm_id,
        year=int(year),
        field_ids=field_ids,
        nue_range=nue_range,
        nsurp_range=nsurp_range,
        require_finite=True,
        filter_feasible=False,
    )

    chosen, feasible, reason = solve_discrete_score_oracle(
        cand_all,
        field_priority_weight=priority_weight,
        total_budget=total_budget,
        nue_range=nue_range,
        nsurp_range=nsurp_range,
        nue_target=0.8,
        nsurp_target=20.0,
        nue_width=0.2,
        nsurp_width=40.0,
        score_mode="range",
    )

    chosen_red = []
    chosen_N = []

    for fid in field_ids:
        p = None
        if fid in chosen:
            p = chosen[fid]
        else:
            for k, v in chosen.items():
                if str(k) == str(fid):
                    p = v
                    break

        if p is None:
            chosen_red.append(0.0)
            chosen_N.append(np.nan)
            feasible = False
            if reason == "ok":
                reason = f"missing_field_in_solution:{fid}"
        else:
            chosen_red.append(float(p.red_level))
            chosen_N.append(float(p.N))

    chosen_red = np.asarray(chosen_red, dtype=float)
    chosen_N = np.asarray(chosen_N, dtype=float)

    reductions_vec_units10 = chosen_red / 10.0

    res = DiscreteLPResult(
        farm_id=farm_id,
        year=int(year),
        field_ids=list(field_ids),
        chosen_red_level=chosen_red,
        chosen_N=chosen_N,
        total_N=float(np.nansum(chosen_N)),
        feasible=bool(feasible),
        reason=str(reason),
    )

    return reductions_vec_units10, res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit per-field NUE / Nsurp response functions from saved rollouts.")

    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the pickle created by your runner/eval (contains per-farm, per-field time series).",
    )

    parser.add_argument(
        "--train_years",
        type=str,
        default="",
        help="Comma-separated years to fit on. Example: '2015,2016,2017,2018,2019'. Empty = use all.",
    )

    parser.add_argument(
        "--min_points",
        type=int,
        default=2,
        help="Minimum number of (finite) datapoints required per field/metric.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Tag used in the output filename (e.g., 'ROT', 'random', 'baseline').",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=_DEFAULT_RESULTSDIR,
        help="Base results directory for saving response models.",
    )

    args = parser.parse_args()

    # 1) Load
    with open(args.file_path, "rb") as f:
        data = pickle.load(f)

    # 2) Make dataframe
    df = make_df_nue_response(data)

    # 3) Fit
    train_years = _parse_years_arg(args.train_years)
    responses = fit_field_metric_responses(
        df,
        train_years=train_years,
        min_points=int(args.min_points),
        metrics=["NUE", "Nsurp"],
    )

    # 4) Save
    out_path = default_out_path(args.results_dir, tag=str(list(responses.keys())[0][0]))
    meta = {
        "file_path": str(args.file_path),
        "train_years": train_years,
        "min_points": int(args.min_points),
        "n_rows": int(len(df)),
        "n_fields_fitted": int(len(responses)),
    }

    saved = save_field_metric_responses(responses, out_path, meta=meta)

    print(f"[fit_nue_response] built df rows: {len(df)}")
    print(f"[fit_nue_response] fitted fields: {len(responses)}")
    print(f"[fit_nue_response] saved -> {saved}")
