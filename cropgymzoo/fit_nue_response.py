import os
import re
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Hashable, Optional, Any

import numpy as np
import pandas as pd

from cropgymzoo import _DEFAULT_RESULTSDIR


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

    Returns columns:
      [farm_id, farm_global_id, farmer_id, field_id, year, red_level, region, crop, N, NUE, Nsurp, budget_rate]
    """
    rows = []

    for farm_key, farm_dict in data.items():
        region, year, farm_global_id, farmer_id, red_level = _parse_farm_key(farm_key)
        if region is None:
            # skip unknown key formats
            continue

        if farm_global_id is not None:
            farm_id = f"farm{int(farm_global_id)}"  # stable global identifier
        else:
            farm_id = f"{region}_farmer_{int(farmer_id)}"  # fallback

        if not isinstance(farm_dict, dict):
            continue

        for field_id, field_data in farm_dict.items():
            if not isinstance(field_data, dict):
                continue

            # --- seasonal N ---
            N_values = field_data.get("Naction", None)
            if N_values is None or len(N_values) == 0:
                continue
            N_season = float(N_values[-1])

            # --- NUE ---
            nue_values = field_data.get("NUE", None)
            if nue_values is None:
                nue_values = field_data.get("Nue", None)  # in case it is spelled differently
            NUE_final = float(nue_values[-1]) if nue_values is not None and len(nue_values) else np.nan

            # --- Nsurp ---
            nsurp_values = field_data.get("Nsurp", None)
            Nsurp_final = float(nsurp_values[-1]) if nsurp_values is not None and len(nsurp_values) else np.nan

            # --- crop (optional) ---
            crop = "unknown"
            crop_values = field_data.get("CropName", None)
            if crop_values is not None and len(crop_values):
                crop = str(crop_values[-1])

            # --- budget rate (optional; useful for debugging) ---
            budget_rate = field_data.get("BudgetTotal", None)
            if isinstance(budget_rate, (list, tuple, np.ndarray)) and len(budget_rate) > 0:
                budget_rate = float(budget_rate[-1])
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
                    "year": int(year),
                    "red_level": float(red_level) if red_level is not None else np.nan,
                    "region": region,
                    "crop": crop,
                    "N": float(N_season),
                    "NUE": float(NUE_final),
                    "Nsurp": float(Nsurp_final),
                    "budget_rate": float(budget_rate),
                }
            )

    return pd.DataFrame(rows)


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


def compute_feasible_bounds_per_field(
    *,
    responses: Dict[Tuple[Hashable, Hashable], FieldMetricResponses],
    max_crop_soil: Dict[Tuple[Hashable, Hashable], float],
    nue_range: Tuple[float, float] = (0.5, 0.9),
    nsurp_range: Tuple[float, float] = (0.0, 40.0),
) -> Dict[Tuple[Hashable, Hashable], Tuple[float, float]]:
    """Compute feasible (lb, ub) for each (farm_id, field_id).

    This converts your *metric* constraints into a simple bound on applied nitrogen N.

    Constraints enforced (per field):
      - NUE in [nue_range[0], nue_range[1]] (if NUE model exists)
      - Nsurp in [nsurp_range[0], nsurp_range[1]] (if Nsurp model exists)
      - 0 <= N <= max_crop_soil[(farm_id, field_id)]

    Parameters
    ----------
    responses:
        Output from `fit_field_metric_responses(...)` or `load_field_metric_responses(...)`.
    max_crop_soil:
        Dict mapping (farm_id, field_id) -> MAX_CROP_SOIL (kg/ha).
        You said you'll fill this externally.

    Returns
    -------
    Dict[(farm_id, field_id)] -> (lb, ub)
        If a field is infeasible under the fitted model, lb may exceed ub.
    """
    out: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = {}

    nue_min, nue_max = float(nue_range[0]), float(nue_range[1])
    ns_min, ns_max = float(nsurp_range[0]), float(nsurp_range[1])

    for key, fr in responses.items():
        bmax = max_crop_soil.get(key)
        if bmax is None:
            # If user didn't provide a max for this field, skip it.
            continue

        nue_model = fr.models.get("NUE")
        nsurp_model = fr.models.get("Nsurp")

        lb, ub = feasible_N_bounds_from_constraints(
            nue_model=nue_model,
            nsurp_model=nsurp_model,
            bmax_rate=float(bmax),
            nue_min=nue_min,
            nue_max=nue_max,
            nsurp_min=ns_min,
            nsurp_max=ns_max,
        )

        out[key] = (float(lb), float(ub))

    return out


def allocate_feasible_N(
    *,
    bounds: Dict[Tuple[Hashable, Hashable], Tuple[float, float]],
    total_budget: Optional[float] = None,
    mode: str = "max_sumN",
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
    - If total_budget is too small to satisfy all lower bounds, we still return lb_i,
      but you should treat it as infeasible (sum(lb) > total_budget).
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

    # If we already exceed the global budget, can't satisfy all lower bounds
    if used > B + 1e-9:
        # Return lower bounds (signals infeasible for the global budget)
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
    nsurp_range: Tuple[float, float] = (0.0, 40.0),
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

    alloc = allocate_feasible_N(bounds=bounds, total_budget=total_budget, mode=mode)

    info = {
        "infeasible_fields": infeasible_fields,
        "sum_lb": sum_lb,
        "sum_ub": sum_ub,
        "total_budget": None if total_budget is None else float(total_budget),
        "feasible_under_global_budget": bool(feasible_under_budget),
        "mode": str(mode),
        "n_fields": int(len(bounds)),
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
    nsurp_range: Tuple[float, float] = (0.0, 40.0),
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

    # Convert Napplied -> env reduction units vector
    reductions_vec = lp_alloc_to_env_reduction_vector(
        alloc_n,
        farm_id=farm_id,
        possible_agents=list(env.possible_agents),
        max_crop_soil=max_crop_soil,
        unit_kg=unit_kg,
        snap=snap,
    )

    info = {
        **lp_info,
        "alloc_n": alloc_n,
        "bounds": bounds,
        "reductions_vec": reductions_vec,
    }
    return reductions_vec, info


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
        help="Tag used in the output filename (e.g., 'ROT', 'MLP_FOCOPS', 'baseline').",
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