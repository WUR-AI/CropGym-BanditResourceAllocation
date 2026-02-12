import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import date, datetime, timedelta

from cropgymzoo import _BASE_PATH

import requests


@dataclass(frozen=True)
class ForecastWindow:
    """
    Scalar summaries over a decision-date -> decision-date+N window.
    Units:
      - precipitation_mean_mm_per_day: mm/day (mean of daily sums over window)
      - tmin_mean_c: °C (mean of daily minima)
      - tmax_mean_c: °C (mean of daily maxima)
      - irradiation_mean_mj_m2_per_day: MJ/m²/day (mean of daily shortwave_radiation_sum)
    """
    precipitation_mean_mm_per_day: float
    tmin_mean_c: float
    tmax_mean_c: float
    irradiation_mean_mj_m2_per_day: float


class OpenMeteoHistoricalForecastStore:
    """
    Fetches historical forecasts from Open-Meteo and optionally caches the raw JSON.

    - If api_key is provided: uses customer-historical-forecast-api
    - Else: uses historical-forecast-api

    Uses DAILY variables (recommended for your 4 context scalars):
      precipitation_sum, temperature_2m_min, temperature_2m_max, shortwave_radiation_sum

    If you absolutely require HOURLY, say so and I’ll adjust this to hourly aggregation.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        timeout_s: float = 30.0,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = (api_key or "").strip() or None
        self.timeout_s = float(timeout_s)
        self.session = session or requests.Session()

        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # in-memory cache to avoid repeated disk reads within a run
        self._mem_cache: Dict[str, Dict[str, Any]] = {}

    def _base_url(self) -> str:
        if self.api_key:
            return "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
        return "https://historical-forecast-api.open-meteo.com/v1/forecast"

    def _archive_url(self) -> str:
        if self.api_key:
            return "https://customer-archive-api.open-meteo.com/v1/archive"
        return "https://archive-api.open-meteo.com/v1/archive"

    @staticmethod
    def _should_use_archive(start_date: date) -> bool:
        # Open-Meteo Historical Forecast API coverage starts at 2016-01-01 (per your observation).
        return start_date < date(2016, 1, 1)

    @staticmethod
    def _key(lat: float, lon: float, start: date, end: date, daily_vars: Tuple[str, ...]) -> str:
        # round coordinates to avoid cache misses due to float noise
        return f"lat={lat:.5f}_lon={lon:.5f}_{start.isoformat()}_{end.isoformat()}_daily={','.join(daily_vars)}"

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe = key.replace("/", "_")
        return self.cache_dir / f"{safe}.json"

    def fetch_daily_raw(
        self,
        *,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
        daily_vars: Tuple[str, ...] = (
            "precipitation_sum",
            "temperature_2m_min",
            "temperature_2m_max",
            "shortwave_radiation_sum",
        ),
        timezone: str = "UTC",
    ) -> Dict[str, Any]:
        """
        Returns the Open-Meteo JSON response (dict).
        """
        key = self._key(lat, lon, start_date, end_date, daily_vars)

        # in-memory cache
        if key in self._mem_cache:
            return self._mem_cache[key]

        # disk cache
        p = self._cache_path(key)
        if p and p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self._mem_cache[key] = data
            return data

        use_archive = self._should_use_archive(start_date)
        url = self._archive_url() if use_archive else self._base_url()

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": ",".join(daily_vars),
            "timezone": timezone,
        }
        if self.api_key:
            # Open-Meteo typically expects api_key in querystring for customer endpoints
            params["apikey"] = self.api_key

        resp = self.session.get(url, params=params, timeout=self.timeout_s)

        # If historical-forecast rejects (400/401/403/etc.), fallback to archive.
        if resp.status_code >= 400 and not use_archive:
            url2 = self._archive_url()
            resp2 = self.session.get(url2, params=params, timeout=self.timeout_s)
            resp2.raise_for_status()
            data = resp2.json()
        else:
            resp.raise_for_status()
            data = resp.json()

        # Some locations/periods return forecast arrays containing None values.
        # In that case, fall back to the ARCHIVE API (observations).
        none_ratio = self._none_ratio_in_daily(data, daily_vars)
        if none_ratio > 0.0 and not use_archive:
            url2 = self._archive_url()
            resp2 = self.session.get(url2, params=params, timeout=self.timeout_s)
            resp2.raise_for_status()
            data2 = resp2.json()

            # Only replace if the archive result is strictly better (fewer Nones)
            if self._none_ratio_in_daily(data2, daily_vars) < none_ratio:
                data = data2

        self._mem_cache[key] = data
        if p:
            p.write_text(json.dumps(data), encoding="utf-8")

        return data

    @staticmethod
    def _none_ratio_in_daily(data: Dict[str, Any], daily_vars: Tuple[str, ...]) -> float:
        daily = (data or {}).get("daily") or {}
        vals = []
        for v in daily_vars:
            arr = daily.get(v)
            if arr is None:
                # treat missing variable as fully None
                continue
            vals.extend(list(arr))
        if not vals:
            return 1.0
        none_count = sum(x is None for x in vals)
        return none_count / float(len(vals))

    def _daily_has_any_signal(self, data: Dict[str, Any], daily_vars: Tuple[str, ...], max_none_ratio: float = 0.95) -> bool:
        # pass if we have at least some non-None values
        return self._none_ratio_in_daily(data, daily_vars) < max_none_ratio

    @staticmethod
    def _neighbor_offsets_deg():
        # ~0.02° ≈ 2.2 km lat; lon distance depends on latitude but ok for small retries
        steps = [0.0, 0.04, -0.04, 0.08, -0.08]
        for dlat in steps:
            for dlon in steps:
                if dlat == 0.0 and dlon == 0.0:
                    continue
                yield dlat, dlon

    @staticmethod
    def _mean(vals: list) -> float:
        # avoid numpy dependency here
        clean = [v for v in vals if v is not None]
        if not clean:
            return float("nan")
        return float(sum(clean) / len(clean))

    def summarize_window(
        self,
        *,
        lat: float,
        lon: float,
        decision_date: date,
        horizon_days: int,
        timezone: str = "UTC",
    ) -> ForecastWindow:
        """
        Decision date inclusive, for `horizon_days` days.
        Example: horizon_days=7 -> [d, d+6] inclusive.
        """
        if horizon_days <= 0:
            raise ValueError("horizon_days must be >= 1")

        start = decision_date
        end = decision_date + timedelta(days=horizon_days - 1)

        raw = self.fetch_daily_raw(
            lat=lat,
            lon=lon,
            start_date=start,
            end_date=end,
            timezone=timezone,
        )

        daily = raw.get("daily", {}) or {}

        pr = daily.get("precipitation_sum", []) or []
        tmin = daily.get("temperature_2m_min", []) or []
        tmax = daily.get("temperature_2m_max", []) or []
        rad = daily.get("shortwave_radiation_sum", []) or []  # usually MJ/m²/day in Open-Meteo daily

        return ForecastWindow(
            precipitation_mean_mm_per_day=self._mean(pr),
            tmin_mean_c=self._mean(tmin),
            tmax_mean_c=self._mean(tmax),
            irradiation_mean_mj_m2_per_day=self._mean(rad),
        )