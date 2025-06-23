import os

import numpy as np

from pcse.input.openmeteo import OpenMeteoWeatherDataProvider


class NoisyOpenMeteo(OpenMeteoWeatherDataProvider):
    SIGMA_RAIN  = 0.15
    SIGMA_OTHER = 0.10

    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng() if rng is None else rng

    def __call__(self, date):
        rec = super().__call__(date)

        r = self._rng

        # update fields directly; __setattr__ writes into the container
        rec.RAIN = np.clip(rec.RAIN * r.normal(1.0, self.SIGMA_RAIN), 0, 100)
        rec.IRRAD = rec.IRRAD + r.normal(0, self.SIGMA_OTHER * abs(rec.IRRAD))
        rec.TMAX = rec.TMAX + r.normal(0, self.SIGMA_OTHER * abs(rec.TMAX))
        rec.TMIN = rec.TMIN + r.normal(0, self.SIGMA_OTHER * abs(rec.TMIN))
        rec.TEMP = rec.TEMP + r.normal(0, self.SIGMA_OTHER * abs(rec.TEMP))
        rec.WIND = rec.WIND + r.normal(0, self.SIGMA_OTHER * abs(rec.WIND))

        for f in ["RAIN", "IRRAD", "TMAX", "TMIN", "TEMP", "WIND"]:
            setattr(rec, f, round(getattr(rec, f), 2))

        return rec  # same object, now containing noisy values

