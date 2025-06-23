import os

import numpy as np

from pcse.input.openmeteo import OpenMeteoWeatherDataProvider


class NoisyOpenMeteo(OpenMeteoWeatherDataProvider):
    SIGMA_RAIN  = 0.15
    SIGMA_OTHER = 0.6

    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng() if rng is None else rng

    def __call__(self, date):
        rec = super().__call__(date)

        r = self._rng

        # update fields directly; __setattr__ writes into the container
        rec.RAIN = np.clip(rec.RAIN * r.normal(1.0, self.SIGMA_RAIN), 0, 25)
        rec.IRRAD = np.clip(rec.IRRAD + r.normal(0, self.SIGMA_OTHER * abs(rec.IRRAD)), 0., 40e6)
        rec.TMIN = np.clip(rec.TMIN + r.normal(0, self.SIGMA_OTHER * abs(rec.TMIN)), -50., 60.)
        rec.TMAX = np.clip(rec.TMAX + r.normal(0, self.SIGMA_OTHER * abs(rec.TMAX)), -50., 60.)
        rec.VAP = np.clip(rec.VAP + r.normal(0, self.SIGMA_OTHER * abs(rec.VAP)), 0.06, 199.3)
        rec.TEMP = np.clip(rec.TEMP + r.normal(0, self.SIGMA_OTHER * abs(rec.TEMP)), -50., 60.)
        rec.WIND = np.clip(rec.WIND + r.normal(0, self.SIGMA_OTHER * abs(rec.WIND)), 0., 100.)

        for f in ["RAIN", "IRRAD", "TMAX", "TMIN", "VAP", "TEMP", "WIND"]:
            setattr(rec, f, round(getattr(rec, f), 2))

        return rec  # same object, now containing noisy values

