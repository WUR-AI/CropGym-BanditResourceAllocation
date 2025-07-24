import os

import numpy as np

from pcse.input.openmeteo import OpenMeteoWeatherDataProvider


class NoisyOpenMeteo(OpenMeteoWeatherDataProvider):
    """
    Class for perturbing the weather values in the OpenMeteo data provider.
    Used for training RL agents.
    """
    SIGMA_RAIN  = 0.15
    SIGMA_OTHER = 0.06

    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng() if rng is None else rng

    def __call__(self, date, member_id=0):
        rec = super().__call__(date)

        r = self._rng

        # update fields directly; __setattr__ writes into the container
        rec.RAIN = np.clip(rec.RAIN * r.normal(1.0, self.SIGMA_RAIN), 0, 25)
        rec.IRRAD = np.clip(rec.IRRAD + r.normal(0, self.SIGMA_OTHER * abs(rec.IRRAD)), 0., 40e6)
        rec.WIND = np.clip(rec.WIND + r.normal(0, self.SIGMA_OTHER * abs(rec.WIND)), 0., 100.)

        # sanity check
        # special random for temperature; ensure TMAX > TMIN
        # perturbing daily mean rather than the values
        t_mean = (rec.TMAX + rec.TMIN) / 2
        t_range = rec.TMAX - rec.TMIN

        t_mean += r.normal(0, self.SIGMA_OTHER * abs(t_mean))
        t_range += r.normal(0, self.SIGMA_OTHER * abs(t_range))

        t_range = max(t_range, 0.1)

        # get new noisy values from the mean
        rec.TMIN = np.clip(t_mean - 0.5 * t_range, -50., 60.)
        rec.TMAX = np.clip(t_mean + 0.5 * t_range, -50., 60.)


        rec.TMIN = np.clip(rec.TMIN + r.normal(0, self.SIGMA_OTHER * abs(rec.TMIN)), -50., 60.)
        rec.TMAX = np.clip(rec.TMAX + r.normal(0, self.SIGMA_OTHER * abs(rec.TMAX)), -50., 60.)

        rec.TEMP = np.clip(((rec.TMIN + rec.TMAX) / 2), -50., 60.)

        # get VAP from TEMP mean
        # idea is that warmer air holds more moisture
        rec.VAP = np.clip(rec.VAP +
                          r.normal(0, self.SIGMA_OTHER * abs(t_mean)) +
                          r.normal(0, 0.2 * self.SIGMA_OTHER * abs(rec.VAP)),
                          0.06, 199.3)


        for f in ["RAIN", "IRRAD", "TMAX", "TMIN", "VAP", "TEMP", "WIND"]:
            setattr(rec, f, round(getattr(rec, f), 2))

        return rec  # same object, now containing noisy values

