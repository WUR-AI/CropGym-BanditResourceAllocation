import os
from copy import deepcopy

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

        # don't change the actual values in the WDP!
        copied_rec = deepcopy(rec)

        # update fields dicopied_rectly; __setattr__ writes into the container
        copied_rec.RAIN = np.clip(copied_rec.RAIN * r.normal(1.0, self.SIGMA_RAIN), 0, 25)
        copied_rec.IRRAD = round(np.clip(copied_rec.IRRAD + r.normal(0, self.SIGMA_OTHER * abs(copied_rec.IRRAD)), 0., 40e6), 0)
        copied_rec.WIND = np.clip(copied_rec.WIND + r.normal(0, self.SIGMA_OTHER * abs(copied_rec.WIND)), 0., 100.)

        # sanity check
        # special random for temperature; ensure TMAX > TMIN
        # perturbing daily mean rather than the values
        t_mean = (copied_rec.TMAX + copied_rec.TMIN) / 2
        t_range = copied_rec.TMAX - copied_rec.TMIN

        t_mean += r.normal(0, self.SIGMA_OTHER * abs(t_mean))
        t_range += r.normal(0, self.SIGMA_OTHER * abs(t_range))

        t_range = max(t_range, 0.1)

        # get new noisy values from the mean
        copied_rec.TMIN = np.clip(t_mean - 0.5 * t_range, -50., 60.)
        copied_rec.TMAX = np.clip(t_mean + 0.5 * t_range, -50., 60.)

        copied_rec.TEMP = np.clip(((copied_rec.TMIN + copied_rec.TMAX) / 2), -50., 60.)

        # get VAP from TEMP mean
        # idea is that warmer air holds more moisture
        copied_rec.VAP = np.clip(copied_rec.VAP +
                          r.normal(0, self.SIGMA_OTHER * abs(t_mean)) +
                          r.normal(0, 0.2 * self.SIGMA_OTHER * abs(copied_rec.VAP)),
                          0.06, 199.3)


        for f in ["RAIN", "IRRAD", "TMAX", "TMIN", "VAP", "TEMP", "WIND"]:
            setattr(copied_rec, f, round(getattr(copied_rec, f), 2))

        return copied_rec  # same copied object, now containing noisy values

