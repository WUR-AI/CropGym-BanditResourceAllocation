import os

import types
import numpy as np

from pcse.input.openmeteo import OpenMeteoWeatherDataProvider

import gymnasium as gym



class PCSERandomizer:

    CROP_PARAMS = ["TBASE",  # lower threshold temperature for ageing of leaves
                   "SPAN",  # life span of leaves growing at 35 Celsius
                   "TDWI",  # initial total crop dry weight
                   "CVL",  # efficiency of conversion into leaves
                   "CVO",  # efficiency of conversion into storage organs
                   "CVR",  # efficiency of conversion into roots
                   "CVS",  # efficiency of conversion into stems
                   "PERDL",  # maximum relative death rate of leaves due to water stress
                   "RGRLAI_MIN"  # maximum relative increase in LAI
                   "RNUPTAKEMAX",  # Maximum rate of daily nitrogen uptake
                   "DVS_N_TRANSL"  # development stage above which N translocation to storage organs does occur
                   ]

    def __init__(self, env: gym.Env):
        self.env = env
        self.training = env.training
        self.rng = np.random.default_rng(self.env.seed)

    def perturb_parameters(self):
        # get and filter relevant crop params
        crop_params = {key: val for key, val in self.env._parameter_provider._cropdata.items()
                       if key in self.CROP_PARAMS and isinstance(val, float)}

        for key, val in crop_params.items():
            # perturb by 2 percent
            self.env._parameter_provider.set_override(key, val*self.rng.normal(1.0, 0.02), check=False)

    def perturb_carbon_dioxide(self, co2):
        return co2 * self.rng.normal(1.0, 0.1)


class NoisyOpenMeteo(OpenMeteoWeatherDataProvider):
    SIGMA_RAIN_REL  = 0.10
    SIGMA_OTHER_REL = 0.03

    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng() if rng is None else rng

    def __call__(self, date):
        rec = super().__call__(date)

        r = self._rng

        # update fields directly; __setattr__ writes into the container
        rec.RAIN = np.clip(rec.RAIN * r.normal(1.0, 0.10), 0, 100)
        rec.IRRAD = rec.IRRAD + r.normal(0, 0.03 * abs(rec.IRRAD))
        rec.TMAX = rec.TMAX + r.normal(0, 0.03 * abs(rec.TMAX))
        rec.TMIN = rec.TMIN + r.normal(0, 0.03 * abs(rec.TMIN))
        rec.TEMP = rec.TEMP + r.normal(0, 0.03 * abs(rec.TEMP))
        rec.WIND = rec.WIND + r.normal(0, 0.03 * abs(rec.WIND))

        for f in ["RAIN", "IRRAD", "TMAX", "TMIN", "TEMP", "WIND"]:
            setattr(rec, f, round(getattr(rec, f), 2))

        return rec  # same object, now containing noisy values

