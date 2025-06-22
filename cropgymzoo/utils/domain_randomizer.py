import os

import types
import numpy as np

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

    def perturb_weather(self):
        w = self.env.wdp  # PCSE WDP object
        orig_call = w.__call__

        rng = self.rng

        def noisy_call(self_, date):
            rec = orig_call(date)  # original record (namedtuple)

            # -----  jitter individual fields  ---------------------------------
            rain = np.clip(rec.RAIN * rng.normal(1.0, 0.10), 0, 100)
            irrad = rec.IRRAD + rng.normal(0, 0.03 * abs(rec.IRRAD))
            tmax = rec.TMAX + rng.normal(0, 0.03 * abs(rec.TMAX))
            tmin = rec.TMIN + rng.normal(0, 0.03 * abs(rec.TMIN))
            temp = rec.TMIN + rng.normal(0, 0.03 * abs(rec.TEMP))
            wind = rec.TMIN + rng.normal(0, 0.03 * abs(rec.WIND))

            # namedtuple --> _replace is the safest way to make a new instance
            rec = rec._replace(RAIN=rain, IRRAD=irrad, TMAX=tmax, TMIN=tmin, TEMP=temp, WIND=wind)
            return rec

        w.__call__ = types.MethodType(noisy_call, w)  # replace with perturbed value

    def perturb_parameters(self):
        # get and filter relevant crop params
        crop_params = {key: val for key, val in self.env._parameter_provider._cropdata.items()
                       if key in self.CROP_PARAMS and isinstance(val, float)}

        for key, val in crop_params.items():
            # perturb by 2 percent
            self.env._parameter_provider.set_override(key, val*self.rng.normal(1.0, 0.02), check=False)

    def perturb_carbon_dioxide(self, co2):
        return co2 * self.rng.normal(1.0, 0.1)

