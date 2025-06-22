import os
import numpy as np

import gymnasium as gym



class PCSERandomizer(gym.Wrapper):

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

    def __init__(self, env):
        super().__init__(env=env)
        self.training = env.unwrapped.training
        self.rng = np.random.default_rng(self.env.unwrapped.seed)

    def _perturb_weather(self):
        w = self.env.unwrapped.wdp  # PCSE WDP object
        orig_getattr = w.__getattr__

        def noisy_attr(name):
            val = orig_getattr(name)
            if name == "RAIN":
                return np.clip(val * self.rng.normal(1.0, 0.1), 0, 100)  # move it 10% around the actual value
            if name in {"IRRAD", "TMAX", "TMIN", "WIND", "TEMP"}:
                return val + self.rng.normal(0, 0.03 * abs(val))
            return val

        w.__getattr__ = noisy_attr  # replace with perturbed value

    def _perturb_parameters(self):
        # get and filter relevant crop params
        crop_params = {key: val for key, val in self.env.unwrapped._parameter_provider._cropdata.items()
                       if key in self.CROP_PARAMS and isinstance(val, float)}

        for key, val in crop_params.items():
            # perturb by 2 percent
            self.env.unwrapped._parameter_provider.set_override(key, val*self.rng.normal(1.0, 0.02), check=False)

