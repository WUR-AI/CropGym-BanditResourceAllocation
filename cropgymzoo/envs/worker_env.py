import os
import yaml
import functools
import pickle

from collections import Counter

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv

from cropgymzoo import _FIELDS_CONFIG, _CONFIG_PATH

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.utils.defaults import get_default_years


class ParallelRLWorkers(ParallelEnv):
    metadata = {
        "name": "CropGymZooEnv",
    }

    def __init__(self,
                 seed: int = 107,
                 warm_up: int = 0,
                 global_budget: int = 400,
                 years: list = get_default_years(),):

        self.seed = seed
        self.years = years

        with open(_FIELDS_CONFIG) as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        self.n_agents = len(dict_fields)
        self.agents = [i for i in dict_fields.keys()]
        self.possible_agents = self.agents.copy()

        self.global_budget = global_budget
        self.global_budget_left = self.global_budget

        self._init_fields()
        self._init_spaces()
        self._init_farm_variables()
        self._init_infos()

        # Do some warm up episodes
        self.warm_up_infos = None
        if warm_up > 0:
            self.warm_up_infos = self._warm_up(warm_up)

    def reset(self, seed=None, options=None):

        # reset infos and variables
        self.global_budget_left = self.global_budget
        self._init_infos()

        # reinitialize agents
        self.agents = self.possible_agents.copy()

        # get obs and infos again
        local_obs, infos = {}, {}
        for ag, env in self.fields.items():
            o, i = env.reset(seed=seed, options=options)
            local_obs[ag], infos[ag] = o, i

        obs = {ag: {"local": local_obs[ag],
                    "shared": self.shared_space,
                    "action_mask": self._get_mask(ag)}
               for ag in self.agents}

        self._update_infos(infos)

        return obs, infos

    def step(self, actions: dict[str, int]):

        # init dict for each variable
        local_obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        # loop through agent steps
        for agent, env in self.fields.items():
            if agent in self.agents:  # has agent terminated?
                o, r, t, tr, i = env.step(actions[agent])
                local_obs[agent], rewards[agent] = o, r
                terminateds[agent], truncateds[agent], infos[agent] = t, tr, i

        # build MARL obs dictionary
        obs = {ag: {"local": local_obs[ag],
                    "shared": self._build_shared(),
                    "action_mask": self._get_mask(ag)}
               for ag in self.agents}

        # rebuild available agents
        self.agents = [agent for agent in self.agents if not (terminateds[agent] or truncateds[agent])]

        # update infos so we don't lose information on dying agents
        self._update_infos(infos)
        return obs, rewards, terminateds, truncateds, self.infos

    def render(self):
        print(self)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_mask(self, agent):
        return self.fields[agent].unwrapped.action_mask()

    def get_field_env(self, n: int):
        return self.fields[self.agents[n]]

    def set_global_budget(self, budget: int):
        self.global_budget = budget

    def _update_infos(self, infos):
        for agents in self.agents:
            self.infos[agents] = infos[agents]

    def _init_farm_variables(self):
        self._emergence_doy = {ag: None for ag in self.agents}

    def _init_fields(self):
        self.fields = {}
        # create each gymnasium cropgym env
        for n in self.agents:
            env = gym.make(n, seed=self.seed)  # set same seed for each parcel. Change?
            self.fields[n] : dict[ParcelEnv] = env
        print("Parcels initialized!")

    def _init_infos(self):
        self.infos = {}

    def _init_spaces(self):
        self.shared_space = gym.spaces.Dict(
            {k: gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32)
            for k in self._get_shared_obs_keys()}
        )

        # observation_spaces from locals, shared and action mask
        self.observation_spaces = {
            ag: gym.spaces.Dict({
                "local": env.observation_space,
                "shared": self.shared_space,
                "action_mask": gym.spaces.MultiBinary(env.unwrapped.action_space.n),
            }) for ag, env in self.fields.items()
        }

        # action space from individual parcels
        self.action_spaces = {agent: env.action_space
                              for agent, env in self.fields.items()}

    def _build_shared(self) -> dict[str, np.ndarray | list]:
        """
        Selected transformed features for shared observation.
        Change features in self._get_shared_obs_keys()
        """
        # change this functionality
        shared_obs = {}
        for feature in self._get_shared_obs_keys():
            # now a list. Maybe a dict?
            # Aggregate somehow?
            shared_obs[feature] = [env.unwrapped.get_latest_info(feature) for env in self.fields.values()]
        return shared_obs

    def _warm_up(self, warm_up_counter):
        print("Checking if warm up was done...")
        if os.path.isfile(os.path.join(_CONFIG_PATH, 'warm_up_infos.pkl')):
            with open(os.path.join(_CONFIG_PATH, 'warm_up_infos.pkl'), 'rb') as f:
                warm_up_infos = pickle.load(f)
            return warm_up_infos
        print("No file found...")
        warm_up_infos = {}
        options = {}
        print('Starting warm up...')
        for i, _ in enumerate(range(warm_up_counter)):
            print('Start warm up iteration {}'.format(i))
            options['year'] = np.random.choice(self.years)
            _, infos = self.reset(seed=self.seed, options=options)
            terminateds = {agent: False for agent in self.agents}
            while not all(terminateds.values()):
                actions = self._get_each_agent_actions()
                _, _, terminateds, _, infos = self.step(actions=actions)
            warm_up_infos[i] = infos
            print(self.__str__())
        print('Finished warm up...')
        print('Attempting to save pickle...')
        with open(os.path.join(_CONFIG_PATH, 'warm_up_infos.pkl'), 'wb') as f:
            pickle.dump(warm_up_infos, file=f)
        print('Successfully saved!')
        return warm_up_infos

    def _get_each_agent_actions(self) -> dict[str, int]:
        """Rule-based fertiliser policy for warm-up episodes."""

        # today_doy = self.shared_space["DayOfYear"]
        actions = {}

        for ag, env in self.fields.items():
            info = env.unwrapped.get_latest_info
            crop = env.unwrapped._get_crop_code()
            n_applied_so_far = n_applied_so_far = info("Naction")  # kg N ha-¹ already used
            cap = self._get_crop_caps()[crop]

            best_frac = 0.0

            # Figure out which split the parcel is currently in
            pending_dose = 0
            for trigger, frac in self._get_schedule()[crop]:
                if "doy" in trigger:
                    low, high = trigger["doy"]
                    if low <= env.unwrapped.date.timetuple().tm_yday <= high:
                        best_frac = max(best_frac, frac)
                elif "days_after_emerg" in trigger:
                    # 1.  Is emergence day already known?
                    emerg_doy = self._emergence_doy.get(ag)
                    today_doy = info("Date").timetuple().tm_yday

                    # 2.  If not, check whether the crop has now emerged.
                    dvs = info("DVS")                       # 0 = sowing, ~1 = anthesis
                    if emerg_doy is None and dvs is not None and dvs > 0.01:
                        emerg_doy = today_doy     # record first emergence
                        self._emergence_doy[ag] = emerg_doy
                    if emerg_doy is not None:
                        dae = (today_doy - emerg_doy) % 365
                        low, high = trigger["days_after_emerg"]
                        if low <= dae <= high:
                            best_frac = max(best_frac, frac)
                elif "leaf_stage" in trigger:
                    leaves = info("LAI")
                    low, high = trigger["leaf_stage"]
                    if low <= leaves <= high:
                        best_frac = max(best_frac, frac)
                else:
                    continue

            planned_total_by_now = cap * best_frac  # kg the crop *ought* to have
            deficit = max(0, planned_total_by_now - n_applied_so_far)
            # clip by remaining farm-level budget
            dose_today = min(deficit, self.global_budget_left, env.unwrapped.max_single_dose)

            if dose_today > 0:
                act = self._dose_to_action(env, dose_today)
                self.global_budget_left -= env.unwrapped.available_doses[act]
            else:
                act = 0  # no-op

            actions[ag] = act

        return actions

    def _dose_to_action(self, env, desired_kg):
        """Map kg N to the closest allowed discrete action ID."""
        doses = env.unwrapped.available_doses  # e.g. [0, 30, 60, 90]
        # pick the smallest dose ≥ desired, else highest available
        target = min((d for d in doses if d >= desired_kg), default=max(doses))
        return doses.index(target)  # assumes ascending order

    def _get_shared_obs_keys(self):
        return ["NO3", "NH4", "Yield", "BudgetLeft", "Naction", "NamountSO", "FertilizerPrice", "CropCode"]

    def _convert_crop_reference(self, dict_to_convert):
        # Get crop name ↔ code map
        code_map = next(iter(self.fields.values())).unwrapped.CROP_CODE_MAP

        # Invert it for code → name lookup
        code_to_name = {v: k for k, v in code_map.items()}

        # Add alternative keys (ints) that map to same caps
        code_caps = {code: dict_to_convert[name]
                     for code, name in code_to_name.items()
                     if name in dict_to_convert}

        return code_caps

    @functools.lru_cache(maxsize=None)
    def _get_crop_caps(self):
        name_caps = {
            "winterwheat": 240,
            "potato": 240,
            "sugarbeet": 150
        }

        code_caps = self._convert_crop_reference(name_caps)

        return {**name_caps, **code_caps}

    @functools.lru_cache(maxsize=None)
    def _get_schedule(self):
        schedule = {
            "winterwheat": [               # :contentReference[oaicite:0]{index=0}
                ({"doy": (45, 90)}, 0.17),   # late Feb – early Apr (tillering GS22-25)  ~40 kg
                ({"doy": (90, 120)}, 0.50),  # stem elong. GS31-32                     ~120 kg
                ({"doy": (120, 140)}, 0.33), # flag-leaf GS37-39                       ~80 kg
            ],
            "potato": [                    # :contentReference[oaicite:1]{index=1}
                ({"days_after_emerg": (0, 7)}, 0.40),   # at planting / emergence
                ({"days_after_emerg": (25, 40)}, 0.60), # tuber initiation / bulking
            ],
            "sugarbeet": [
            ({"days_after_emerg": (0, 10)}, 0.50),   # seed-bed / emergence
            ({"days_after_emerg": (20, 40)}, 1.00),  # finish N before canopy closes
            ],
        }

        crop_schedule = self._convert_crop_reference(schedule)

        return {**schedule, **crop_schedule}

    def __str__(self) -> str:
        """
        Return a multiline string such as

            Farm status – budget left: 350 / 400 kg N
            Field          Crop        N applied   Yield (t/ha)
            ---------------------------------------------------
            parcel_001     winterwheat       80          3.5
            parcel_002     potato            30            –
            parcel_003     sugarbeet          –            –

            Crop distribution → winterwheat:1 | potato:1 | sugarbeet:1
        """

        def safe(env, key, default="–"):
            try:
                val = env.unwrapped.get_latest_info(key)
                if val is None:
                    return default
                return val
            except Exception:
                return default

        # use a flexible formatter
        def format_val(val, width, prec=2):
            return f"{val:>{width}.{prec}f}" if isinstance(val, (int, float)) else f"{val:>{width}}"

        header = f"Farm status – budget left: {self.global_budget_left} / {self.global_budget} kg N"
        cols = ("Field", "Crop", "Date", "N applied", "Yield (t/ha)", "NUE", "Nsurp")
        fmt_header = "{:15} {:12} {:10} {:>10} {:>15} {:>10} {:>10}"
        lines = [header, fmt_header.format(*cols), "-" * 85]

        # build one row per parcel
        crop_counts = Counter()
        for field_id, env in self.fields.items():
            crop = env.unwrapped.crop
            crop_counts[crop] += 1

            val_yield = safe(env, "Yield")
            val_yield = val_yield / 1000 if isinstance(val_yield, (int, float)) else val_yield
            val_date = safe(env, "Date")
            val_date = val_date.strftime("%m/%d/%Y")

            vals = [
                field_id,
                crop,
                val_date,
                safe(env, "Naction"),
                val_yield,
                safe(env, "Nue"),
                safe(env, "Nsurp")
            ]

            line = (
                f"{vals[0]:15} {vals[1]:12} {vals[2]:10} "
                f"{format_val(vals[3], 10)} "
                f"{format_val(vals[4], 15)} "
                f"{format_val(vals[5], 10)} "
                f"{format_val(vals[6], 10)}"
            )
            lines.append(line)

        # add a small summary line
        summary = " | ".join(f"{c}:{n}" for c, n in sorted(crop_counts.items()))
        lines.append(f"\nCrop distribution → {summary}")

        return "\n".join(lines)


