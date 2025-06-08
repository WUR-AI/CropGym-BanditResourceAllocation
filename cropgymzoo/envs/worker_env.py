import os
import yaml
import functools

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv

from cropgymzoo import _FIELDS_CONFIG

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.utils.defaults import get_default_years


class ParallelRLWorkers(ParallelEnv):
    metadata = {
        "name": "CropGymZooEnv",
    }

    def __init__(self,
                 seed: int = 107,
                 warm_up: int = 100,
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

        self._init_fields()
        self._init_spaces()
        self._init_farm_variables()

        if warm_up > 0:
            self.warm_up_infos = self._warm_up(warm_up)



    def reset(self, seed=None, options=None):
        # reinitialize agents
        self.agents = self.possible_agents.copy()
        locals_, infos = {}, {}
        for ag, env in self.fields.items():
            o, i = env.reset(seed=seed, options=options)
            locals_[ag], infos[ag] = o, i

        obs = {ag: {"local": locals_[ag],
                    "shared": self.shared_space,
                    "action_mask": self._get_mask(ag)}
               for ag in self.agents}

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
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        pass

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

    def _init_farm_variables(self):
        self.global_budget_left = self.global_budget

    def _init_fields(self):
        self.fields = {}
        # create each gymnasium cropgym env
        for n in self.agents:
            env = gym.make(n, seed=self.seed)  # set same seed for each parcel. Change?
            self.fields[n] : dict[ParcelEnv] = env
        print("Parcels initialized!")

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

    def _build_context(self, obs):
        ...

    def _warm_up(self, warm_up_counter):
        warm_up_infos, infos = {}, {}
        options = {}
        print('Starting warm up...')
        for i, _ in enumerate(range(warm_up_counter)):
            options['year'] = np.random.choice(self.years)
            _, _ = self.reset(seed=self.seed, options=options)
            while self.agents is not False:
                actions = self._get_each_agent_actions()
                _, _, _, _, infos = self.step(actions=actions)
            warm_up_infos[i] = infos
        print('Finished warm up...')
        return warm_up_infos

    def _get_each_agent_actions(self) -> dict[str, int]:
        """Rule-based fertiliser policy for warm-up episodes."""

        # today_doy = self.shared_space["DayOfYear"]
        actions = {}

        for ag, env in self.fields.items():
            info = env.unwrapped.get_latest_info
            crop = info("CropCode")
            n_applied_so_far = info("Naction_total")  # kg N ha-¹ already used
            cap = self._get_crop_caps()[crop]

            # Figure out which split the parcel is currently in
            pending_dose = 0
            for trigger, frac in self._get_schedule()[crop]:
                if "doy" in trigger:
                    low, high = trigger["doy"]
                    if low <= env.unwrapped.date.timetuple().tm_yday <= high:
                        planned_total_by_now = cap * frac
                elif "days_after_emerg" in trigger:
                    dae = info("DaysAfterEmergence")
                    low, high = trigger["days_after_emerg"]
                    if low <= dae <= high:
                        planned_total_by_now = cap * frac
                elif "leaf_stage" in trigger:
                    leaves = info("LeafStage")
                    low, high = trigger["leaf_stage"]
                    if low <= leaves <= high:
                        planned_total_by_now = cap * frac
                else:
                    continue

            # how much more N does this parcel still need today?
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
        doses = list(env.unwrapped.budget_left)  # e.g. [0, 30, 60, 90]
        # pick the smallest dose ≥ desired, else highest available
        target = min((d for d in doses if d >= desired_kg), default=max(doses))
        return doses.index(target)  # assumes ascending order

    def _get_shared_obs_keys(self):
        return ["NO3", "NH4", "Yield", "BudgetLeft", "Naction", "NamountSO", "FertilizerPrice", "CropCode"]

    @functools.lru_cache(maxsize=None)
    def _get_crop_caps(self):
            return {"winterwheat": 240, "potato": 240, "sugarbeet": 150}

    @functools.lru_cache(maxsize=None)
    def _get_schedule(self):
        return {
            "winterwheat": [               # :contentReference[oaicite:0]{index=0}
                ({"doy": (45, 90)}, 0.17),   # late Feb – early Apr (tillering GS22-25)  ~40 kg
                ({"doy": (90, 120)}, 0.50),  # stem elong. GS31-32                     ~120 kg
                ({"doy": (120, 140)}, 0.33), # flag-leaf GS37-39                       ~80 kg
            ],
            "potato": [                    # :contentReference[oaicite:1]{index=1}
                ({"days_after_emerg": (0, 7)}, 0.40),   # at planting / emergence
                ({"days_after_emerg": (25, 40)}, 0.60), # tuber initiation / bulking
            ],
            "sugarbeet": [                 # :contentReference[oaicite:2]{index=2}
                ({"doy": (80, 105)}, 0.50),   # pre-plant – 4-leaf
                ({"leaf_stage": (6, 9)}, 0.50) # 6- to 8-leaf (≈ canopy closure)
            ],
        }

    def __str__(self) -> str:
        from collections import Counter
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

        # small helper for “safe” look-ups
        def safe(env, key, default="–"):
            try:
                return env.unwrapped.get_latest_info(key)
            except Exception:
                return default

        header = f"Farm status – budget left: {self.global_budget_left} / {self.global_budget} kg N"
        cols = ("Field", "Crop", "N applied", "Yield (t/ha)")
        fmt = "{:15} {:12} {:>10} {:>12}"
        lines = [header, fmt.format(*cols), "-" * 55]

        # build one row per parcel
        crop_counts = Counter()
        for field_id, env in self.fields.items():
            crop = safe(env, "CropCode")
            crop_counts[crop] += 1

            line = fmt.format(
                field_id,
                crop,
                safe(env, "Naction_total"),
                safe(env, "Yield"),
            )
            lines.append(line)

        # add a small summary line
        summary = " | ".join(f"{c}:{n}" for c, n in sorted(crop_counts.items()))
        lines.append(f"\nCrop distribution → {summary}")

        return "\n".join(lines)


