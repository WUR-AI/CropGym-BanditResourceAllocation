import os
import yaml
import functools

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv

from cropgymzoo import _FIELDS_CONFIG

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.envs.allocation_env import AllocationBandit

class ParallelRLWorkers(ParallelEnv):
    metadata = {
        "name": "CropGymZooEnv",
    }

    def __init__(self,
                 seed: int = 107,
                 warm_up: int = 100,
                 global_budget: int = 400,
                 allocator: str = 'random',
                 allocator_env: AllocationBandit = None,):

        self.seed = seed

        with open(_FIELDS_CONFIG) as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        self.n_agents = len(dict_fields)
        self.agents = [i for i in dict_fields.keys()]
        self.possible_agents = self.agents.copy()

        # either 'random' or 'bandit'
        self.allocation_type = allocator
        self.allocator_agent = allocator_env
        if self.allocator_agent == 'bandit':
            assert self.allocator_agent is not None

        self.global_budget = global_budget

        self._init_fields()
        self._init_spaces()
        self._init_farm_variables()

        if warm_up:
            self._warm_up()



    def reset(self, seed=None, options=None):
        assert 'global_budget' in options, "Please reset env with global_budget key!"

        self.global_budget = options.get('global_budget')

        locals_, infos = {}, {}
        for ag, env in self.fields.items():
            o, i = env.reset(seed=seed, options=options)
            locals_[ag], infos[ag] = o, i

        obs = {ag: {"local": locals_[ag],
                    "shared": self.shared_space,
                    "action_mask": self._get_mask(ag)}
               for ag in self.agents}

        if self.allocation_type == 'random':
            # please fill in here
            allocations = self._allocate_random_budgets()
        else:
            context = self._build_context(obs)
            allocations = self.allocator_agent.reset(options=context)

        return obs, infos

    def step(self, actions: dict[str, int]):

        # init dict for each variable
        locals_, rews, terms, truncs, infos = {}, {}, {}, {}, {}

        # loop through agent steps
        for ag, env in self.fields.items():
            o, r, t, tr, i = env.step(actions[ag])
            # self._apply_dose(ag, actions[ag])          # update budget
            locals_[ag], rews[ag] = o, r
            terms[ag], truncs[ag], infos[ag] = t, tr, i

        obs = {ag: {"local": locals_[ag],
                    "shared": self.shared_space,
                    "action_mask": self._get_mask(ag)}
               for ag in self.agents}

        self.agents = [ag for ag in self.agents if not (terms[ag] or truncs[ag])]
        return obs, rews, terms, truncs, infos

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

    def _init_farm_variables(self):
        self.global_budget_left = self.global_budget

    def _init_fields(self):
        self.fields = {}
        # create each gymnasium env
        for n in self.agents:
            env = gym.make(n, seed=self.seed)  # set same seed for each field. Change?
            self.fields[n] : dict[ParcelEnv] = env

    def _init_spaces(self):
        # TODO

        self.shared_space = self._build_shared()

        self.observation_spaces = {
            ag: gym.spaces.Dict({
                "local": env.observation_space,
                "shared": self.shared_space,
                "action_mask": env.unwrapped.action_mask(),
            }) for ag, env in self.fields.items()
        }
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

    def _warm_up(self):
        ...

    def _get_shared_obs_keys(self):
        return ["NO3", "NH4", "Yield", "BudgetLeft", "Naction", "NamountSO", "FertilizerPrice", "CropCode"]

    def _allocate_random_budgets(self) -> dict[str, float]:
        """
        Return a dict {agent_id: kg_budget} that d sums to self.global_budget
        and d never exceeds each field’s legal ceiling.
        Requires:
            • self.fields        : dict[str, ParcelEnv]
            • self.global_budget : float   (kg for this season)
            • self.crop_caps     : dict[str, float]  # e.g. {'wheat':240,…}
        """
        rng = np.random.default_rng()  # or use self.np_random
        agents = list(self.fields.keys())
        n = len(agents)
        q = 10 # kg/ha

        # ----------------------------------------------------------------
        # 1) find per-field ceiling  m_j  from either the parcel or a lookup
        # ----------------------------------------------------------------
        cap_q = np.empty(n, dtype=int)
        for k, ag in enumerate(agents):
            env = self.fields[ag]
            # priority 1: an attribute on the parcel env
            # TODO check this logic
            if hasattr(env, "max_allowed_kg"):
                caps = env.max_allowed_kg
            else:  # fallback from crop type
                caps = self._get_crop_caps[env.unwrapped.crop]  # e.g. 240, 150 …
            cap_q[k] = int(np.floor(caps / q))

        # ---- 2) global budget in quanta -------------------------------
        Q_total = int(np.round(self.global_budget / q))
        if Q_total > cap_q.sum():
            raise ValueError("Budget exceeds joint crop ceilings")

        alloc_q = np.zeros(n, dtype=int)
        remaining_q = Q_total
        remaining_idx = np.arange(n)

        # ---- 3) iterative multinomial with clipping -------------------
        while remaining_q > 0 and remaining_idx.size:
            probs = rng.dirichlet(np.ones(remaining_idx.size))
            # sample how many quanta each remaining field *would* get
            proposal_q = rng.multinomial(remaining_q, probs)
            room_q = cap_q[remaining_idx] - alloc_q[remaining_idx]
            applied_q = np.minimum(proposal_q, room_q)  # clip
            alloc_q[remaining_idx] += applied_q
            remaining_q -= applied_q.sum()
            # keep only fields that can still accept quanta
            remaining_idx = remaining_idx[(room_q - applied_q) > 0]

        if remaining_q > 0:
            raise RuntimeError("Could not allocate all quanta; all fields full")

        return {ag: float(alloc_q[k] * q) for k, ag in enumerate(agents)}

    @functools.lru_cache(maxsize=None)
    def _get_crop_caps(self):
            return {"wheat": 240, "potato": 240, "beet": 150}
