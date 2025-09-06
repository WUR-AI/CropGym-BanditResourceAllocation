import os
import yaml
import pickle

from collections import Counter, defaultdict

import numpy as np

import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle

from pettingzoo import AECEnv

try:
    from pettingzoo.utils import AgentSelector
except ImportError:
    from pettingzoo.utils import agent_selector as AgentSelector

from cropgymzoo import _FIELDS_CONFIG, _CONFIG_PATH

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.utils.defaults import get_default_years
from cropgymzoo.utils.curriculum import make_default_stage_manager


def make_multi_env(
        *,
        training: bool = False,
        random_budget: bool = False,
        warm_up: int = 0,
        **kwargs
    ):
    env = MultiFieldEnv(
        training=training,
        random_budget=random_budget,
        warm_up=warm_up,
        **kwargs)          # build the base env
    return env


class MultiFieldEnv(AECEnv, EzPickle):

    fertilization_schedule = {
        "winterwheat": {
            165: {"clay": 100, "sand": 80, "silt": 80, "peat": 40},
            190: {"clay": 60, "sand": 40, "silt": 70, "peat": 40},
            225: {"clay": 60, "sand": 40, "silt": 60, "peat": 40},
        },
        "potato": {
            5: {"clay": 130, "sand": 130, "silt": 100, "peat": 130},
            15: {"clay": 55, "sand": 30, "silt": 55, "peat": 55},
            35: {"clay": 40, "sand": 40, "silt": 40, "peat": 40},
        },
        "sugarbeet": {
            5: {"clay": 40, "sand": 40, "silt": 40, "peat": 40},
            40: {"clay": 60, "sand": 50, "silt": 40, "peat": 50},
            75: {"clay": 30, "sand": 30, "silt": 20, "peat": 30},
        },
    }

    metadata = {
        "name": "CropGymZooEnv",
        "is_parallelizable": True,
    }
    """
    MARL Petting Zoo environment for Multi-agent RL with CropGym.
    The main idea is that each agent will take care of its own field, where the fields
    will (most likely) have heterogeneous crops and soil conditions.
    
    It requires a ::global_budget:: and ::warm_up:: to initialize.
    """

    def __init__(
            self,
            seed: int = 107,
            warm_up: int = 0,
            use_rl_warm_up_actions: bool = True,
            years: list = get_default_years(),
            training: bool = False,
            random_budget: bool = False,
            dict_obs: bool = True,
            shared_obs: bool = False,
            render: bool = False,
            stage: int = 0,
    ):
        EzPickle.__init__(
            self,
            seed=seed,
            warm_up=warm_up,
            years=years,
            training=training,
            random_budget=random_budget,
            shared_obs=shared_obs,
            render=render,
        )
        super().__init__()
        self.render_mode = None if not render else 'human'
        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)
        self.years = years
        self.training = training
        self.shared_obs = shared_obs
        self.dict_obs = dict_obs
        self.year = self.years[0]
        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)

        self.has_reset = False

        with open(_FIELDS_CONFIG) as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        self.n_agents = len(dict_fields)
        self.agents = [i for i in dict_fields.keys()]
        self.possible_agents = self.agents[:]
        # self._agent_selector = SkippingSelector(self.possible_agents) #   #
        self._agent_selector = AgentSelector(self.possible_agents)
        self.dead_step = {ag: False for ag in self.possible_agents}
        self.agent_to_keep = None

        self.current_step = {agent: 0 for agent in self.possible_agents}
        self.current_obs = {agent: {} for agent in self.possible_agents}

        # init important stuff
        self._init_fields()
        self._init_spaces()
        self._init_farm_variables()
        self._init_infos()

        self.random_budget = random_budget

        self.stage = stage

        self.global_budget = self._get_global_max_budget() if not self.random_budget else self._get_global_random_budget()
        self.global_budget_left = self.global_budget

        self.total_area = np.sum([self.get_field_size(agent) for agent in self.possible_agents])

        # Do some warm up episodes
        self.warm_up_infos = None
        if warm_up > 0:
            self.warm_up_infos = self._warm_up(warm_up)

    def reset(self, seed=None, options=None):
        # Check curriculum; only to be called in callbacks.
        if options is not None:
            if options.get('curriculum_stage', None) is not None:
                self.set_curriculum_stage(options.pop('curriculum_stage'))
        # reinitialize agents
        self.dead_step = {ag: False for ag in self.possible_agents}
        self.agents = self.possible_agents[:]

        # Still works sort-of parallel now: could change to date-based
        # Is it useful to change it to date-based? Maybe for future functionality and tasks
        # self.agent_selection = self._agent_selector.reset()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {ag: 0.0 for ag in self.possible_agents}
        self._cumulative_rewards = {ag: 0.0 for ag in self.possible_agents}
        self.terminations = {ag: False for ag in self.possible_agents}
        self.truncations = {ag: False for ag in self.possible_agents}

        self.current_step = {agent: 0 for agent in self.possible_agents}

        # reset infos and variables
        self._init_infos()

        # get the options before reset
        # If training, ignore all options and override with only year.
        # Good idea? Check for resource allocation too.
        options = options or {'year': 2010}
        if self.training:
            options = {'year': self.rng.choice(self.years)}
        # set year
        self.year = options['year']

        self.global_budget = (
            self._get_global_max_budget()
            if not self.random_budget
            else self._get_global_random_budget()
        )
        self.global_budget_left = self.global_budget

        # reset each field and get obs and infos again
        infos = {}
        for agent, env in self.fields.items():
            _, info = env.reset(seed=seed, options=options)
            infos[agent] = info

        self._update_infos(infos)

        # AECEnv doesn't return anything for reset()

   # following AEC env API
    def step(self, action: dict[str, int] | int | None):
        # need to call this apparently to make sure dead agent doesn't
        # get called again in the iterator

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(None)
            return

        agent = self.agent_selection
        is_last = self._agent_selector.is_last()

        # step the gymnasium PCSE parcel
        obs_parcel, rew_parcel, ter_parcel, tru_parcel, info_parcel = self.fields[agent].step(action)

        # scaled reward based on farm area
        rew_parcel = self._process_rewards(rew_parcel)

        # update global budget left
        self.global_budget_left = self._get_global_budget_left()

        # write results into the mandatory dicts
        self.rewards.update({agent: rew_parcel})
        self.terminations.update({agent: ter_parcel})
        self.truncations.update({agent: tru_parcel})
        self.infos.update({agent: info_parcel})

        self.agent_selection = self._agent_selector.next()


        # advance to next agent in the cycle
        self._accumulate_rewards()  # provided by AECEnv

        self.current_step[agent] += 1

        # AECEnv doesn't return anything for step()

    def observe(self, agent) -> dict:
        obs = self.fields[agent].unwrapped.observe()
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        if not self.dict_obs:
            return obs
        mask = self.fields[agent].unwrapped.action_mask()
        return {
            # "agent_id": str(agent),
            "observation": obs,
            **({"shared": self._build_shared()} if self.shared_obs else {}),
            "action_mask": mask,
        }

    def render(self):
        print(self)

    '''
    AECenv overrides
    '''

    def _was_dead_step(self, action) -> None:
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
            self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                assert self._skip_agent_selection is not None
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None

            # don't keep pointing to dead agent
            # not doing this will call env.last() and crash
            if self.agents:
                self._agent_selector._current_agent = self.possible_agents.index(self.agent_selection)
                self.agent_selection = self._agent_selector.next()

        self._clear_rewards()


    '''
    Callable helper functions and property
    '''

    def allocate_bandit_budgets(self, allocations):
        assert len(allocations) == len(self.fields)

        for agent, reduction in zip(self.possible_agents, allocations):
            agent_max = self.get_per_parcel_max_budget(agent)
            allocation = agent_max - reduction
            self.set_per_parcel_budget(agent, allocation)

        print(f'Allocated budget reductions of {allocations}')

    def set_curriculum_stage(self, stage: int):
        for agent in self.possible_agents:
            self.fields[agent].unwrapped.random_manager.set_stage(stage)
        self.stage = stage
        self.random_budget = self.fields[self.possible_agents[-1]].unwrapped.random_manager.budget

    def observation_space(self, _agent):
        return self.observation_spaces[_agent]

    def action_space(self, _agent):
        return self.action_spaces[_agent]

    def sample_observation_space_agent(self):
        return self.observation_space(self.possible_agents[0])['observation']

    def sample_masked_action(self, _agent):
        return self.fields[_agent].unwrapped.sample_masked_action()

    def get_field_env_with_idx(self, n: int):
        return self.fields[self.possible_agents[n]]

    def get_per_parcel_max_budget(self, _agent):
        return self.fields[_agent].unwrapped.max_budget_n

    def get_per_parcel_budget(self, _agent):
        return self.fields[_agent].unwrapped.budget_n

    def get_per_parcel_budget_left(self, _agent):
        return self.fields[_agent].unwrapped.budget_left

    def set_per_parcel_budget(self, _agent, budget):
        self.fields[_agent].unwrapped.set_budget(budget)

    def set_global_budget(self, budget: float):
        self.global_budget = budget

    def get_field_size(self, _agent):
        return self.fields[_agent].unwrapped.area

    def _get_mask(self, _agent):
        return self.fields[_agent].unwrapped.action_mask()

    def _update_infos(self, infos):
        for _agent in self.agents:
            self.infos[_agent] = infos[_agent]

    def _get_global_max_budget(self):
        return np.sum([self.get_per_parcel_max_budget(a) for a in self.possible_agents])

    def _get_global_budget(self):
        return np.sum([self.get_per_parcel_budget(a) for a in self.possible_agents])

    def _get_global_budget_left(self):
        return np.sum([self.get_per_parcel_budget_left(a) for a in self.possible_agents])

    def get_per_field_crop_code(self):
        return {
            a: self.fields[a].unwrapped.CROP_CODE_MAP[
                self.fields[a].unwrapped.crop
            ] for a in self.possible_agents
        }
    
    def get_per_field_crop_name(self):
        return {a: self.fields[a].unwrapped.crop for a in self.possible_agents}

    def get_per_field_soil_type(self):
        return {a: self.fields[a].unwrapped.soil_type for a in self.possible_agents}

    def get_per_field_crop_price(self):
        return {a: self.fields[a].unwrapped.crop_price for a in self.possible_agents}

    def get_per_field_fertilizer_price(self):
        return {a: self.fields[a].unwrapped.fertilizer_price for a in self.possible_agents}

    def get_initial_no3(self):
        return {a: self.fields[a].unwrapped.infos['NO3'][0] for a in self.possible_agents}

    def get_initial_nh4(self):
        return {a: self.fields[a].unwrapped.infos['NH4'][0] for a in self.possible_agents}

    def get_initial_n(self):
        return {a: self.fields[a].unwrapped.infos['NAVAIL'][0] for a in self.possible_agents}

    def _get_global_random_budget(self):
        # get dict of default max budget
        parcel_budgets = {a: self.get_per_parcel_max_budget(a) for a in self.possible_agents}

        lowest_budgets = {
            a: float(np.ceil(parcel_budgets[a] * 0.7 / 10) * 10)
            for a in self.possible_agents
        }

        # get random reductions by choice for each agent limited by the default budget of the parcel
        # change the logic of random allocation here if needed!
        choices = {}
        for (a, max_budget), (_, lowest_budget) in zip(parcel_budgets.items(), lowest_budgets.items()):
            # list_choice = [*np.arange(lowest_budget, max_budget, 10.)]
            # probs = self.left_heavy_weights(len(list_choice))
            # choices[a] = self.rng.choice(list_choice, p=probs)
            choice = self.rng.uniform(low=lowest_budget, high=max_budget)
            choices[a] = choice

        # set random budget reduction for each parcel
        for (_agent, choice), (_, budget) in zip(choices.items(), parcel_budgets.items()):
            self.set_per_parcel_budget(_agent, choice)

        self.set_global_budget(self._get_global_budget())

    @staticmethod
    def left_heavy_weights(x: int, steepness: float = 1.1) -> list[float]:
        idx = np.arange(x)  # 0, 1, 2, …, x-1
        raw = np.exp(-steepness * idx / (x - 1))  # exponential decay
        weights = raw / raw.sum()  # normalise to 1
        return weights.tolist()


    '''
    Init helpers
    '''

    def _init_farm_variables(self):
        self._emergence_doy = {ag: None for ag in self.agents}

    def _init_fields(self):
        """
        This is where we initialize the sub-environments where each agent will work.
        :return: a dict called "fields", filled with different CropGym envs
        """
        self.fields: dict[str, ParcelEnv] = {}
        # create each gymnasium cropgym env
        for n in self.agents:
            env = gym.make(
                n,
                seed=self.seed,  # set same seed for each parcel. Change?
                training=self.training,
                random_manager=make_default_stage_manager()
            )
            self.fields[n] : ParcelEnv = env
        print("Parcels initialized!")

    def _init_infos(self):
        self.infos = {ag: {} for ag in self.possible_agents}

    def _init_spaces(self):
        self.shared_space = gym.spaces.Dict(
            {k: gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32)
            for k in self._get_shared_obs_keys()}
        )

        # observation_spaces from locals, shared and action mask
        if self.dict_obs:
            self.observation_spaces = {
                ag: gym.spaces.Dict({
                    "observation": env.observation_space,
                    **({"shared": self.shared_space} if self.shared_obs else {}),
                    "action_mask": gym.spaces.MultiBinary(env.unwrapped.action_space.n),
                }) for ag, env in self.fields.items()
            }
        else:
            self.observation_spaces = {
                ag: env.observation_space
                for ag, env in self.fields.items()
            }

        # action space from individual parcels
        self.action_spaces = {agent: env.action_space
                              for agent, env in self.fields.items()}

    def _process_rewards(self, reward):
        """
        Uses the "weighted by area" policy reward.
        """
        field_size = self.get_field_size(self.agent_selection)

        # This is for the PNY reward
        # weighted_reward = (reward * field_size) / self.total_area

        # This is for the PNB reward
        weighted_reward = reward * field_size

        self.rewards[self.agent_selection] = weighted_reward
        # reward = rewards / self.total_area

        return weighted_reward

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

    @staticmethod
    def _is_at_date(dap: int, day: int):
        return day - 3 <= dap <= day + 3

    def _warm_up(self, warm_up_counter):
        print("Checking if warm up was done...")
        if os.path.isfile(os.path.join(_CONFIG_PATH, 'warm_up_infos.pkl')):
            with open(os.path.join(_CONFIG_PATH, 'warm_up_infos.pkl'), 'rb') as f:
                warm_up_infos = pickle.load(f)
            print("Loaded warm up info!")
            return warm_up_infos
        print("No file found...")
        warm_up_infos = defaultdict(dict)
        options = {}
        print('Starting warm up...')
        for i, _ in enumerate(range(warm_up_counter)):
            print('Start warm up iteration {}'.format(i))
            options['year'] = np.random.choice(self.years)
            self.reset(seed=self.seed, options=options)
            for agent in self.agent_iter():
                _, _, _, _, infos = self.last()
                action = self.farmers_practice(agent, infos)
                if self.terminations[agent]:
                    warm_up_infos[i].setdefault(agent, {})
                    warm_up_infos[i][agent] = infos
                    self.step(None)
                else:
                    self.step(action)
            print(self)
        print('Finished warm up...')
        print('Attempting to save pickle...')
        with open(os.path.join(_CONFIG_PATH, 'warm_up_infos.pkl'), 'wb') as f:
            pickle.dump(warm_up_infos, file=f)
        print('Successfully saved!')
        return warm_up_infos

    def farmers_practice(self, agent_name, infos):
        """Simple farmer rule-based fertilization schedule based on crop + soil."""
        crop = self.get_per_field_crop_name()[agent_name]
        soil = self.get_per_field_soil_type()[agent_name]

        # derive days after planting (DAP) from infos
        dap_plant = infos["DaysAfterPlanting"][-1]  # assume infos contains this
        fert = 0.0

        # check if today matches any scheduled DAP for this crop
        for day, soil_map in self.fertilization_schedule.get(crop, {}).items():
            if self._is_at_date(dap_plant, day):
                fert = soil_map.get(soil, 0.0)  # default 0 if soil not found
                break  # stop after first match

        return fert / 10  # align with action space

    def _get_each_agent_actions(self) -> dict[str, int]:
        """Rule-based fertiliser policy for warm-up episodes."""

        # today_doy = self.shared_space["DayOfYear"]
        actions = {}

        for ag, env in self.fields.items():
            info = env.unwrapped.get_latest_info
            crop = env.unwrapped._get_crop_code()
            n_applied_so_far = info("Naction")  # kg N ha-¹ already used
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

    def _get_crop_caps(self):
        name_caps = {
            "winterwheat": 240,
            "potato": 240,
            "sugarbeet": 150
        }

        code_caps = self._convert_crop_reference(name_caps)

        return {**name_caps, **code_caps}

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

        header = f"Farm status; sowing year {self.year} – budget left: {self.global_budget_left} / {self.global_budget} kg N"
        cols = ("Field (area[ha])", "Crop", "Date", "N applied", "Yield[t/ha]", "NUE", "Nsurp", "Profit")
        fmt_header = "{:18} {:12} {:10} {:>10} {:>15} {:>7} {:>7} {:>10}"
        lines = [header, fmt_header.format(*cols), "-" * 95]

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
                f"{field_id} ({env.unwrapped.area:.1f})",
                crop,
                val_date,
                safe(env, "Naction"),
                val_yield,
                safe(env, "Nue"),
                safe(env, "Nsurp"),
                safe(env, "Profit"),
            ]

            line = (
                f"{vals[0]:18} {vals[1]:12} {vals[2]:10} "
                f"{format_val(vals[3], 10)} "
                f"{format_val(vals[4], 15)} "
                f"{format_val(vals[5], 7)} "
                f"{format_val(vals[6], 7)} "
                f"{format_val(vals[7], 10)} "
            )
            lines.append(line)

        # add a small summary line
        summary = " | ".join(f"{c}:{n}" for c, n in sorted(crop_counts.items()))
        lines.append(f"\nCrop distribution → {summary}")

        return "\n".join(lines)
