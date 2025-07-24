import os
import yaml
import functools
import pickle

from collections import Counter

import numpy as np

import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv, AECEnv
from pettingzoo.utils import agent_selector

from cropgymzoo import _FIELDS_CONFIG, _CONFIG_PATH

from cropgymzoo.envs.singular_env import ParcelEnv
from cropgymzoo.utils.defaults import get_default_years


class MultiFieldEnv(AECEnv, EzPickle):
    metadata = {
        "name": "CropGymZooEnv",
    }
    """
    MARL Petting Zoo environment for Multi-agent RL with CropGym.
    The main idea is that each agent will take care of its own field, where the fields
    will (most likely) have heterogeneous crops and soil conditions.
    
    It requires a ::global_budget:: and ::warm_up:: to initialize.
    """

    def __init__(self,
                 seed: int = 107,
                 warm_up: int = 0,
                 years: list = get_default_years(),
                 training: bool = False,
                 random_budget: bool = False,
                 shared_obs: bool = False,
                 render: bool = False,):
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

        self.has_reset = False

        with open(_FIELDS_CONFIG) as f:
            dict_fields = yaml.load(f, Loader=yaml.SafeLoader)

        self.n_agents = len(dict_fields)
        self.agents = [i for i in dict_fields.keys()]
        self.possible_agents = self.agents.copy()
        # self._agent_selector = SkippingSelector(self.possible_agents) #   #
        self._agent_selector = agent_selector(self.possible_agents)
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

        self.global_budget = self._get_global_max_budget() if not self.random_budget else self._get_global_random_budget()
        self.global_budget_left = self.global_budget

        self.total_area = np.sum([self.get_field_size(agent) for agent in self.possible_agents])

        # Do some warm up episodes
        self.warm_up_infos = None
        if warm_up > 0:
            self.warm_up_infos = self._warm_up(warm_up)

    def reset(self, seed=None, options=None):
        # reinitialize agents
        self.dead_step = {ag: False for ag in self.possible_agents}
        self.agents = self.possible_agents[:]

        # Still works sort-of parallel now: could change to date-based
        # Is it useful to change it to date-based? Maybe for future functionality and tasks
        self._agent_selector.reinit(self.agents)
        # self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()

        self.rewards = {ag: 0.0 for ag in self.possible_agents}
        self._cumulative_rewards = {ag: 0.0 for ag in self.possible_agents}
        self.terminations = {ag: False for ag in self.possible_agents}
        self.truncations = {ag: False for ag in self.possible_agents}

        self.current_step = {agent: 0 for agent in self.possible_agents}

        # reset infos and variables
        self._init_infos()

        # get the options before reset
        options = options or {'year': self.rng.choice(self.years)}

        # TODO Pass global budget into options if using allocator.
        self.global_budget = self._get_global_max_budget() if not self.random_budget else self._get_global_random_budget()
        # allocation must be a dict
        if 'allocation' in options:
            for _agent, budget in options['allocation'].items():
                self.set_per_parcel_budget(_agent, budget)
            self.set_global_budget(self._get_global_budget())

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

        # called = False
        if (self.terminations[self.agent_selection] and self.dead_step[self.agent_selection]) or action is None:
            self.agent_selection = self._agent_selector.next()

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # self._was_dead_step(None)
            self.agents.remove(self.agent_selection)
            self.dead_step[self.agent_selection] = True
            self._agent_selector.reinit(self.agents)
            #
            # self.agent_selection = self._agent_selector.next()

            # return infos for logging
            self.infos = {agent: self.fields[agent].unwrapped.infos for agent in self.possible_agents}

            return

        agent = self.agent_selection

        self.current_step[agent] += 1

        # step the gymnasium PCSE parcel
        obs_parcel, rew_parcel, ter_parcel, tru_parcel, info_parcel = self.fields[agent].step(action)

        # scaled reward based on farm area
        rew_parcel = self._process_rewards(rew_parcel)

        # write results into the mandatory dicts
        # observations are obtained by calling self.observe()
        self.rewards.update({agent: rew_parcel})
        self.terminations.update({agent: ter_parcel})
        self.truncations.update({agent: tru_parcel})
        self.infos.update({agent: info_parcel})

        # advance to next agent in the cycle
        self._accumulate_rewards()  # provided by AECEnv

        # update global budget left
        self.global_budget_left = self._get_global_budget_left()

        self.agent_selection = self._agent_selector.next()

        # AECEnv doesn't return anything for step()

    def observe(self, agent) -> dict:
        obs = self.fields[agent].unwrapped.observe()
        mask = self.fields[agent].unwrapped.action_mask()
        return {
            "agent_id": str(agent),
            "observation": obs,
            **({"shared": self._build_shared()} if self.shared_obs else {}),
            "action_mask": mask,
        }

    def render(self):
        print(self)

    '''
    Callable helper functions and property
    '''

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
        return {a: self.fields[a].unwrapped.crop_code for a in self.possible_agents}

    def get_per_field_crop_price(self):
        return {a: self.fields[a].unwrapped.crop_price for a in self.possible_agents}

    def get_per_field_fertilizer_price(self):
        return {a: self.fields[a].unwrapped.fertilizer_price for a in self.possible_agents}

    def get_initial_no3(self):
        return {a: self.fields[a].unwrapped.infos['NO3'][0] for a in self.possible_agents}

    def get_initial_nh4(self):
        return {a: self.fields[a].unwrapped.infos['NH4'][0] for a in self.possible_agents}

    def _get_global_random_budget(self):
        # get dict of default max budget
        parcel_budgets = {a: self.get_per_parcel_max_budget(a) for a in self.possible_agents}

        # get random reductions by choice for each agent limited by the default budget of the parcel
        # change the logic of random allocation here if needed!
        choices = {
            a: self.rng.choice([*np.arange(0., min(200., parcel_budgets[a]), 5.)])
            for a in self.possible_agents
        }

        # set random budget reduction for each parcel
        for (_agent, choice), (_, budget) in zip(choices.items(), parcel_budgets.items()):
            self.set_per_parcel_budget(_agent, budget-choice)

        self.set_global_budget(self._get_global_budget())


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
        self.fields = {}
        # create each gymnasium cropgym env
        for n in self.agents:
            env = gym.make(n, seed=self.seed, training=self.training)  # set same seed for each parcel. Change?
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
                "observation": env.observation_space,
                **({"shared": self.shared_space} if self.shared_obs else {}),
                "action_mask": gym.spaces.MultiBinary(env.unwrapped.action_space.n),
            }) for ag, env in self.fields.items()
        }

        # action space from individual parcels
        self.action_spaces = {agent: env.action_space
                              for agent, env in self.fields.items()}

    def _process_rewards(self, reward):
        """
        Uses the "weighted by area" policy reward.
        """
        field_size = self.get_field_size(self.agent_selection)
        weighted_reward = (reward * field_size) / self.total_area

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


class SkippingSelector:
    '''
    Here we don't use the agent_selector from PettingZoo.
    This loads agents just sequenti
    '''
    def __init__(self, order: list[str]):
        self.order = order[:]                 # fixed global order
        self.alive = {a: True for a in self.order} # or a set(order)
        self.idx = -1                         # points to last returned

    def kill(self, agent: str):
        self.alive[agent] = False

    def reset(self):
        self.alive = {a: True for a in self.order} # or a set(order)

    def next(self) -> str:
        n = len(self.order)
        if not any(self.alive.values()):
            raise StopIteration("All agents are dead.")
        for _ in range(n):
            self.idx = (self.idx + 1) % n
            cand = self.order[self.idx]
            if self.alive[cand]:
                return cand
        # shouldn't get here
        raise RuntimeError("Alive list inconsistent with order.")

    @property
    def selected_agent(self):
        return self.order[self.idx] if self.idx >= 0 else None


