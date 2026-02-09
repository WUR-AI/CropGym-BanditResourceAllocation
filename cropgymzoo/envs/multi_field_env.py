import os
import yaml
import datetime

from collections import Counter

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
from cropgymzoo.utils.defaults import (
    get_default_years,
    get_wofost_default_crop_features,
    get_default_weather_features,
    get_default_action_features
)
from cropgymzoo.utils.curriculum import make_default_stage_manager
from cropgymzoo.utils.scenario_utils import choose_soil_type
from cropgymzoo.utils.agent_helpers import last_before_nan


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
        "winterwheat_max": {
            5: {"clay": 150, "sand": 100, "silt": 100, "peat": 100},
            190: {"clay": 90, "sand": 60, "silt": 90, "peat": 60},
        },
        "winterwheat_low": {
            5: {"clay": 150, "sand": 100, "silt": 100, "peat": 100},
            190: {"clay": 40, "sand": 10, "silt": 40, "peat": 10},
        },
        "potato_max": {
            5: {"clay": 170, "sand": 130, "silt": 130, "peat": 170},
            30: {"clay": 100, "sand": 130, "silt": 70, "peat": 100},
        },
        "potato_low": {
            5: {"clay": 170, "sand": 130, "silt": 130, "peat": 170},
            30: {"clay": 50, "sand": 70, "silt": 20, "peat": 50},
        },
        "sugarbeet_max": {
            5: {"clay": 100, "sand": 100, "silt": 110, "peat": 100},
            20: {"clay": 50, "sand": 40, "silt": 0, "peat": 40},
        },
        "sugarbeet_low": {
            5: {"clay": 100, "sand": 90, "silt": 60, "peat": 90},
            20: {"clay": 0, "sand": 0, "silt": 0, "peat": 0},
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
            warm_up: int = 0,   # deprecated
            reward: str = 'PNR',
            use_rl_warm_up_actions: bool = True,
            years: list = get_default_years(),
            training: bool = False,
            random_budget: bool = False,
            dict_obs: bool = True,
            shared_obs: bool = False,
            render: bool = False,
            stage: int = 0,
            farm_dict: dict | str = None,
            domain_repeat = 10,
            special_action_space: bool = False,
            concise_obs: bool = False,
    ):
        EzPickle.__init__(
            self,
            seed=seed,
            warm_up=warm_up,
            reward=reward,
            use_rl_warm_up_actions=use_rl_warm_up_actions,
            years=years,
            training=training,
            random_budget=random_budget,
            dict_obs=dict_obs,
            shared_obs=shared_obs,
            render=render,
            stage=stage,
            farm_dict=farm_dict,
            domain_repeat=domain_repeat,
            special_action_space=special_action_space,
            concise_obs=concise_obs,
        )
        super().__init__()
        self.render_mode = None if not render else 'human'
        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)
        self.years = years
        self.training = training
        self.shared_obs = shared_obs
        self.dict_obs = dict_obs
        self.year = self.years[0]
        self.year_cache = self.year
        self.domain_repeat = domain_repeat
        self._domain_repeat_left = 0
        self.reward_code = reward
        self.special_action_space = special_action_space
        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)
        self.concise_obs = concise_obs

        self.has_reset = False

        if farm_dict is None:
            with open(_FIELDS_CONFIG) as f:
                dict_fields = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            dict_fields = farm_dict

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
        if isinstance(farm_dict, str):
            with open(farm_dict) as f:
                farm_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self._init_fields(farm_dict=farm_dict)
        self._init_spaces()
        self._init_farm_variables()
        self._init_infos()

        self.random_budget = random_budget

        self.stage = stage

        self.global_budget = self._get_global_max_budget() if not self.random_budget else self._get_global_random_budget()
        self.global_budget_left = self.global_budget
        self.global_allocated_budget  = self.global_budget

        self.total_area = np.sum([self.get_field_size(agent) for agent in self.possible_agents])

        self._print_season_year = None


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
        self._init_farm_variables()

        # get the options before reset
        # If training, ignore all options and override with only year.
        # Good idea? Check for resource allocation too.
        options = options or {'year': 2010}
        if self.training:
            if self._domain_repeat_left == 0:
                self.year_cache = self.rng.choice(self.years)
                self._domain_repeat_left = self.domain_repeat
            self._domain_repeat_left -= 1
            self._domain_repeat_left = max(self._domain_repeat_left, 0)
            options = {'year': self.year_cache}
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
            if self.training:
                env.unwrapped.domain_repeat_left = self._domain_repeat_left
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

        self._clear_rewards()

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

    def allocate_bandit_budgets(self, allocations: list | np.ndarray):
        """
        It is assumed that the allocation is in the factor of single digits, since it will be multiplied by
        10 in this method.
        """
        assert len(allocations) == len(self.fields)

        for agent, reduction in zip(self.possible_agents, allocations):
            # multiply here
            reduction = reduction * 10
            agent_max = self.get_per_parcel_max_budget(agent)
            allocation = agent_max - reduction
            self.set_per_parcel_budget(agent, allocation)

        self.global_allocated_budget = self._get_global_budget_left()

        print(f'Allocated budget reductions of {allocations}')

    def reconfigure_farm(self, farm_dict: dict, *, year: int | None = None):
        """
        Reconfigure this MultiFieldEnv to represent a new farm.

        This reuses ParcelEnv instances when possible (via ParcelEnv.reconfigure)
        and closes/drops envs for fields that no longer exist.

        Parameters
        ----------
        farm_dict:
            Dict of field configs (same structure as in the YAML files).
        year:
            Optional year override for the underlying ParcelEnv instances.
        """

        # If YAML path is passed accidentally, load it
        if isinstance(farm_dict, str):
            import yaml
            with open(farm_dict) as f:
                farm_dict = yaml.load(f, Loader=yaml.SafeLoader)

        # 1) Drop fields that are no longer present
        new_keys = set(farm_dict.keys())
        old_keys = set(getattr(self, "fields", {}).keys()) if getattr(self, "fields", None) else set()
        to_remove = list(old_keys - new_keys)

        for k in to_remove:
            try:
                self.fields[k].close()
            except Exception:
                pass
            try:
                del self.fields[k]
            except Exception:
                pass

        # 2) Reconfigure/create ParcelEnv objects for new farm
        # (this calls ParcelEnv.reconfigure() when keys already exist)
        self.set_new_fields(farm_dict=farm_dict, year=year)

        # 3) Rebuild agent bookkeeping based on the NEW field set
        self.n_agents = len(farm_dict)
        self.agents = list(farm_dict.keys())
        self.possible_agents = self.agents[:]

        # selector must be rebuilt because agent list changed
        self._agent_selector = AgentSelector(self.possible_agents)
        self.dead_step = {ag: False for ag in self.possible_agents}
        self.agent_to_keep = None

        self.current_step = {agent: 0 for agent in self.possible_agents}
        self.current_obs = {agent: {} for agent in self.possible_agents}

        # 4) Rebuild spaces (obs/action may depend on budgets/action-space)
        self._init_spaces()

        # 5) Reset episode-level tracking
        self._init_infos()
        self._init_farm_variables()

        # 6) Recompute global budgets for the new farm
        self.global_budget = (
            self._get_global_max_budget()
            if not self.random_budget
            else self._get_global_random_budget()
        )
        self.global_budget_left = self.global_budget
        self.global_allocated_budget = self.global_budget

        self.total_area = np.sum([self.get_field_size(agent) for agent in self.possible_agents])

        # Make selector point to a valid first agent
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def set_new_fields(self, farm_dict: dict, year: int = None):
        # Avoid piling up ParcelEnv instances when rotations/scenarios are reloaded.
        # Close and drop old env objects before allocating new ones.
        # if getattr(self, "fields", None):
        #     for _old_env in list(self.fields.values()):
        #         _old_env.close()
        #     self.fields = {}

        for key, field in farm_dict.items():
            soil_type = choose_soil_type(crop=field['crop'], location=(field['soil_lat'], field['soil_lon']))
            if key in self.fields:
                self.fields[key].unwrapped.reconfigure(
                    crop=field["crop"],
                    year=self.year if year is None else year,
                    location=(field["soil_lat"], field["soil_lon"]),
                    area=field["area"],
                    soil_type=field["type"],
                )
            else:
                self.fields[key] = ParcelEnv(
                    crop_features=get_wofost_default_crop_features(),
                    weather_features=get_default_weather_features(),
                    action_features=get_default_action_features(),
                    location=(field['soil_lat'], field['soil_lon']),
                    crop=field['crop'],
                    year=self.year if year is None else year,
                    name=key,
                    area=field['area'],
                    reward=self.reward_code,
                    original=True,
                    training=False,
                    flatten_obs=True,
                    type=field["type"],
                )
        print("Scenario fields initialized!")

    def advance_fields_to_allocation_dates(
            self,
            *,
            days_before_sowing: int = 60,
            preseason_N: float = 0.0,
            apply_preseason_N: bool = False,
            season_year: int | None = None,
            farm_dict_by_year: dict | None = None,
    ) -> dict:
        """Advance each field's internal PCSE model to its own allocation date.

        alloc_date = crop_start_date - days_before_sowing

        During multi-season evaluation, this also:
        - resets action counters and seasonal budget for the *requested* season_year
        - updates the per-field crop label used in reward/price logic

        Returns: dict(agent -> alloc_date)
        """

        # Determine crop per agent for the requested season (if provided)
        crop_by_agent = {}
        if season_year is not None:
            if isinstance(farm_dict_by_year, dict) and int(season_year) in farm_dict_by_year:
                for agent in self.possible_agents:
                    crop_by_agent[agent] = farm_dict_by_year[int(season_year)][agent]["crop"]

        alloc_dates = {}
        for agent in self.possible_agents:
            env = self.fields[agent].unwrapped

            crop_name = crop_by_agent[agent]
            # choose a max budget for this crop; fallback to current max_budget_n/global max
            max_budget = env.CROP_SOIL_MAX[crop_name][env.soil_type]

            # setting a bunch of stuff
            env.year=int(season_year)
            env.crop=crop_name
            env.reward_container.reset()
            env.rewards_obj.reset()
            env._reset_action_variables()
            env.max_budget_n = float(max_budget)
            env.budget_n = float(max_budget)
            env.budget_left = float(max_budget)
            env._reset_prices()


            # Now compute allocation date based on the (possibly updated) agmt crop_start_date
            sow_date = env.agmt.crop_start_date
            alloc_date = sow_date - datetime.timedelta(days=int(days_before_sowing))
            alloc_dates[agent] = alloc_date

            self._advance_field_to_date(
                agent,
                alloc_date,
                preseason_N=preseason_N,
                apply_preseason_N=apply_preseason_N,
            )

        return alloc_dates

    def _advance_field_to_date(
            self,
            agent: str,
            target_date: datetime.date,
            *,
            preseason_N: float = 0.0,
            apply_preseason_N: bool = False,
    ) -> None:
        """Advance a single field's PCSE model to `target_date` using Engine.run()."""
        env = self.fields[agent].unwrapped

        # Determine current simulation date
        try:
            curr_date = env.date  # PCSEEnv property
        except Exception:
            curr_date = getattr(getattr(env, "_model", None), "day", None)

        if curr_date is None:
            raise RuntimeError(f"Cannot determine current PCSE date for agent {agent}")

        days_to_run = int((target_date - curr_date).days)
        if days_to_run <= 0:
            return

        model = getattr(env, "model", None)
        if model is None:
            model = getattr(env, "_model", None)
        if model is None:
            raise RuntimeError(f"Cannot access PCSE model for agent {agent}")

        # Optional: apply preseason N exactly once on target_date
        if apply_preseason_N and float(preseason_N) > 0.0 and days_to_run >= 1:
            if days_to_run > 1:
                model.run(days=days_to_run - 1, action=0)
            model.run(days=1, action=float(preseason_N))
        else:
            model.run(days=days_to_run, action=0)

    # ------------------------------------------------------------------
    # Daisy-chained multi-season evaluation helpers (RL-only or RoT-only)
    # ------------------------------------------------------------------

    def _all_fields_past_season_year(self, season_year: int) -> bool:
        """True iff every field's latest SeasonYear is strictly > season_year."""
        for ag in self.possible_agents:
            infos = getattr(self.fields[ag].unwrapped, "infos", {})
            sy_seq = infos.get("SeasonYear", None)
            if not sy_seq:
                return False
            sy = sy_seq[-1]
            if sy is None:
                return False
            try:
                if int(sy) <= int(season_year):
                    return False
            except Exception:
                return False
        return True

    def collect_agent_infos_for_season(self, season_year: int) -> dict:
        """Return {agent -> {info_key -> list(values_for_that_season)}}."""
        out = {}
        for ag in self.possible_agents:
            env = self.fields[ag].unwrapped
            infos = getattr(env, "infos", {})
            sy_seq = infos.get("SeasonYear", [])
            idx = [i for i, yy in enumerate(sy_seq) if yy == season_year]
            sub = {}
            for k, seq in infos.items():
                try:
                    sub[k] = [seq[i] for i in idx]
                except KeyError:
                    print(f"Warning: {k} not found in agent {ag}'s infos")
            out[ag] = sub
        return out

    def run_until_past_season_year(
        self,
        *,
        season_year: int,
        env_agent,
        next_states: dict | None = None,
    ) -> dict | None:
        """Step the AEC env until all fields are past `season_year`.

        Returns next_states if `env_agent` is MultiRLAgent-like, else None.
        """
        is_multirl = env_agent.__class__.__name__ == "MultiRLAgent"

        if is_multirl and next_states is None:
            next_states = {ag: None for ag in self.possible_agents}

        max_iters = int(10_000_000)
        it = 0

        for agent in self.agent_iter():
            it += 1
            if it > max_iters:
                break

            obs, rew, term, trunc, info = self.last()

            if self.terminations.get(agent, False) or self.truncations.get(agent, False):
                self.step(None)
            else:
                if is_multirl:
                    from tianshou.data import Batch
                    import torch

                    processed_info = Batch({k: [v[-1]] for k, v in info.items()})
                    processed_info["env_id"] = [0]

                    with torch.no_grad():
                        out = env_agent.get_action(
                            agent,
                            obs=obs,
                            next_states=next_states,
                            info=processed_info,
                        )
                    action = out.act.item()
                    state = None if not hasattr(out, "state") else out.state
                    next_states[agent] = state
                else:
                    action = env_agent.get_action(agent, env=self)

                self.step(action)

            if self._all_fields_past_season_year(int(season_year)):
                break

            if not getattr(self, "agents", None):
                break

        return next_states if is_multirl else None

    def set_curriculum_stage(self, stage: int):
        for agent in self.possible_agents:
            self.fields[agent].unwrapped.random_manager.set_stage(stage)
        self.stage = stage
        self.random_budget = True if self.fields[self.possible_agents[-1]].unwrapped.random_manager.budget > 0 else False

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

    def get_per_parcel_dvs(self, _agent):
        return self.fields[_agent].unwrapped.infos['DVS']

    def set_per_parcel_budget(self, _agent, budget):
        self.fields[_agent].unwrapped.set_budget(budget)

    def set_global_budget(self, budget: float):
        self.global_budget = budget

    def get_field_size(self, _agent):
        return self.fields[_agent].unwrapped.area

    def get_farm_area_sum(self):
        return np.sum([self.fields[a].unwrapped.area for a in self.possible_agents])

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

    def get_per_field_area(self):
        return {a: self.fields[a].unwrapped.area for a in self.possible_agents}

    def get_initial_no3(self):
        return {a: self.fields[a].unwrapped.infos['NO3'][0] for a in self.possible_agents}

    def get_initial_nh4(self):
        return {a: self.fields[a].unwrapped.infos['NH4'][0] for a in self.possible_agents}

    def get_initial_n(self, use_navail = False):
        if use_navail:
            return {a: self.fields[a].unwrapped.infos['NAVAIL'][0] for a in self.possible_agents}
        else:
            return {a: self.fields[a].unwrapped.infos['NO3'][0] + self.fields[a].unwrapped.infos['NH4'][0] for a in self.possible_agents}

    def get_initial_wc(self):
        return {a: self.fields[a].unwrapped.infos['WC'][0] for a in self.possible_agents}

    def get_cumulative_reward(self):
        return np.sum([np.cumsum(self.fields[a].unwrapped.infos['Reward'])[-1] for a in self.possible_agents])

    def get_agent_non_zero_action_count(self, agent):
        return self.fields[agent].unwrapped.infos['NonZeroActionCount'][-1]

    def get_dap(self, agent):
        return self.fields[agent].unwrapped._calculate_dap()

    def override_action_space(self):
        self.special_action_space = True

        for agent in self.possible_agents:
            self.fields[agent].unwrapped.make_special_action_space()

        self.action_spaces = {agent: env.action_space
                              for agent, env in self.fields.items()}

    def _get_global_random_budget(self):
        level = self.get_field_env_with_idx(0).random_manager.budget

        # get dict of default max budget
        parcel_budgets = {a: self.get_per_parcel_max_budget(a) for a in self.possible_agents}

        # Maximum reduction allowed by the curriculum level (kg/ha)
        # e.g. level 1 -> 20, level 2 -> 40, ... capped at 160
        max_level_reduction = min(float(level) * 20.0, 160.0)

        for agent, max_budget in parcel_budgets.items():
            # If this parcel's max budget is lower than 160, don't reduce it at all
            # if max_budget < 160.0 or max_level_reduction <= 0.0:
            #     reduction_choices = [0.0]
            # else:
            # Max reduction for this parcel is constrained by both level and its own max budget
            effective_max_reduction = min(max_level_reduction, max_budget)

            # Build {0, 20, 40, ..., effective_max_reduction} in 20 kg/ha steps
            n_steps = int(effective_max_reduction // 20.0)
            reduction_choices = [20.0 * i for i in range(n_steps + 1)]

            # Sample a reduction and set the new budget
            reduction = float(self.rng.choice(reduction_choices))
            new_budget = max_budget - reduction
            self.set_per_parcel_budget(agent, new_budget)

        new_global_budget = self._get_global_budget()

        self.global_allocated_budget = new_global_budget

        self.set_global_budget(new_global_budget)

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

        # --- DVS-based one-time triggers (per episode) ---
        # previous DVS value (to detect threshold crossing)
        self._dvs_prev = {ag: 0.0 for ag in self.agents}
        # which DVS thresholds have already been triggered for this agent
        self._dvs_triggered = {ag: set() for ag in self.agents}


    def _init_fields(self, seed: int = None, farm_dict: dict = None):
        """
        This is where we initialize the sub-environments where each agent will work.
        :return: a dict called "fields", filled with different CropGym envs
        """
        self.fields: dict[str, ParcelEnv] = {}
        if farm_dict is None:
            # create each gymnasium cropgym env
            for n in self.agents:
                env = gym.make(
                    n,
                    seed=seed or self.seed,  # set same seed for each parcel. Change?
                    training=self.training,
                    random_manager=make_default_stage_manager(),
                    domain_repeat=self.domain_repeat,
                    reward=self.reward_code,
                    special_action_space=self.special_action_space,
                    concise_obs=self.concise_obs,
                )
                self.fields[n] : ParcelEnv = env
            print(f"Fields initialized with seed no. {self.seed}!")
        else:
            for key, field in farm_dict.items():
                soil_type = choose_soil_type(crop=field['crop'], location=(field['soil_lat'], field['soil_lon']))
                self.fields[key] = ParcelEnv(
                    crop_features=get_wofost_default_crop_features(),
                    weather_features=get_default_weather_features(),
                    action_features=get_default_action_features(),
                    location=(field['soil_lat'], field['soil_lon']),
                    crop=field['crop'],
                    year=2000,
                    name=key,
                    area=field['area'],
                    reward=self.reward_code,
                    original=True,
                    training=False,
                    flatten_obs=True,
                    type=field["type"],
                    special_action_space=self.special_action_space,
                    concise_obs=self.concise_obs,
                )
            print("Scenario fields initialized!")

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
        # field_size = self.get_field_size(self.agent_selection)

        # This is for the PNY reward
        # weighted_reward = (reward * field_size) / self.total_area

        # This is for the PNB reward
        weighted_reward = reward

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


    def get_handbook_dict(self, agent):
        n_init = self.fields[agent].unwrapped.infos['NAVAIL'][0]
        sb = 200 - (1.7 * min(n_init, 60))
        pt_cl = 285 - (1.1 * min(n_init, 60))
        pt_s = 300 - (1.8 * min(n_init, 30))
        ww_1 = min(100, 140 - n_init)
        ww_2 = 80
        ww_3 = 80
        handbook = {
            "sugarbeet": {
                10: {"clay": sb, "sand": sb, "silt": sb, "peat": sb},
                30: {"clay": 30, "sand": 30, "silt": 30, "peat": 30},
            },
            "potato": {
                10: {"clay": min(200, pt_cl), "sand": min(170, pt_s), "silt": min(200, pt_cl), "peat": min(170, pt_s)},
                30: {"clay": pt_cl, "sand": pt_s, "silt": pt_cl, "peat": pt_s},
            },
            "winterwheat": {
                10: {"clay": ww_1, "sand": ww_1, "silt": ww_1, "peat": ww_1},
                120: {"clay": ww_2, "sand": ww_2, "silt": ww_2, "peat": ww_2},
                140: {"clay": ww_3, "sand": ww_3, "silt": ww_3, "peat": ww_3},
            }
        }
        return handbook

    def get_handbook_dict_dvs(self, agent):
        n_init = self.fields[agent].unwrapped.infos['NAVAIL'][0]
        sb = 200 - (1.7 * min(n_init, 60))
        pt_cl = 285 - (1.1 * min(n_init, 60))
        pt_s = 300 - (1.8 * min(n_init, 30))
        ww_1 = min(100, 140 - n_init)
        ww_2 = 80
        ww_3 = 80
        handbook = {
            "sugarbeet": {
                -0.1: {"clay": sb, "sand": sb, "silt": sb, "peat": sb},
                0.1: {"clay": 30, "sand": 30, "silt": 30, "peat": 30},
            },
            "potato": {
                -0.1: {"clay": min(200, pt_cl), "sand": min(170, pt_s), "silt": min(200, pt_cl), "peat": min(170, pt_s)},
                0.0: {"clay": pt_cl, "sand": pt_s, "silt": pt_cl, "peat": pt_s},
            },
            "winterwheat": {
                -0.1: {"clay": ww_1, "sand": ww_1, "silt": ww_1, "peat": ww_1},
                0.3: {"clay": ww_2, "sand": ww_2, "silt": ww_2, "peat": ww_2},
                0.5: {"clay": ww_3, "sand": ww_3, "silt": ww_3, "peat": ww_3},
            }
        }
        return handbook

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

    def random_fertilization(self, agent_name):
        """Random fertilization schedule based on crop + soil."""
        budget_left  = self.get_per_parcel_budget_left(agent_name)

        fert = 0.0
        if self.get_agent_non_zero_action_count(agent_name) <= 4:
            fert = self.rng.choice([0, min(80, max(budget_left, 0))], p=[0.95, 0.05])

        return fert / 10

    def rule_of_thumb(self, agent_name, use_dvs=True):
        """Simple farmer rule-based fertilization schedule based on crop + soil.    Robust to multi-season evaluation:
        - DVS can be None during fallow / between campaigns.
        - Crop/season can change for the same field agent across daisy-chained campaigns.
        """
        crop = self.get_per_field_crop_name()[agent_name]
        soil = self.get_per_field_soil_type()[agent_name]
        budget_left = self.get_per_parcel_budget_left(agent_name)

        # --- lazy init of tracking dicts (in case older checkpoints/envs don't have them) ---
        if not hasattr(self, "_dvs_prev"):
            self._dvs_prev = {}
        if not hasattr(self, "_dvs_triggered"):
            self._dvs_triggered = {}
        if not hasattr(self, "_dvs_crop_prev"):
            self._dvs_crop_prev = {}
        if not hasattr(self, "_dvs_season_prev"):
            self._dvs_season_prev = {}

        self._dvs_prev.setdefault(agent_name, None)
        self._dvs_triggered.setdefault(agent_name, set())

        # Detect current season label year if available (daisy-chaining), else None
        try:
            season_year = self.fields[agent_name].unwrapped.get_latest_info("SeasonYear")
        except Exception:
            season_year = None

        # Reset triggers when crop or season changes
        if self._dvs_crop_prev.get(agent_name, None) != crop or self._dvs_season_prev.get(agent_name,
                                                                                          None) != season_year:
            self._dvs_triggered[agent_name] = set()
            self._dvs_prev[agent_name] = None
            self._dvs_crop_prev[agent_name] = crop
            self._dvs_season_prev[agent_name] = season_year

        # Get current DVS safely (can be None between campaigns)
        try:
            dvs = self.fields[agent_name].model.get_output()[-1].get("DVS", None)
        except Exception:
            dvs = None

        # If crop is inactive/fallow, do not fertilize and reset DVS tracking
        if dvs is None:
            self._dvs_prev[agent_name] = None
            self._dvs_triggered[agent_name] = set()
            return 0.0

        # Ensure float for comparisons
        try:
            dvs = float(dvs)
        except Exception:
            self._dvs_prev[agent_name] = None
            self._dvs_triggered[agent_name] = set()
            return 0.0

        dap_plant = self.get_dap(agent_name)
        fert = 0.0

        if use_dvs:
            prev_dvs = self._dvs_prev.get(agent_name, None)
            handbook = self.get_handbook_dict_dvs(agent_name).get(crop, {})  # dict(threshold -> soil_map)

            for thr, soil_map in handbook.items():
                # Trigger once per threshold per season; allow "crossing" logic when prev_dvs exists
                crossed = (dvs >= float(thr)) and (prev_dvs is None or prev_dvs < float(thr))
                if crossed and (thr not in self._dvs_triggered[agent_name]):
                    fert = float(soil_map.get(soil, 0.0))
                    self._dvs_triggered[agent_name].add(thr)
                    break
        else:
            for day, soil_map in self.get_handbook_dict(agent_name).get(crop, {}).items():
                if self._is_at_date(dap_plant, day):
                    fert = soil_map.get(soil, 0.0)
                    break

        self._dvs_prev[agent_name] = dvs

        allowed_fert = max(min(max(0, budget_left), fert), 0)
        return allowed_fert / 10


    @staticmethod
    def _get_shared_obs_keys():
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

    def set_print_season_year(self, season_year: int | None) -> None:
        """Force __str__ to print a specific season year (useful for multi-season eval)."""
        self._print_season_year = None if season_year is None else int(season_year)

    def __str__(self) -> str:
        """
        Return a multiline string such as

            Farm status – budget left: 350 / 400 kg N
            Field          Crop        Season  Date       N applied   Yield[t/ha]         NUE   Nsurp     Profit     Reward
            ---------------------------------------------------------------------------------------------------------------
            parcel_001     winterwheat   2021  03/15/2021       80          3.5         0.80    12.0     1000.0      0.95
            parcel_002     potato        2021  03/17/2021       30            –            –       –          –         –
            parcel_003     sugarbeet     2021  03/20/2021        –            –            –       –          –         –

            Crop distribution → winterwheat:1 | potato:1 | sugarbeet:1
        """

        from collections import Counter

        def safe(env, key, default="–"):
            try:
                val = env.unwrapped.get_latest_info(key)
                if val is None:
                    return default
                return val
            except Exception:
                return default

        def safe_season(env, key, season, default="–"):
            try:
                val = env.unwrapped.get_latest_season_info(key, season)
                if val is None:
                    return default
                return val
            except Exception:
                return default

        def safe_grab(env, key, season, default="-"):
            if season is None:
                return safe(env, key, default)
            return safe_season(env, key, season, default)

        def latest_season_year(env):
            try:
                sy = env.unwrapped.get_latest_info("SeasonYear")
                return int(sy) if sy is not None else None
            except Exception:
                return None

        def _display_season_year() -> int | None:
            # If an external caller set a display season, prefer it.
            _sy = self._print_season_year
            return int(_sy) if _sy is not None else None

        def season_sum(env, key, season_year):
            """Sum a time-series info key restricted to the given season_year, if possible."""
            try:
                infos = env.unwrapped.infos

                sy_list = infos.get("SeasonYear", [])
                seq = infos.get(key, [])
                crop_active = infos.get("CropActive", None)  # may be missing

                if not seq:
                    return 0.0

                # If no season requested, optionally sum only during CropActive
                if season_year is None:
                    n = len(seq)
                    if crop_active is not None:
                        n = min(n, len(crop_active))
                        vals = [seq[i] for i in range(n) if bool(crop_active[i])]
                    else:
                        vals = list(seq)
                    return float(np.nansum(vals)) if vals else 0.0

                if not sy_list:
                    return 0.0

                # Align lengths defensively
                n = min(len(sy_list), len(seq))
                if crop_active is not None:
                    n = min(n, len(crop_active))

                target = int(season_year)

                # Filter: SeasonYear == target AND CropActive == True (if available)
                if crop_active is not None:
                    vals = [seq[i] for i in range(n) if sy_list[i] == target and bool(crop_active[i])]
                else:
                    vals = [seq[i] for i in range(n) if sy_list[i] == target]

                return float(np.nansum(vals)) if vals else 0.0
            except Exception:
                return 0.0

        # use a flexible formatter
        def format_val(val, width, prec=2):
            if isinstance(val, (int, float, np.floating)):
                # avoid printing nan as 'nan' with fixed width
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    return f"{'–':>{width}}"
                return f"{val:>{width}.{prec}f}"
            return f"{str(val):>{width}}"

        # Determine which season we are currently in (for daisy-chained eval)
        current_season_year = _display_season_year()
        if current_season_year is None:
            season_candidates = []
            for _fid, _env in self.fields.items():
                sy = latest_season_year(_env)
                if sy is not None:
                    season_candidates.append(sy)
            current_season_year = max(season_candidates) if season_candidates else None

        # Header: show both env.year and current_season_year when they differ
        season_str = (
            f"season {current_season_year}" if current_season_year is not None else "season –"
        )
        base_year_str = getattr(self, "year", None)
        # In daisy-chained eval, self.year may remain at the reset year; prefer current season.
        if current_season_year is not None:
            base_year_str = int(current_season_year)
        if base_year_str is None:
            year_str = season_str
        else:
            year_str = f"year {base_year_str} ({season_str})" if (current_season_year is not None and int(base_year_str) != int(current_season_year)) else f"year {base_year_str}"

        # Budget left: in multi-season eval budgets may be per-season; keep current fields' view.
        header = (
            f"Farm status; {year_str} – budget left: "
            f"{round(self.global_budget_left * self.get_farm_area_sum(), 1)} / "
            f"{round(self.global_allocated_budget * self.get_farm_area_sum(), 1)} kg N "
            f"or {self.global_budget_left} / {self.global_allocated_budget} kg N / ha "
            f"(Max. {self._get_global_max_budget()}) | Cum. Reward: {self.get_cumulative_reward():.1f}"
        )

        cols = ("Field (area[ha])", "Crop", "Season", "Date", "N applied", "Yield[t/ha]", "NUE", "Nsurp", "Profit", "Reward")
        fmt_header = "{:20} {:12} {:6} {:10} {:>10} {:>15} {:>7} {:>7} {:>10} {:>10}"
        lines = [header, fmt_header.format(*cols), "-" * 128]

        # build one row per parcel
        crop_counts = Counter()
        for field_id, env in self.fields.items():
            sy = current_season_year if current_season_year is not None else latest_season_year(env)
            # Prefer infos-driven CropName (handles multi-crop daisy-chaining)
            crop = safe_grab(env, "CropName", sy, default=getattr(env.unwrapped, "crop", "–"))
            crop_counts[str(crop)] += 1

            sy_disp = str(sy) if sy is not None else "–"

            val_yield = safe_grab(env, "Yield", sy)
            val_yield = val_yield / 1000 if isinstance(val_yield, (int, float, np.floating)) else val_yield

            val_date = safe_grab(env, "Date", sy)
            if hasattr(val_date, "strftime"):
                val_date = val_date.strftime("%m/%d/%Y")

            # Reward: show per-season cumulative reward when SeasonYear exists, else full cumulative.
            r_cum = season_sum(env, "Reward", sy)

            vals = [
                f"{field_id} ({env.unwrapped.area:.1f})",
                crop,
                sy,
                val_date,
                safe_grab(env, "Naction", sy),
                val_yield,
                safe_grab(env, "Nue", sy),
                safe_grab(env, "Nsurp", sy),
                safe_grab(env, "Profit", sy),
                r_cum,
            ]

            line = (
                f"{vals[0]:20} {str(vals[1]):12} {format_val(vals[2], 6, prec=0)} {str(vals[3]):10} "
                f"{format_val(vals[4], 10)} "
                f"{format_val(vals[5], 15)} "
                f"{format_val(vals[6], 7)} "
                f"{format_val(vals[7], 7)} "
                f"{format_val(vals[8], 10)} "
                f"{format_val(vals[9], 10)}"
            )
            lines.append(line)

        # add a small summary line
        summary = " | ".join(f"{c}:{n}" for c, n in sorted(crop_counts.items()))
        lines.append(f"\nCrop distribution → {summary}")

        return "\n".join(lines)
