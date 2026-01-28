import os
import argparse
from copy import deepcopy
import yaml
import torch
import pickle
import itertools
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cropgymzoo.utils.agent_helpers import _make_base_arms, _make_topk_super_arms

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.defaults import get_default_years
from cropgymzoo.utils.scenario_utils import model_picker
from cropgymzoo.utils.rewards import Rewards
from cropgymzoo.train_policy import load_model, initialize_policy
from cropgymzoo.eval_policy import MultiRLAgent, load_policy, RoTAgent, RandomAgent

from cropgymzoo import _SCENARIO_PATH, _CONFIG_PATH

# ---------------------------------------------------------------------
# Gymnasium env that works for any n_fields
# ---------------------------------------------------------------------
class AllocationBandit(gym.Env):
    """
    A one-Step combinatorial multi-armed bandit environment for resource allocation.
    """
    metadata = {"render_modes": []}

    def __init__(
            self,
            delta_kg: float = 5.0,
            warm_up_eps: int = 10,
            cap: float = 1.0,
            years: list = get_default_years(),
            seed: int = 107,
            action_type: str = 'continuous',
            args: argparse.Namespace = None,
            field_reward: str = 'NSU',
            flat_context: bool = True,
            region: str = None,
            farm_id: int = None,
            render: bool = False,
    ):
        super().__init__()

        self.flat_context = flat_context
        self.warm_up_eps = warm_up_eps
        self.field_reward = field_reward

        assert action_type in ['discrete', 'multi_discrete', 'continuous']
        self.action_type = action_type

        self.rng, self.seed = gym.utils.seeding.np_random(seed=seed)

        self.years = years
        self.year = None
        self.cap = cap
        self.region = region
        self.farm_id = farm_id
        self.render = render
        self.saved_model = None
        self.original_saved_model = None

        self._elite_center_action = None
        self._last_action = None

        # The MARL env
        self._init_envs(args, warm_up_eps=warm_up_eps)

        # set up per parcel budgets
        self._init_meta_info()

        # init spaces
        self.bins = float(delta_kg)
        self._init_spaces()

        self._construct_info()

    '''
    Gymnasium functions
    '''
    def reset(self, *, seed=None, options=None):

        options = self._get_default_reset_options() if options is None else options

        assert 'year' in options, "If testing, make sure to pass 'year' in the options dictionary!"

        self.year = options.get('year')

        options['seed'] = seed

        self.farm.reset(seed=seed, options=options)

        return self._get_context(), self._construct_info(options)

    def step(self, action):

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        try:
            self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            self._last_action = None

        action = np.asarray(action, dtype=np.float32)

        # check if action is valid
        assert self.action_space.contains(action), f"{action} is an invalid action"

        # save action this episode
        self.infos['AllocationAction'] = action

        # allocate here
        self.farm.allocate_bandit_budgets(self.infos['AllocationAction'])

        # runs one episode of the MARL agent
        infos_agents = self.env_agent.run([self.year], year_key=False)

        self.infos['AgentInfos'] = infos_agents
        reward = self._get_reward()

        return np.zeros(1, dtype=np.float32), reward, True, False, self.infos


    def _get_reward(self):
        separate_nsurp = [self.infos['AgentInfos'][agent]['Nsurp'][-1] for agent in self.parcel_meta_infos.keys()]
        separate_reward = [Rewards.ContainerNUE.nsurplus_score(n, low=15.0, high=40.0, max_dev=40) for n in separate_nsurp]

        n_arms = len(separate_reward)
        weighted_reward = [n / n_arms for n in separate_reward]

        reward = np.sum(weighted_reward)
        return reward

    def unflatten_context_per_field(self, theta_flat: np.ndarray) -> np.ndarray:
        """
        Convert flat theta (keys-major) into per-field theta matrix.

        Assumes each context key contributes exactly 1 scalar per field and that
        `_flatten_context` concatenates keys in `_get_context_keys()` order.

        Returns: (n_fields, n_keys)
        """
        keys = self._get_context_keys()
        n_fields = int(self.n_fields)
        theta_flat = np.asarray(theta_flat, dtype=np.float32).reshape(-1)

        n_keys = len(keys)
        expected = n_fields * n_keys
        if theta_flat.shape[0] != expected:
            raise ValueError(
                f"Cannot unflatten theta: expected length {expected} (= n_fields {n_fields} * n_keys {n_keys}), "
                f"got {theta_flat.shape[0]}. This likely means one or more context keys are vector-valued per field; "
                "in that case we need a context layout map."
            )

        # theta is [key1(field1..fieldN), key2(field1..fieldN), ...]
        blocks = [theta_flat[i * n_fields:(i + 1) * n_fields] for i in range(n_keys)]
        return np.stack(blocks, axis=1)  # (n_fields, n_keys)

    @staticmethod
    def compute_per_field_rewards_from_nsurp(n_surps: dict) -> np.ndarray:
        """Return per-field reward components consistent with `_get_reward`."""
        r = [Rewards.ContainerNUE.nsurplus_score(n, low=15.0, high=40.0, max_dev=40) for n in n_surps]
        return np.asarray(r, dtype=np.float32)


    @staticmethod
    def _get_context_keys():
        return [
            "InitialN",
            "CropPrice",
            "CropCode",
            "FertilizerPrice",
            "Area",
            "MaxBudget",
            "EarlySeasonPrecipitation",
            "EarlySeasonTemperatureMin",
            # "EarlySeasonTemperatureMax",
            "EarlySeasonIrradiation",
            # "HistoricalCropPrices",
            # "HistoricalFertilizerPrices",
            # "HistoricalProfit",
            # "HistoricalYield",
            # "HistoricalFertilizerUse",
            # "HistoricalBudget",
            # "HistoricalBudgetLeft",
            # "HistoricalNUE",
            # "HistoricalNsurplus",
            "HistoricalPrecipitation",
            "HistoricalTemperatureMin",
            # "HistoricalTemperature",
            # "HistoricalTemperatureMax",
            "HistoricalIrradiation",
        ]

    def _context_value(self, key: str):
        """Compute a single context feature."""
        if key == "InitialN":
            return [self.farm.get_initial_n()[a] for a in self.agents_order]

        if key == "CropPrice":
            return [self.farm.get_per_field_crop_price()[a] for a in self.agents_order]

        if key == "CropCode":
            return [self.farm.get_per_field_crop_code()[a] for a in self.agents_order]

        if key == "FertilizerPrice":
            return [self.farm.get_per_field_fertilizer_price()[a] for a in self.agents_order]

        if key == "Area":
            return [self.farm.get_per_field_area()[a] for a in self.agents_order]

        if key == "MaxBudget":
            return self._get_max_budgets()

        # --- early season features ---
        if key == "EarlySeasonPrecipitation":
            return self._get_early_season_weather_features("RAIN")

        if key == "EarlySeasonTemperatureMin":
            return self._get_early_season_weather_features("TMIN")

        if key == "EarlySeasonTemperatureMax":
            return self._get_early_season_weather_features("TMAX")

        if key == "EarlySeasonIrradiation":
            return self._get_early_season_weather_features("IRRAD")

        # --- historical end-season features ---
        if key == "HistoricalCropPrices":
            return self._get_historical_end_season_features("CropPrice")

        if key == "HistoricalFertilizerPrices":
            return self._get_historical_end_season_features("FertilizerPrice")

        if key == "HistoricalProfit":
            return self._get_historical_end_season_features("Profit")

        if key == "HistoricalYield":
            return self._get_historical_end_season_features("Yield")

        if key == "HistoricalFertilizerUse":
            return self._get_historical_end_season_features("Naction")

        if key == "HistoricalBudget":
            return self._get_historical_end_season_features("BudgetTotal")

        if key == "HistoricalBudgetLeft":
            return self._get_historical_end_season_features("BudgetLeft")

        if key == "HistoricalNUE":
            return self._get_historical_end_season_features("Nue")

        if key == "HistoricalNsurplus":
            return self._get_historical_end_season_features("Nsurp")

        # --- weather ---
        if key == "HistoricalPrecipitation":
            return self._get_historical_weather_features("RAIN")

        if key == "HistoricalTemperatureMin":
            return self._get_historical_weather_features("TMIN")

        if key == "HistoricalTemperature":
            return self._get_historical_weather_features("TEMP")

        if key == "HistoricalTemperatureMax":
            return self._get_historical_weather_features("TMAX")

        if key == "HistoricalIrradiation":
            return self._get_historical_weather_features("IRRAD")

        raise KeyError(f"Unknown context key: {key}")

    def _get_historical_context_keys(self):
        return self._get_context_keys()[5:]

    def _get_max_budgets(self) -> list:
        return [self.farm.get_per_parcel_max_budget(a) for a in self.agents_order]

    def super_arms_limit(self, limit: float) -> np.ndarray:
        """
        Keep rows where the *remaining* total allocation is <= limit:
            sum_i (M_i - R_i) <= limit
        Equivalently:
            sum_i R_i >= sum(M) - limit

        Also enforces 0 <= R_i <= M_i.
        """
        reductions = np.asarray(self.super_arms, dtype=np.float32)  # (K, N)
        max_budgets = np.asarray(self.max_budgets, dtype=np.float32)  # (N,)

        m_sum = float(max_budgets.sum())
        assert (0.0 <= float(limit) <= m_sum), f"limit must be in [0, {m_sum:.3f}]"

        # sanity check
        # sum each arm
        row_sum_reduction = reductions.sum(axis=1)

        # get the difference between max and given limit
        threshold = m_sum/10 - float(limit/10)

        # get mask of all row sums that are above this difference
        meets_total = row_sum_reduction >= threshold

        return reductions[meets_total]

    '''
    Helper functions
    '''

    def get_rotation_year(self, year):
        assert year in [2020, 2021, 2022, 2023, 2024]

        # Avoid rebuilding all ParcelEnv objects if we are already on this rotation year.
        if getattr(self, "year", None) == year and getattr(self.farm, "year", None) == year and getattr(self.farm, "fields", None):
            return

        with open(os.path.join(_SCENARIO_PATH, f"{self.region}", f"{year}", f"farmer_{self.farm_id}.yaml"), 'r') as f:
            dict_fields = yaml.safe_load(f)

        self.farm.set_new_fields(dict_fields, year=year)

        if self.original_saved_model:
            # after setting new fields, replace the working RL agents
            saved_model = model_picker(self.original_saved_model, dict_fields)

            # load model in runner
            policy_manager, obs_rms = load_policy(self.farm, saved_model)
            self.env_agent.policy_manager = policy_manager
            self.env_agent.obs_rms = obs_rms

    def _construct_info(self, options=None):
        if options is not None:
            self.infos = {
                'options': options,
                'seed': options.get('seed', 0),
                'year': options.get('year', self.rng.choice(self.years)),
            }
        else:
            self.infos = {}
        return self.infos

    def _get_default_reset_options(self):
        return {'year': self.rng.choice(self.years)}

    '''
    Context helper functions
    '''

    def _flatten_context(self, context: dict) -> np.ndarray:
        return np.concatenate(
            [
                np.array(context[k], dtype=float).ravel()
                for k in self._get_context_keys()
            ],
            dtype=np.float32,
        )

    def _get_context(self):
        keys = self._get_context_keys()

        context = {
            key: self._context_value(key)
            for key in keys
        }

        if not self.flat_context:
            return context
        else:
            return self._flatten_context(context)


    def _get_historical_end_season_features(self, feature: str):
        """Return [mean_over_iters( last value of feature for this agent ), for each agent]."""
        out = []
        for agent in self.agents_order:
            vals = []
            for iter_info in self.warm_up_infos:  # iter_info: dict per iteration
                agent_info = iter_info.get(agent)
                seq = agent_info.get(feature)
                vals.append(seq[-1])
            out.append(float(np.mean(vals)))
        return out

    def _get_historical_weather_features(self, feature: str):
        """Return [mean_over_iters( mean of the feature sequence for this agent ), per agent]."""
        out = []
        for agent in self.agents_order:
            vals = []
            for iter_info in self.warm_up_infos:
                agent_info = iter_info.get(agent)
                seq = agent_info.get(feature)
                vals.append(np.mean(seq) / 1e6 if feature == 'IRRAD' else np.mean(seq))
            out.append(float(np.mean(vals)))
        return out

    def _get_early_season_weather_features(self, feature: str):
        """Return [mean_over_iters( mean of the feature sequence for this agent ), per agent]."""
        out = []
        for agent in self.agents_order:
            days = [day['day'] for day in self.farm.fields[agent].model.get_output()]
            vals = []
            for day in days:
                val = getattr(self.farm.fields[agent].wdp(day), feature)
                vals.append(val / 1e6 if feature == 'IRRAD' else val)
            out.append(float(round(np.mean(vals), 3)))
        return out

    def set_elite_center_action(self, action):
        if action is None:
            self._elite_center_action = None
            return
        self._elite_center_action = np.asarray(action, dtype=np.float32).reshape(-1)

    def init_model_sampler(
            self,
            eps: float = 0.10,
            alpha: float = 0.20,
            min_prob: float = 1e-6,
    ):
        """
        Initialize adaptive per-field categorical sampling distributions.

        eps:   exploration mixing with uniform distribution
        alpha: EMA update rate toward elite-induced distribution
        """
        self._model_sampler_eps = float(eps)
        self._model_sampler_alpha = float(alpha)
        self._model_sampler_min_prob = float(min_prob)

        # One categorical distribution per field/agent, aligned with base_arms[agent]
        self._sampler_probs: dict[str, np.ndarray] = {}
        for ag in self.agents_order:
            vals = np.asarray(self.base_arms[ag], dtype=np.float32)
            p = np.ones(len(vals), dtype=np.float32)
            p = p / p.sum()
            self._sampler_probs[ag] = p

    def _ensure_model_sampler(self):
        """Make sure sampler is initialized (safe to call each round)."""
        if not hasattr(self, "_sampler_probs") or self._sampler_probs is None:
            self.init_model_sampler()

    def sample_model_informed_super_arms(
            self,
            n_candidates: int,
            reduced: bool = False,
            rng: np.random.RandomState | None = None,
            eps = None,
    ) -> np.ndarray:
        """
        Sample combinatorial arms using learned per-field categorical probabilities.

        Samples each field independently from a categorical distribution over
        that field's discrete base arms, optionally mixed with uniform exploration.
        """
        self._ensure_model_sampler()

        if rng is None:
            rng = self.rng

        n_fields = self.n_fields
        candidates = np.empty((n_candidates, n_fields), dtype=np.float32)
        agents = self.agents_order

        eps_use = self._model_sampler_eps if eps is None else float(eps)

        for k in range(n_candidates):
            for i, ag in enumerate(agents):
                vals = np.asarray(self.base_arms[ag], dtype=np.float32)
                p = np.asarray(self._sampler_probs[ag], dtype=np.float32)

                # mix with uniform to prevent collapse
                if eps_use > 0.0:
                    u = np.ones_like(p, dtype=np.float32) / float(len(p))
                    p_mix = (1.0 - eps_use) * p + eps_use * u
                else:
                    p_mix = p

                p_mix = np.maximum(p_mix, self._model_sampler_min_prob)
                p_mix = p_mix / p_mix.sum()

                idx = rng.choice(len(vals), p=p_mix)
                candidates[k, i] = float(vals[idx])

        if reduced:
            candidates = self._apply_reduced_constraint(candidates)

        return candidates

    def update_model_sampler_probs(
            self,
            X_candidates: np.ndarray,
            scores: np.ndarray,
            top_k: int = 256,
            alpha: float = None,
    ):
        """
        Update per-field categorical distributions using top-scoring candidates.

        X_candidates: (M, n_fields)
        scores:       (M,)  acquisition scores (e.g., UCB)
        """
        self._ensure_model_sampler()

        Xc = np.asarray(X_candidates, dtype=np.float32)
        s = np.asarray(scores, dtype=np.float32).reshape(-1)

        if Xc.size == 0 or s.size == 0:
            return

        M = int(Xc.shape[0])
        top_k = int(min(max(top_k, 1), M))
        elite_idx = np.argsort(s)[-top_k:]
        elite = Xc[elite_idx]  # (top_k, n_fields)

        alpha_use = self._model_sampler_alpha if alpha is None else float(alpha)
        agents = self.agents_order

        for i, ag in enumerate(agents):
            vals = np.asarray(self.base_arms[ag], dtype=np.float32)
            counts = np.zeros(len(vals), dtype=np.float32)

            # count elite occurrences (snap to closest discrete value)
            col = elite[:, i]
            for v in col:
                j = int(np.argmin(np.abs(vals - float(v))))
                counts[j] += 1.0

            if counts.sum() <= 0:
                continue

            target = counts / counts.sum()

            p_old = np.asarray(self._sampler_probs[ag], dtype=np.float32)
            p_new = (1.0 - alpha_use) * p_old + alpha_use * target

            p_new = np.maximum(p_new, self._model_sampler_min_prob)
            p_new = p_new / p_new.sum()

            self._sampler_probs[ag] = p_new

    def add_stats_to_context(self, info):
        self.warm_up_infos.append(info)

    def sample_super_arms(
            self,
            n_candidates: int,
            reduced: bool = False,
            rng: np.random.RandomState | None = None,
    ) -> np.ndarray:
        """
        Sample `n_candidates` combinatorial arms without enumerating all of them.

        Each candidate is a length-n_fields vector; entry i is sampled from the
        discrete base arms of field i.

        If `reduced=True`, you can enforce your 'reduced' scenario logic here
        (e.g. global budget reduction).
        """
        if rng is None:
            rng = self.rng

        n_fields = self.n_fields
        candidates = np.empty((n_candidates, n_fields), dtype=np.float32)

        # Assume base_arms is a dict: agent_id -> 1D np.array of allowed values
        agents = self.agents_order

        for k in range(n_candidates):
            vec = []
            for a in agents:
                vals = self.base_arms[a]  # discrete values for this field
                val = rng.choice(vals)
                vec.append(val)
            candidates[k] = np.array(vec, dtype=np.float32)

        if reduced:
            # Example: enforce a global-budget-like constraint
            candidates = self._apply_reduced_constraint(candidates)

        return candidates

    def sample_crop_grid_super_arms(
        self,
        n_candidates: int,
        rng: np.random.RandomState | None = None,
        reduced: bool = False,
        n_steps: int = 5,
        include_center: bool = True,
        max_cartesian: int = 200_000,
        unique: bool = True,
    ) -> np.ndarray:
        """Grid/permutation sampler around crop-typical centers.

        For each field i, we build a small discrete set of values around a
        crop-typical center using the env bins.

        Example (bins=0.5):
            center=0.0 -> {0.0, 0.5, 1.0, 1.5, 2.0, 2.5}

        Then we form candidates by drawing from the cartesian product of these
        per-field option sets. If the full cartesian is small, we can enumerate it.
        Otherwise, we sample random permutations efficiently.

        Parameters
        ----------
        n_candidates : int
            How many super-arms to return.
        n_steps : int
            Number of +bin steps to include beyond the center.
        max_cartesian : int
            If total cartesian size <= max_cartesian, enumerate all combos and
            then sample from them. Otherwise, sample randomly from the product.
        unique : bool
            If True, remove duplicates from the returned candidate matrix.
        """
        if rng is None:
            rng = self.rng

        n_fields = self.n_fields
        agents = self.agents_order

        # crop names per field (stable and cheap)
        crop_names = [self.farm.get_per_field_crop_name()[a] for a in agents]

        # Build per-field option sets around the *actual discrete center action*
        options_per_field: list[np.ndarray] = []

        for i, ag in enumerate(agents):
            vals = np.asarray(self.base_arms[ag], dtype=np.float32)

            # --- choose center from an actual action value ---
            # priority: elite -> last_action -> random
            if self._elite_center_action is not None and i < len(self._elite_center_action):
                center = float(self._elite_center_action[i])
            # elif self._last_action is not None and i < len(self._last_action):
            #     center = float(self._last_action[i])
            else:
                center = self._typical_reduction(crop_names[i])

            # center must be a valid discrete value; find its index
            idx_center = int(np.argmin(np.abs(vals - center)))

            # --- neighbors by stepping indices in the discrete base arms ---
            idxs = []
            if include_center:
                idxs.append(idx_center)

            for k in range(1, n_steps + 1):
                idxs.append(idx_center + k)
                idxs.append(idx_center - k)

            idxs = np.asarray(idxs, dtype=np.int32)
            idxs = np.clip(idxs, 0, len(vals) - 1)

            snapped = vals[idxs]

            # Deduplicate and sort
            snapped = np.unique(snapped)
            snapped.sort()

            options_per_field.append(snapped)

        # Determine cartesian size
        sizes = [len(o) for o in options_per_field]
        total_cart = int(np.prod(sizes)) if sizes else 0

        # ---- Candidate generation ----
        if total_cart > 0 and total_cart <= max_cartesian:
            # enumerate all combos (safe because total_cart is small)
            # build an index grid using np.meshgrid, then stack
            grids = np.meshgrid(*[np.arange(s, dtype=np.int32) for s in sizes], indexing="ij")
            idx_mat = np.stack([g.reshape(-1) for g in grids], axis=1)  # (total_cart, n_fields)

            # map indices -> values
            all_cands = np.empty((idx_mat.shape[0], n_fields), dtype=np.float32)
            for i in range(n_fields):
                all_cands[:, i] = options_per_field[i][idx_mat[:, i]]

            # sample n_candidates from full set
            if all_cands.shape[0] > n_candidates:
                sel = rng.choice(all_cands.shape[0], size=n_candidates, replace=False)
                candidates = all_cands[sel]
            else:
                candidates = all_cands
        else:
            # large product -> random permutations from per-field option sets
            candidates = np.empty((n_candidates, n_fields), dtype=np.float32)
            for k in range(n_candidates):
                vec = np.empty((n_fields,), dtype=np.float32)
                for i in range(n_fields):
                    vec[i] = rng.choice(options_per_field[i])
                candidates[k] = vec

        # add several default actions to the candidates
        vals = [0.0, 0.5]
        default_cands = np.asarray(
            [
                list(x)
                for x in itertools.product(vals, repeat=n_fields)
                if sum(x) <= 0.5
            ]
        )

        candidates = np.vstack([candidates, default_cands])

        # Optionally enforce reduced scenario
        if reduced:
            candidates = self._apply_reduced_constraint(candidates)

        # Deduplicate candidates if requested
        if unique and candidates.shape[0] > 0:
            candidates = np.unique(candidates, axis=0)

        return candidates

    @staticmethod
    def _typical_reduction(name: str) -> float:
        """Return a typical REDUCTION fraction in [0,1] of max budget.

        IMPORTANT: In this allocator env, an action is a *reduction* (kg N/ha) per field.
        Smaller reduction => more allocated N.

        Heuristic defaults (tune later if you want):
          - potato:      low reduction (high N demand)
          - winterwheat: medium reduction
          - sugarbeet:   higher reduction (lower N demand)
        """
        name = str(name).lower()
        if "potato" in name:
            return 3.0
        if "wheat" in name:
            return 1.0
        if "sugar" in name or "beet" in name:
            return 0.0
        return 0.0

    def _apply_reduced_constraint(self, arms: np.ndarray) -> np.ndarray:
        """
        Enforce the 'reduced' scenario on sampled reduction vectors.

        Parameters
        ----------
        arms : np.ndarray
            Array of shape (n_candidates, n_fields), where each entry is a
            *reduction* (kg N/ha) for a given field.

        Idea
        ----
        We treat the farm as if each field's max budget was lowered by a fixed
        amount (e.g. 100 kg/ha), but we still allow *redistribution* of that
        reduction across fields.

        Let:
            reduction_per_field = 100 kg/ha   (example)
            required_total_reduction = reduction_per_field * n_fields

        Then we keep only those candidate vectors whose total reduction
        across fields is at least `required_total_reduction`.

        If filtering would remove all candidates, we fall back to the
        unfiltered set.
        """
        arms = np.asarray(arms, dtype=np.float32)

        # Sanity check: arms should have one column per field
        assert arms.shape[1] == self.n_fields, (
            f"Expected arms with {self.n_fields} fields, got {arms.shape[1]}"
        )

        # How much we want to reduce per field in the 'reduced' scenario (kg/ha)
        reduction_per_field = 100.0
        required_total_reduction = reduction_per_field * float(self.n_fields)

        # Total reduction per candidate
        row_sums = arms.sum(axis=1)

        # Keep arms that meet or exceed the required total reduction
        mask = row_sums >= required_total_reduction

        # If everything gets filtered out, fall back to unfiltered
        if not mask.any():
            return arms
        return arms[mask]

    def sample_neighbors(
            self,
            center: np.ndarray,
            n_neighbors: int = 32,
            reduced: bool = False,
            rng: np.random.RandomState | None = None,
    ) -> np.ndarray:
        """
        Sample discrete neighbor arms around a given center arm.

        center: shape (n_fields,) array of reductions/budgets
        Returns: shape (n_neighbors, n_fields)
        """
        if rng is None:
            rng = self.rng

        center = np.asarray(center, dtype=np.float32)
        n_fields = self.n_fields
        agents = list(self.farm.possible_agents)

        neighbors = np.tile(center, (n_neighbors, 1)).astype(np.float32)

        for k in range(n_neighbors):
            # how many fields to perturb in this neighbor (1–3, bounded by n_fields)
            n_changes = int(rng.integers(1, min(3, n_fields) + 1))
            idx_fields = rng.choice(n_fields, size=n_changes, replace=False)

            for j in idx_fields:
                a = agents[j]
                vals = self.base_arms[a]  # 1D np.array of allowed discrete values for this field
                cur_val = neighbors[k, j]

                # find closest index to current value
                idx = int(np.argmin(np.abs(vals - cur_val)))

                # move one step up/down if possible, otherwise random neighbor
                step = int(rng.choice([-1, 1]))
                new_idx = idx + step
                if new_idx < 0 or new_idx >= len(vals):
                    # if we're at the edge, pick some random index
                    new_idx = int(rng.integers(0, len(vals)))
                neighbors[k, j] = vals[new_idx]

        if reduced:
            # If you already have a reduced/global-budget constraint, apply it here.
            # E.g., if you defined `_apply_reduced_constraint` as before:
            neighbors = self._apply_reduced_constraint(neighbors)

        return neighbors

    def filter_historical_info(self, agent_info):
        agents = getattr(self, "agents_order", list(agent_info.keys()))
        for agent in agents:
            agent_info[agent] = {
                k: v
                for k, v in agent_info[agent].items()
                if k in ["RAIN", "TMIN", "IRRAD"]
            }
        return agent_info

    '''
    Init helpers
    '''

    def _warm_up(self, warm_up_year, budget_levels=4):
        # assert all([y < 2020 for y in warm_up_years])
        warm_up_infos: deque[dict] = deque(maxlen=100)
        options = {}
        budget_reductions = [np.zeros(1)]
        if budget_levels > 1:
            budget_reductions = [np.asarray([b * 2 for _ in range(len(self.farm.possible_agents))]) for b in range(0, budget_levels)]
        print('Starting warm up...')
        options['year'] = warm_up_year
        for j in budget_reductions:
            self.farm.reset(seed=self.seed, options=options)
            self.farm.allocate_bandit_budgets(j)
            iter_info = {}
            for agent in self.farm.agent_iter():
                _, _, _, _, infos = self.farm.last()
                action = self.farm.rule_of_thumb(agent)
                if self.farm.terminations[agent]:
                    iter_info[agent] = infos
                    self.farm.step(None)
                else:
                    self.farm.step(action)
            iter_info = self.filter_historical_info(iter_info)
            warm_up_infos.append(iter_info)
            print(self.farm)
        print('Finished warm up...')
        # print('Attempting to save pickle...')
        # with open(os.path.join(_CONFIG_PATH, 'warm_up_infos.pkl'), 'wb') as f:
        #     pickle.dump(warm_up_infos, file=f)
        # print('Successfully saved!')
        return warm_up_infos

    def _init_envs(self, args, warm_up_eps=0):
        # make farm
        dict_fields = None
        if args.farm is not None:
            with open(os.path.join(_SCENARIO_PATH, f"{self.region}", "2020",
                                   f"farmer_{self.farm_id}.yaml"), 'rb') as f:
                dict_fields = yaml.safe_load(f)

        self.farm = MultiFieldEnv(
            warm_up=self.warm_up_eps,
            years=self.years,
            farm_dict=dict_fields,
            reward=self.field_reward
        )

        # Do some warm up episodes
        self.warm_up_infos = None
        if warm_up_eps > 0:
            years_warm_up = list(range(2020 - 1, 2020 - warm_up_eps - 1, -1))
            print(years_warm_up)

            for year in years_warm_up:
                self.get_rotation_year(2020 + ((2019 - year) % 5))
                self.warm_up_infos = self._warm_up(year)

        self.env_agent = None
        if args is not None and hasattr(args, 'use_model'):
            if args.model_dir == "ROT":
                self.env_agent = RoTAgent(
                    env=self.farm,
                    render=args.render,
                )
            elif args.model_dir == "random":
                self.env_agent = RandomAgent(
                    env=self.farm,
                    render=args.render,
                )
            else:
                saved_model = load_model(args)
                self.original_saved_model = deepcopy(saved_model)
                if args.farm is not None:
                    saved_model = model_picker(self.original_saved_model, dict_fields)
                self.env_agent = MultiRLAgent(
                    env = self.farm,
                    saved_model=saved_model,
                    render=args.render,
                )
            self.farm = self.env_agent.env

    def _init_spaces(self):

        # Set up action space based on farm
        self.base_arms = _make_base_arms(self, cap=self.cap)  # if len(self.farm.possible_agents) < 8 else 0.4)
        # self.super_arms = _make_super_arms(self, self.base_arms)
        # self.super_arms_reduced = _make_super_arms(self, self.base_arms, reduced=True)
        # assert self.super_arms.size > self.super_arms_reduced.size
        # self.top_super_arms = _make_topk_super_arms(
        #     self.base_arms,
        #     self.farm.possible_agents,
        #     top_k=3
        # )
        # self.super_arm_to_idx = {
        #     tuple(a): i for i, a in enumerate(self.super_arms)
        # }

        highs = np.array(self.max_budgets, dtype=np.float32)
        lows = np.zeros_like(highs, dtype=np.float32)

        # Action space
        # discrete and multi_discrete is not implemented properly yet.
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(len(self.super_arms))
        if self.action_type == 'multi_discrete':
            self.action_space = spaces.MultiDiscrete(
                [
                    len(self.base_arms[a])
                    for a in self.farm.possible_agents
                ]
            )
        if self.action_type == 'continuous':
            self.action_space = spaces.Box(low=lows, high=highs, shape=(self.n_fields,), dtype=np.float32)

        # Observation space
        if not self.flat_context:
            self.observation_space = spaces.Dict(
                {
                    feature: spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=(self.n_fields,),
                        dtype=np.float32
                    )
                    for feature in self._get_context_keys()
                }
            )
        else:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=(len(self._get_context_keys()) * self.n_fields,),
                dtype=np.float32
            )

    def _init_meta_info(self):
        # Canonical ordering of fields/agents — use this everywhere
        self.agents_order = list(self.farm.possible_agents)

        self.n_fields = len(self.agents_order)
        self.global_budget = self.farm.global_budget

        self.parcel_meta_infos = {
            agent: {
                'max_budget': self.farm.fields[agent].unwrapped.max_budget_n,
                'crop': self.farm.fields[agent].unwrapped.crop,
                'crop_code': self.farm.fields[agent].unwrapped.CROP_CODE_MAP[
                    self.farm.fields[agent].unwrapped.crop
                ],
                'soil_type': self.farm.fields[agent].unwrapped.soil_type,
                'area': self.farm.fields[agent].unwrapped.area,
            }
            for agent in self.agents_order
        }

        self.max_budgets = self._get_max_budgets()

# ---------------------------------------------------------------------
# Single-field version of AllocationBandit
# ---------------------------------------------------------------------
class ParcelAllocationBandit(AllocationBandit):
    """AllocationBandit but for a SINGLE field (one bandit per field).

    - Reuses AllocationBandit logic by subclassing (so all samplers & context logic remain identical).
    - Only overrides:
        * _init_envs: load a farm_dict filtered to one field
        * get_rotation_year: reload rotation yaml but keep only that field

    Constraints honored:
    - No changes to MultiFieldEnv or ParcelEnv.
    """

    def __init__(
        self,
        *,
        field_key: str,
        delta_kg: float = 5.0,
        warm_up_eps: int = 10,
        cap: float = 1.0,
        years: list = get_default_years(),
        seed: int = 107,
        action_type: str = "continuous",
        args: argparse.Namespace = None,
        field_reward: str = "NSU",
        flat_context: bool = True,
        region: str | None = None,
        farm_id: int | None = None,
        render: bool = False,
    ):
        if not isinstance(field_key, str) or not field_key:
            raise ValueError("ParcelAllocationBandit requires a non-empty `field_key` (e.g. 'field-1').")
        self.field_key = field_key

        super().__init__(
            delta_kg=delta_kg,
            warm_up_eps=warm_up_eps,
            cap=cap,
            years=years,
            seed=seed,
            action_type=action_type,
            args=args,
            field_reward=field_reward,
            flat_context=flat_context,
            region=region,
            farm_id=farm_id,
            render=render,
        )

        # Sanity: enforce single field
        if getattr(self, "n_fields", None) != 1:
            raise RuntimeError(
                f"ParcelAllocationBandit expected n_fields==1 after init, got {getattr(self, 'n_fields', None)}. "
                "This means scenario YAML filtering did not apply."
            )

    # -------------------------
    # Overrides: env init + rotation year
    # -------------------------

    def _init_envs(self, args, warm_up_eps=0):
        """Same as AllocationBandit._init_envs but loads ONLY `self.field_key`."""
        dict_fields = None

        if args is not None and getattr(args, "farm", None) is not None:
            farm_path = os.path.join(
                _SCENARIO_PATH, f"{self.region}", "2020", f"farmer_{self.farm_id}.yaml"
            )
            with open(farm_path, "rb") as f:
                dict_all = yaml.safe_load(f)

            if self.field_key not in dict_all:
                raise KeyError(
                    f"field_key='{self.field_key}' not found in {farm_path}. "
                    f"Available: {list(dict_all.keys())}"
                )

            dict_fields = {self.field_key: dict_all[self.field_key]}

        # Build MultiFieldEnv with ONLY one field
        self.farm = MultiFieldEnv(
            warm_up=self.warm_up_eps,
            years=self.years,
            farm_dict=dict_fields,
            reward=self.field_reward,
        )

        # Warm up episodes (same logic, but now it's one field)
        self.warm_up_infos = None
        if warm_up_eps > 0:
            years_warm_up = list(range(2020 - 1, 2020 - warm_up_eps - 1, -1))
            print(years_warm_up)

            for year in years_warm_up:
                self.get_rotation_year(2020 + ((2019 - year) % 5))
                self.warm_up_infos = self._warm_up(year)

        # Policy selection (same as AllocationBandit, but pass filtered dict_fields to model_picker)
        self.env_agent = None
        if args is not None and hasattr(args, "use_model"):
            if args.model_dir == "ROT":
                self.env_agent = RoTAgent(env=self.farm, render=args.render)
            elif args.model_dir == "random":
                self.env_agent = RandomAgent(env=self.farm, render=args.render)
            else:
                saved_model = load_model(args)
                self.original_saved_model = deepcopy(saved_model)

                if dict_fields is not None:
                    saved_model = model_picker(self.original_saved_model, dict_fields)

                self.env_agent = MultiRLAgent(
                    env=self.farm,
                    saved_model=saved_model,
                    render=args.render,
                )

            # keep AllocationBandit invariant
            self.farm = self.env_agent.env

    def get_rotation_year(self, year: int):
        """Reload scenario YAML for `year`, but keep ONLY `self.field_key`."""
        assert year in [2020, 2021, 2022, 2023, 2024], f"Unexpected rotation year: {year}"

        farm_path = os.path.join(
            _SCENARIO_PATH, f"{self.region}", f"{year}", f"farmer_{self.farm_id}.yaml"
        )
        with open(farm_path, "rb") as f:
            dict_all = yaml.safe_load(f)

        if self.field_key not in dict_all:
            raise KeyError(
                f"field_key='{self.field_key}' not found in {farm_path}. "
                f"Available: {list(dict_all.keys())}"
            )

        dict_fields = {self.field_key: dict_all[self.field_key]}

        # IMPORTANT: keep compatibility with your existing AllocationBandit approach
        self.farm.set_new_fields(dict_fields, year=year)

        # Recompute meta/spaces (these depend on possible_agents + budgets)
        self._init_meta_info()
        self._init_spaces()

        # If using learned policy, reload the correct policy for this field
        if getattr(self, "original_saved_model", None) is not None:
            try:
                saved_model = model_picker(self.original_saved_model, dict_fields)
                self.env_agent = MultiRLAgent(
                    env=self.farm,
                    saved_model=saved_model,
                    render=getattr(getattr(self, "env_agent", None), "render", False),
                )
                self.farm = self.env_agent.env
            except Exception:
                # If policy reload fails, keep the current policy; env still runs.
                pass

        return dict_fields

    @property
    def agent_name(self) -> str:
        """Convenience: the single field agent name."""
        return self.field_key