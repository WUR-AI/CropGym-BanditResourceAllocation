import os
import itertools
import copy
from functools import partial
from typing import Sequence, Any, cast
from collections import defaultdict
import datetime

import pettingzoo
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import numpy as np
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space, flatdim, flatten

from cropgymzoo.envs.worker_env import ParallelRLWorkers
from cropgymzoo import _DEFAULT_LOGDIR

from pettingzoo.utils.conversions import parallel_to_aec
from pettingzoo import ParallelEnv




try:
    # ---- Tianshou imports ----
    from tianshou.data import Collector, VectorReplayBuffer, Batch
    from tianshou.env import PettingZooEnv, DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs, BaseVectorEnv, VectorEnvWrapper
    from tianshou.utils.net.common import NetBase, RecurrentStateBatch
    from tianshou.utils.net.discrete import Actor, Critic  # will wrap our GRU core
    from tianshou.utils.net.common import Recurrent
    from tianshou.utils.logger.tensorboard import TensorboardLogger
    from tianshou.utils import tqdm_config
    from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
    from tianshou.trainer import OnpolicyTrainer
    from tianshou.utils.statistics import RunningMeanStd
except ImportError:
    tianshou = None


'''
Obs wrapper
'''


def flatten_dict_space(
    dict_space: gym.spaces.Dict,
    key_order: tuple[str, ...] | None = None
) -> tuple[gym.spaces.Box, tuple[str, ...]]:
    """
    Turn a Dict(...) gym space into a single Box that stacks all leaves.

    Returns
    -------
    flat_space : Box
    key_order  : the deterministic key order used
    slices     : slices[i] tells you where key_order[i] lives in the flat vector
    """
    if key_order is None:
        # Gym's Dict preserves insertion order, but be explicit
        key_order = tuple(dict_space.spaces.keys())

    lows, highs, slices = [], [], []
    start = 0
    for k in key_order:
        space_k = dict_space[k]
        if not isinstance(space_k, gym.spaces.Box):
            raise TypeError(
                f"Only Box leaves supported, got {type(space_k)} for key '{k}'"
            )
        flat_low  = np.asarray(space_k.low , dtype=np.float32).reshape(-1)
        flat_high = np.asarray(space_k.high, dtype=np.float32).reshape(-1)
        lows.append(flat_low)
        highs.append(flat_high)

        end = start + flat_low.size
        slices.append(slice(start, end))
        start = end

    low  = np.concatenate(lows)
    high = np.concatenate(highs)

    flat_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    return flat_space, key_order


# ------------------------------------------------------------------ #
# 2)  SAMPLE (dict)  →  np.ndarray
# ------------------------------------------------------------------ #
def flatten_dict_obs(
    obs_dict: dict[str, np.ndarray | float | int],
    key_order: tuple[str, ...],
) -> np.ndarray:
    """Flatten a single observation dict using the same key order."""

    first = np.asarray(obs_dict[key_order[0]], dtype=np.float32)
    n_envs = first.shape[0]

    flats = []

    for k in key_order:
        # v = obs_dict[k]
        # flats.append(np.asarray(v, dtype=np.float32).reshape(-1))
        v = np.asarray(obs_dict[k], dtype=np.float32)  # (n_envs, …)
        flats.append(v.reshape(n_envs, -1))  # keep env axis
    return np.concatenate(flats, axis=-1)


class MultiAgentDictObsFlatten:
    """Flatten nested dict observations for **every agent** and carry the mask.

    After wrapping, the env returns:
        obs  : Dict[str, 1‑D np.ndarray]
        info : Dict[str, Dict], where each dict has "action_mask": ndarray

    This matches what `MultiAgentPolicyManager` expects – each policy sees its
    own flattened vector plus `info[agent]["action_mask"]`.
    """

    def __init__(self, env: gym.Env | pettingzoo.AECEnv | PettingZooEnv):
        sample_obs, _ = env.reset(seed=0)
        self.agent_ids = env.unwrapped.possible_agents
        self.current_masks: dict[str, np.ndarray] = {k: None for k in self.agent_ids}  # type: ignore

        # Build obs space dict with flattened dimensions per agent
        spaces = {}
        for agent_id in self.agent_ids:
            spaces[agent_id] = sample_obs[agent_id]
        self.observation_spaces = gym.spaces.Dict(spaces)

    # ------------------------------------------------------------------
    @staticmethod
    def _dict_to_flat(d: dict) -> np.ndarray:
        parts: list[np.ndarray] = []
        for v in d.values():
            if isinstance(v, dict):
                parts.append(MultiAgentDictObsFlatten._dict_to_flat(v))
            else:
                parts.append(np.asarray(v, dtype=np.float32).flatten())
        return np.concatenate(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _split_and_flatten(obs: dict) -> np.ndarray:
        """Return (flat, mask) from one *agent*’s raw observation dict."""
        obs = obs.copy()  # avoid side‑effects
        parts: list[np.ndarray] = []
        for v in obs.values():
            if isinstance(v, dict):
                parts.append(MultiAgentDictObsFlatten._dict_to_flat(v))
            else:
                parts.append(np.asarray(v, dtype=np.float32).flatten())
        flat_obs = np.concatenate(parts, dtype=np.float32)
        return flat_obs

    # ------------------------------------------------------------------
    def _process_obs_dict(self, obs_dict: dict[str, dict]) -> dict[str, np.ndarray]:
        flat_dict = {}
        for agent, ob in obs_dict.items():
            flat, mask = self._split_and_flatten(ob)
            flat_dict[agent] = flat
            self.current_masks[agent] = mask
        return flat_dict

    # ------------------------------------------------------------------
    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        flat_obs = self._process_obs_dict(obs)
        info = info or {}
        for a in self.agent_ids:
            info.setdefault(a, {})["action_mask"] = self.current_masks[a]
        return flat_obs, info

    def step(self, action):  # type: ignore[override]
        obs, rew, term, trunc, info = self.env.step(action)
        flat_obs = self._process_obs_dict(obs)
        for a in self.agent_ids:
            info.setdefault(a, {})["action_mask"] = self.current_masks[a]
        return flat_obs, rew, term, trunc, info

class VecNormObs(VectorEnvNormObs):
    """
    Normalises a Dict observation by first flattening it (stable key order),
    then applying a single RunningMeanStd.
    """
    def __init__(self,
                 venv: BaseVectorEnv,
                 obs_dim: int | tuple[int] | None = None,
                 update_obs_rms: bool = True,
                 device="cpu"):
        super().__init__(venv, update_obs_rms)
        self.obs_dim = obs_dim
        self.device = device

    def collapse_info_dict(self, info: dict[str, dict[str, list | float]]) -> dict[str, dict[str, float]]:
        """
        Replace any list-valued items in the info dict with their last element.

        Args:
            info: Dict[agent_id, Dict[info_key, value]]

        Returns:
            A cleaned-up info dict with all values scalar (no growing lists).
        """
        collapsed = {}
        for k, v in info.items():
            if isinstance(v, list) and v:
                collapsed[k] = v[-1]
            elif isinstance(v, list) and not v:
                collapsed[k] = np.nan
            else:
                collapsed[k] = v
        return collapsed

    # ---------------- overrides ------------------------- #
    def reset(
        self,
        env_id: int | list[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs, info = self.venv.reset(env_id, **kwargs)

        if isinstance(obs, tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )

        obs_extracted = obs.copy()
        if isinstance(obs_extracted, (list, np.ndarray)):  # the common case
            obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs_extracted])

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_extracted)
        obs_extracted = self._norm_obs(obs_extracted)


        for i, venv_obs in enumerate(obs_extracted):
            obs[i]["obs"] = obs_extracted[i]

        for i, _i in enumerate(info):
            info[i] = self.collapse_info_dict(_i)

        return obs, info

    def step(
            self,
            action: np.ndarray | torch.Tensor | None,
            id: int | list[int] | np.ndarray | None = None,
    ):
        step_results = self.venv.step(action, id)

        obs = step_results[0].copy()
        if isinstance(obs, (list, np.ndarray)):  # the common case
            obs_extracted = np.array([d["observation"] if "observation" in d else d["obs"] for d in obs])

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_extracted)
        obs_extracted = self._norm_obs(obs_extracted)

        for i, venv_obs in enumerate(obs_extracted):
            obs[i]["obs"] = obs_extracted[i]

        info = []
        for i, _i in enumerate(step_results[-1]):
            info.append(self.collapse_info_dict(_i))
        info = np.stack(info)

        return (obs, *step_results[1:-1], info)


def _new_rms():
    # from tianshou.utils.statistics import RunningMeanStd
    return RunningMeanStd()


class DictVectorEnvNormObs(VectorEnvWrapper):
    """
    Normalise every scalar in  ``obs_item["obs"]``  *independently*
    (zero mean, unit variance).

    Supports both batching styles a vector-env can emit:

    ❶  **list of per-env dicts**              ← what you actually get
        `[{"obs": {...}, "mask": ...},                 # env 0
          {"obs": {...}, "mask": ...}]                 # env 1 …]`

    ❷  **dict of arrays** (rare, but still valid)
        `{"obs": {"TEMP": np.ndarray[n_envs], …}, "mask": …}`
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(venv)
        self.update_obs_rms = update_obs_rms
        self.eps = eps
        # one RunningMeanStd per scalar key (mean / std are 0-D arrays)
        self.obs_rms: defaultdict[str, RunningMeanStd] = defaultdict(_new_rms)

    # ------------------------------------------------------------------ #
    # ── helpers -------------------------------------------------------- #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scalar_dicts_from(
        obs_container: Any,
    ) -> list[dict[str, float]]:
        """
        Return a **list** whose i-th element is the scalar dict
        of env *i*, *AND* keep references so modifications happen in place.
        """
        if isinstance(obs_container, (list, np.ndarray)):           # the common case
            return [d["observation"] if "observation" in d else d["obs"] for d in obs_container]

        if isinstance(obs_container, tuple):          # rarely tuples
            return [d["observation"] if "observation" in d else d["obs"] for d in list(obs_container)]

        raise TypeError(
            f"Unsupported obs container type: {type(obs_container)}"
        )

    # ------------------------------------------------------------------ #
    def _normalise_in_place(self, obs_container: Any) -> None:
        """
        Update running statistics and overwrite the original scalars
        with their normalised values.
        """
        per_env_dicts = self._scalar_dicts_from(obs_container)

        # 1. stack   key → np.ndarray[n_env]
        stacked: dict[str, np.ndarray] = {}
        for k in per_env_dicts[0]:
            stacked[k] = np.asarray(
                [d[k] for d in per_env_dicts], dtype=np.float32
            )
        print(stacked)
        print(self.obs_rms)
        # 2. update rms & normalise
        for k, vec in stacked.items():
            rms = self.obs_rms[k]
            if self.update_obs_rms:
                rms.update(vec)
            stacked[k] = (vec - rms.mean) / (rms.var + self.eps)

        # 3. write back to the original objects
        for i, d in enumerate(per_env_dicts):
            for k in d:
                d[k] = stacked[k][i]

    # ------------------------------------------------------------------ #
    # ── VectorEnv API -------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def reset(self, env_id=None, **kwargs):
        obs, info = self.venv.reset(env_id, **kwargs)
        self._normalise_in_place(obs)
        return obs, info

    def step(self, action, id=None):
        obs, rew, term, trunc, info = self.venv.step(action, id)
        self._normalise_in_place(obs)
        return obs, rew, term, trunc, info

    # ------------------------------------------------------------------ #
    # ── share statistics between train / test envs -------------------- #
    # ------------------------------------------------------------------ #
    def get_obs_rms(self):
        return {k: copy.deepcopy(v) for k, v in self.obs_rms.items()}

    def set_obs_rms(self, other: dict[str, RunningMeanStd]):
        self.obs_rms = defaultdict(
            lambda: RunningMeanStd(),
            {k: copy.deepcopy(v) for k, v in other.items()},
        )

# -----------------------------------------------------------------------------
# 3.  Recurrent network --------------------------------------------------------
# -----------------------------------------------------------------------------
class GRUBackbone(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int | Sequence[int] = 128, activation = nn.Tanh):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)
            if len(hidden_dims) == 0:
                raise ValueError("`hidden_dim` sequence must contain at least one element.")

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), activation()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        # 3️⃣ GRU whose input_size **and** hidden_size = last MLP width
        last_dim = hidden_dims[-1]
        self.gru = nn.GRU(input_size=last_dim,
                          hidden_size=last_dim,
                          batch_first=True)

        self._hidden_dim = last_dim  # handy for downstream code

    def forward(self, obs: torch.Tensor,
                state: torch.Tensor | None = None):
        """
        obs   : (batch, obs_dim)
        state : (1, batch, hidden) for a single-layer GRU (or None)
        """
        x = self.mlp(obs)  # (batch, last_dim)
        x = x.unsqueeze(1)  # add time dimension → (batch, 1, last_dim)

        y, h = self.gru(x, state)  # y: (batch, 1, last_dim)
        y = y.squeeze(1)  # remove the time dim  → (batch, last_dim)
        return y, h

class RecurrentGRU(NetBase[RecurrentStateBatch]):
    """Tianshou-compatible GRU network (same API as common.Recurrent)."""

    def __init__(
        self,
        layer_num: int,
        state_shape: int | Sequence[int],
        action_shape: int | Sequence[int],
        hidden_layer_size: int = 128,
        device: str | int | torch.device = "cpu",
        key_order: tuple[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.key_order = key_order
        self.output_dim = int(np.prod(action_shape))

        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.gru = nn.GRU(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))

    def forward(                      # pylint: disable=arguments-differ
        self,
        obs: Batch,
        state: RecurrentStateBatch | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Batch]:

        if isinstance(obs, Batch):
            obs = obs.obs  # or dict(obs)   (no copy of scalars)

        # input -> [bsz, len, dim] for training, [bsz, dim] for eval
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).to(self.device)

        if obs.ndim == 1:  # single env
            obs = obs.unsqueeze(0)  # [1, D]

        # 2. feed-forward + add time dim
        x = torch.tanh(self.fc1(obs)) # [B, H]
        x = x.unsqueeze(1)  # [B, 1, H]
        # self.gru.flatten_parameters()

        if state is None or "hidden" not in state:
            y, h_in = self.gru(x)            # hidden: [num_layers, bsz, h]
        else:
            h_in = state["hidden"].transpose(0, 1).contiguous()
        y, hidden = self.gru(x, h_in)

        logits = self.fc2(y.squeeze(1))              # take last time-step

        next_state = Batch({"hidden": hidden.transpose(0, 1).detach()})
        return logits, next_state

class RecurrentLSTM(Recurrent):
    def __init__(
            self,
            layer_num: int,
            state_shape: int | Sequence[int],
            action_shape,
            device: str | int | torch.device = "cpu",
            hidden_layer_size: int = 128,
    ) -> None:
        super().__init__(
            layer_num=layer_num,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            hidden_layer_size=hidden_layer_size,
        )
        self.output_dim = int(np.prod(action_shape))

    def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            state: RecurrentStateBatch | None = None,
            info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:

        if isinstance(obs, Batch):
            obs = obs.obs  # or dict(obs)   (no copy of scalars)

        print(state)
        return super().forward(obs, state, info)




# ---------------------------------------------------------------------------
class MaskedActor(Actor):
    """Actor that zeroes logits for illegal actions via provided mask."""

    def __init__(self, preprocess_net, action_dim, device='cpu', key_order = None):
        super().__init__(preprocess_net=preprocess_net, action_shape=action_dim,
                         softmax_output=False, device=device)  # remember for logits
        self.key_order = key_order

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None, info: dict = {}):

        latent, h = self.preprocess(obs, state)
        logits = self.last(latent)
        if isinstance(info, dict) and "action_mask" in info:
            mask = torch.as_tensor(info["action_mask"], device=logits.device)
            logits[mask == 0] = -1e10
        return logits, h

class DictObsCritic(Critic):
    def __init__(self, preprocess_net, device='cpu', key_order = None):
        super().__init__(preprocess_net=preprocess_net)
        self.device = device
        self.key_order = key_order

    def forward(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:

        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)


def make_recurrent_policy(obs_dim: int, act_dim: int, lr: float = 3e-4, hidden: int = 128, layer_num: int = 1, key_order=None) -> PPOPolicy:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor_net = RecurrentGRU(layer_num=layer_num, state_shape=obs_dim, action_shape=act_dim, device=device, hidden_layer_size=hidden) #GRUBackbone(obs_dim, hidden_dim=[128, 128])
    critic_net = RecurrentGRU(layer_num=layer_num, state_shape=obs_dim, action_shape=act_dim, device=device, hidden_layer_size=hidden) #GRUBackbone(obs_dim, hidden_dim=[128, 128])

    actor = MaskedActor(preprocess_net=actor_net, action_dim=act_dim, key_order=key_order).to(device)
    critic = DictObsCritic(preprocess_net=critic_net, key_order=key_order).to(device)

    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    # dist = torch.distributions.Categorical  # DISCRETE!

    dist = lambda logits: torch.distributions.Categorical(logits=logits)

    return PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
        eps_clip=0.2,
        value_clip=True,
        action_space=gym.spaces.Discrete(act_dim),
        action_scaling=False,
        reward_normalization=False,
    ).to(device)

def make_vec_env(parallel: bool = True, indep: bool = True, num_envs: int = 4) -> SubprocVectorEnv | DummyVectorEnv:
    """Each subprocess builds → PettingZooEnv"""
    env_fns = [partial(get_petting_zoo_env, indep) for _ in range(num_envs)]
    if parallel:
        return SubprocVectorEnv(env_fns)
    else:
        return DummyVectorEnv(env_fns)

def get_petting_zoo_env(indep):
    env = make_env(independent_learning=indep)
    env = PettingZooEnv(env)
    return env

def make_env(independent_learning=True): # type: ignore
    """Return one wrapped PettingZoo environment instance."""
    env = ParallelRLWorkers(
        warm_up=0,
        shared_obs=False if independent_learning else True,
        training=True,
    )
    if isinstance(env, ParallelEnv):
        env = parallel_to_aec(env)
    return env

def get_dummy_env():
    return ParallelRLWorkers()

def train_gru_ppo(hyperparams: dict):

    # extract dict
    indep = hyperparams.get('independent', True)
    train_envs_num = hyperparams.get('train_envs_num', 1)
    test_envs_num = hyperparams.get('test_envs_num', 1)
    seed = hyperparams.get('seed', 107)
    lr = hyperparams.get('lr', 1e-3)
    buffer_size = hyperparams.get('buffer_size', int(10_000))
    epoch = hyperparams.get('epoch', 300)
    logdir = hyperparams.get('logdir', _DEFAULT_LOGDIR)
    # batch_size = hyperparams.get('batch_size', 64)
    step_per_epoch = hyperparams.get('step_per_epoch', 10_000)
    step_per_collect = hyperparams.get('step_per_collect', 64)
    episode_per_collect = hyperparams.get('episode_per_collect', 8)
    repeat_per_collect = hyperparams.get('repeat_per_collect', 2)
    parallel = hyperparams.get('parallel', False)

    # Inspect one spawned env to grab spaces & agent list
    dummy_env = get_dummy_env()
    dummy_env.reset(seed=seed)
    sample_obs, _, _, _, _ = dummy_env.unwrapped.last()
    first_agent = 'field-1'
    observation_space = dummy_env.sample_observation_space_agent()
    # flat_space, key_order = flatten_dict_space(observation_space)
    obs_dim = observation_space.shape

    # Create vector env
    train_envs = make_vec_env(parallel, indep, train_envs_num)
    test_envs = make_vec_env(parallel, indep, test_envs_num)

    # Normalize Vector env,  using subclassed norm class
    # train_envs = DictVectorEnvNormObs(train_envs, update_obs_rms=True)  #, dict_space=dummy_env.sample_observation_space_agent())
    # test_envs = DictVectorEnvNormObs(test_envs, update_obs_rms=False)  #, dict_space=dummy_env.sample_observation_space_agent())
    train_envs = VecNormObs(train_envs, update_obs_rms=True)
    test_envs = VecNormObs(test_envs, update_obs_rms=False)
    train_envs.reset(options={'year': np.random.choice(range(1951, 2024))})
    test_envs.set_obs_rms(train_envs.get_obs_rms())

    # assuming Discrete(.) identical for all
    act_dim = dummy_env.action_spaces[first_agent].n
    agents = dummy_env.possible_agents

    # Build policies
    if indep:
        policies = {a: make_recurrent_policy(obs_dim, act_dim, lr) for a in agents}
    else:
        shared = make_recurrent_policy(obs_dim, act_dim, lr)
        policies = {a: shared for a in agents}

    policy_mgr = MultiAgentPolicyManager(policies=list(policies.values()),
                                         env=PettingZooEnv(dummy_env),)

    # Buffers / collectors
    train_collector = Collector(
        policy=policy_mgr,
        env=train_envs,
        buffer=VectorReplayBuffer(
            total_size=buffer_size,
            buffer_num=len(train_envs) * len(policies)
        ),  # use this buffer
        exploration_noise=True
    )
    test_collector = Collector(
        policy=policy_mgr,
        env=test_envs,
    )


    # Logger
    run_name = f"PPO_GRU_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(os.path.join(logdir, run_name))
    # writer.add_text("hyperparams", str(*hyperparams.values()))
    logger = TensorboardLogger(writer)

    # make callbacks within this method
    os.makedirs(os.path.join(logdir, run_name, "best"), exist_ok=True)
    # os.makedirs(os.path.join(logdir, "best", run_name), exist_ok=True)
    os.makedirs(os.path.join(logdir, run_name, "checkpoints"), exist_ok=True)
    # os.makedirs(os.path.join(logdir, "checkpoints", run_name), exist_ok=True)

    def save_best_fn(ma_policy: MultiAgentPolicyManager):
        torch.save(
            {
                "models": {
                    aid: p.state_dict()  # one file for every agent
                    for aid, p in ma_policy.policies.items()
                },
                "obs_rms": train_envs.get_obs_rms(),
            },
            os.path.join(logdir, run_name, "best", "best.pth")
        )

    def save_checkpoint_fn(epoch: int, env_step: int, grad_step: int) -> None:
        # copy running statistics into the frozen eval envs *once per epoch*
        test_envs.set_obs_rms(train_envs.get_obs_rms())
        torch.save(
            {
                "epoch": epoch,
                "env_step": env_step,
                "grad_step": grad_step,
                "model": policy_mgr.state_dict(),
                "obs_rms": train_envs.get_obs_rms(),
            },
            os.path.join(logdir, run_name, "checkpoints", f"check_{epoch:04d}.pth")
        )


    result = OnpolicyTrainer(
        policy=policy_mgr,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        # step_per_collect=step_per_collect,
        episode_per_collect=episode_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=2,
        batch_size=step_per_collect * len(train_envs),
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
    ).run()
    print(f"Training done → best avg reward: {result['best_reward']:.3f}")


'''
NOTE FOR WHEN RESUMING MODEL

ckpt = torch.load("checkpoints/best.pth", map_location="cpu")

policy.load_state_dict(ckpt["model"])
policy.optim.load_state_dict(ckpt["optimizer"])        # if you saved it

train_envs.set_obs_rms(ckpt["obs_rms"])                # keep collecting
test_envs.set_obs_rms(ckpt["obs_rms"])                 # deterministic eval
'''





