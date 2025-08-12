import os
from functools import partial
import datetime
from typing import Sequence
from argparse import Namespace

from copy import deepcopy

import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import numpy as np
import gymnasium as gym
import supersuit as ss

from cropgymzoo.agents.networks_tianshou import RecurrentGRU, MaskedActor, DictObsCritic, NetObs
from cropgymzoo.agents.marl_algorithms_tianshou import IPPOPolicy, IPPOCollector
from cropgymzoo.envs.multi_field_env import MultiFieldEnv

from cropgymzoo.envs.wrappers_tianshou import MultiAgentVecNormObs
from cropgymzoo.utils.callbacks_tianshou import yearly_eval_test_fn, save_checkpoint_fn, save_best_fn

try:
    # ---- Tianshou imports ----
    from tianshou.data import Collector, VectorReplayBuffer, Batch, ReplayBuffer
    from tianshou.data.collector import EpisodeRolloutHookProtocol, StepHook
    from tianshou.env import PettingZooEnv, DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs, BaseVectorEnv, VectorEnvWrapper, ShmemVectorEnv
    from tianshou.utils.net.common import NetBase, RecurrentStateBatch, Net
    from tianshou.utils.net.discrete import Actor, Critic  # will wrap our GRU core
    from tianshou.utils.net.common import Recurrent
    from tianshou.utils.logger.tensorboard import TensorboardLogger
    from tianshou.utils import tqdm_config
    from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
    from tianshou.trainer import OnpolicyTrainer
    from tianshou.utils.statistics import RunningMeanStd
except ImportError:
    tianshou = None


class HiddenStateHook(StepHook):
    """
    Let's try using this. Because of the stupid MARL RNN bug from the Tianshou collector.
    """
    def __init__(self, agents: list, buffer: ReplayBuffer, num_envs: int):
        self.hidden_state_bank = Batch({
            agent: Batch() for agent in agents
        })

        self.buffer = buffer
        # global_env_id -> { agent_id -> last pending buffer index }
        self._pending_idx_by_env_agent: dict[int, dict] = {i: {} for i in range(num_envs)}
        # self.previous_hidden_state = Batch()

    def __call__(
            self,
            action_batch,
            rollout_batch,
    ) -> None:
        hidden_state = getattr(action_batch, "hidden_state", None)
        # temp_hidden_state = deepcopy(hidden_state)
        hidden_state.replace_empty_batches_by_none()
        if isinstance(hidden_state, Batch) and len(hidden_state):
            for agent, state in hidden_state.items():
                if state is not None:
                    self.hidden_state_bank[agent] = state

        # check deaths
        alives = getattr(rollout_batch.info, "Alive")
        agent_ids = getattr(rollout_batch.obs, "agent_id")
        for i, (agent_id, alive) in enumerate(zip(agent_ids, alives)):
            if alive is False:
                self.hidden_state_bank[agent_id] = Batch()

        action_batch.hidden_state = deepcopy(self.hidden_state_bank)
        action_batch.hidden_state.policy_entry = deepcopy(self.hidden_state_bank)
        rollout_batch.policy.hidden_state = deepcopy(self.hidden_state_bank)

        # """Pre-add: close previous pending transitions with the *current* obs (same agent)."""
        # # Which agent acted in each ready env this step:
        # agent_ids = getattr(rollout_batch.obs, "agent_id")
        # done_R = getattr(rollout_batch, "done")
        #
        # for local_i, global_env_id in enumerate(ready_env_ids_R):
        #     agent_id = agent_ids[local_i]
        #     pendings = self._pending_idx_by_env_agent[int(global_env_id)]
        #     prev_idx = pendings.pop(agent_id, None)
        #     if prev_idx is not None:
        #         # Set obs_next of the previous same-agent transition to the *current* obs
        #         self.buffer.set_array_at_key(
        #             rollout_batch.obs[local_i],
        #             key="obs_next",
        #             index=np.array([int(prev_idx)]),
        #         )
        #
        # # If an env finished now, flush any leftovers with a terminal-like obs_next (shape-safe)
        # for local_i, global_env_id in enumerate(ready_env_ids_R):
        #     if bool(done_R[local_i]):
        #         pendings = self._pending_idx_by_env_agent[int(global_env_id)]
        #         if pendings:
        #             terminal_obs_like = rollout_batch.obs_next[local_i]
        #             for _, prev_idx in list(pendings.items()):
        #                 self.buffer.set_array_at_key(
        #                     terminal_obs_like,
        #                     key="obs_next",
        #                     index=np.array([int(prev_idx)]),
        #                 )
        #             self._pending_idx_by_env_agent[int(global_env_id)] = {}

    def post_add(self, insertion_idx_R: np.ndarray, rollout_batch: Batch, *, ready_env_ids_R: np.ndarray) -> None:
        """Post-add: register the just-added transitions as pending for when the agent acts next."""
        agent_ids = getattr(rollout_batch.obs, "agent_id")
        for local_i, global_env_id in enumerate(ready_env_ids_R):
            agent_id = agent_ids[local_i]
            self._pending_idx_by_env_agent[int(global_env_id)][agent_id] = int(insertion_idx_R[local_i])

    def reset_env(self, env_done_global_idx_D: np.ndarray) -> None:
        """Optional: call on resets to clear any stale pendings for those envs."""
        for g in env_done_global_idx_D:
            self._pending_idx_by_env_agent[int(g)] = {}


    def hidden_state_stuff(self, action_batch, rollout_batch):
        # check states
        hidden_state = getattr(action_batch, "hidden_state", None)
        # temp_hidden_state = deepcopy(hidden_state)
        hidden_state.replace_empty_batches_by_none()
        if isinstance(hidden_state, Batch) and len(hidden_state):
            for agent, state in hidden_state.items():
                if state is not None:
                    self.hidden_state_bank[agent] = state

        # check deaths
        alives = getattr(rollout_batch.info, "Alive")
        agent_ids = getattr(rollout_batch.obs, "agent_id")
        for i, (agent_id, alive) in enumerate(zip(agent_ids, alives)):
            if alive is False:
                self.hidden_state_bank[agent_id] = Batch()

        action_batch.hidden_state = deepcopy(self.hidden_state_bank)
        action_batch.hidden_state.policy_entry = deepcopy(self.hidden_state_bank)
        rollout_batch.policy.hidden_state = deepcopy(self.hidden_state_bank)

        return action_batch, rollout_batch

def marl_reward_calculator(
        rewards: np.ndarray  # with shape (num_episode, agent_num)
) -> np.ndarray:  # with shape (num_episode,)
    avg = []
    for env_idx, reward in enumerate(rewards):
        avg_env_reward = np.mean(reward)
        avg.append(avg_env_reward)
    return np.array(avg)

def make_ppo_policy(
    obs_dim: int,
    act_dim: int,
    lr: float = 3e-4,
    hidden: Sequence = [64],
    recurrent: bool = False,
) -> PPOPolicy:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not recurrent:
        actor_net = NetObs(state_shape=obs_dim, action_shape=act_dim, hidden_sizes=hidden).to(device)
        critic_net = NetObs(state_shape=obs_dim, action_shape=act_dim, hidden_sizes=hidden).to(device)

        actor = MaskedActor(preprocess_net=actor_net, action_dim=act_dim).to(device)
        critic = Critic(preprocess_net=critic_net, device=device)
    else:
        actor_net = RecurrentGRU(
            layer_num=1,
            state_shape=obs_dim,
            action_shape=act_dim,
            device=device,
        )  # GRUBackbone(obs_dim, hidden_dim=[128, 128])
        critic_net = RecurrentGRU(
            layer_num=1,
            state_shape=obs_dim,
            action_shape=act_dim,
            device=device,
        )  # GRUBackbone(obs_dim, hidden_dim=[128, 128])

        actor = MaskedActor(preprocess_net=actor_net, action_dim=act_dim).to(device)
        critic = DictObsCritic(preprocess_net=critic_net).to(device)

    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    # dist = torch.distributions.Categorical  # DISCRETE!

    dist = lambda logits: torch.distributions.Categorical(logits=logits)
    # dist = torch.distributions.Categorical

    return IPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.0,
        eps_clip=0.2,
        value_clip=True,
        action_space=gym.spaces.Discrete(act_dim),
        action_scaling=False,
        advantage_normalization=True,
        reward_normalization=True,
    ).to(device)

def make_vec_env(
        parallel: bool = True,
        independent: bool = True,
        num_envs: int = 4,
        norm: bool = True,
        train: bool = True,
        agents: list['str'] = None,
    ) -> SubprocVectorEnv | DummyVectorEnv | MultiAgentVecNormObs:
    """Each subprocess builds → PettingZooEnv"""
    if parallel:
        env_fns = [
            lambda indep=independent, tr=train:
            get_petting_zoo_env(indep, tr)
            for _ in range(num_envs)
        ]
        env = SubprocVectorEnv(env_fns, context='fork')
    else:
        env_fns = [partial(get_petting_zoo_env, independent, train) for _ in range(1)]
        env = DummyVectorEnv(env_fns)
    if norm:
        env = MultiAgentVecNormObs(
            env,
            agents=agents,
            update_obs_rms=train)
    return env


def get_petting_zoo_env(indep, training):
    env = make_env(independent_learning=indep, training=training)
    env = PettingZooEnv(env)
    return env

def make_env(independent_learning=True, training=True): # type: ignore
    """Return one wrapped PettingZoo environment instance."""
    env = MultiFieldEnv(
        warm_up=0,
        shared_obs=False if independent_learning else True,
        training=training,
        random_budget=training,
    )
    return env

def get_dummy_env():
    return MultiFieldEnv()

def grab_spaces(seed):
    # Inspect one spawned env to grab spaces & agent list
    dummy_env = get_dummy_env()
    dummy_env.reset(seed=seed)
    sample_obs, _, _, _, _ = dummy_env.unwrapped.last()
    first_agent = 'field-1'
    observation_space = dummy_env.sample_observation_space_agent()
    # flat_space, key_order = flatten_dict_space(observation_space)
    obs_dim = observation_space.shape

    # assuming Discrete(.) identical for all
    act_dim = dummy_env.action_spaces[first_agent].n
    agents = dummy_env.possible_agents

    return dummy_env, agents, obs_dim, act_dim

def create_logger(logdir):
    # Logger
    run_name = f"PPO_GRU_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(os.path.join(logdir, run_name))
    logger = TensorboardLogger(writer)
    # make callbacks within this method
    os.makedirs(os.path.join(logdir, run_name, "best"), exist_ok=True)
    os.makedirs(os.path.join(logdir, run_name, "checkpoints"), exist_ok=True)
    return logger, run_name


def collect_test_episodes(collector: Collector, years: list[int] = list(range(2010, 2015))):
    results = []
    for year in years:
        res = collector.collect(
            n_episode=1,
            render=False,
            gym_reset_kwargs={
                "options": {
                    "year": year
                }
            }
        )
        results.append(res)

    return results


# Training methods

def train_gru_ppo(args: Namespace):
    """
    Script to train a GRU-PPO agent
    :param hyperparams:
    :return:
    """

    print(f"\nTraining PPO with {'GRU' if args.recurrent else 'MLP'} Network")
    print(f"Using {'Dummy' if not args.parallel else 'SubProc'}VectorEnv\n")
    print(f"Training with {args.train_envs_num} env(s)\n")

    dummy_env, agents, obs_dim, act_dim = grab_spaces(args.seed)

    # Create vector env
    normalize = True
    train_envs = make_vec_env(
        parallel=args.parallel,
        independent=args.independent,
        num_envs=args.train_envs_num,
        norm=normalize,
        train=True,
        agents=agents,
    )
    test_envs = make_vec_env(
        parallel=args.parallel,
        independent=args.independent,
        num_envs=args.test_envs_num,
        norm=normalize,
        train=False,
        agents=agents,
    )

    if normalize:
        train_envs.reset(options={'year': np.random.choice(range(1951, 2024))})
        test_envs.set_obs_rms(train_envs.get_obs_rms())

    # Build policies
    if args.independent:
        policies = {
            a: make_ppo_policy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                lr=args.lr,
                recurrent=args.recurrent)
            for a in agents
        }
    else:
        shared = make_ppo_policy(obs_dim, act_dim, args.lr, recurrent=args.recurrent)
        policies = {a: shared for a in agents}

    policy_mgr = MultiAgentPolicyManager(policies=list(policies.values()),
                                         env=PettingZooEnv(dummy_env),)

    # Buffers / collectors
    train_collector = IPPOCollector(
        policy=policy_mgr,
        env=train_envs,
        buffer=VectorReplayBuffer(
            total_size=args.buffer_size,
            buffer_num=args.train_envs_num if args.parallel else 1,
            stack_num=1,
        ),  # use this buffer
        exploration_noise=True,
    )
    test_collector = IPPOCollector(
        policy=policy_mgr,
        env=test_envs,
    )

    logger, run_name = create_logger(args.logdir)

    result = OnpolicyTrainer(
        policy=policy_mgr,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect
                         if args.step_per_collect
                         else None,
        episode_per_collect=args.episode_per_collect
                            if not args.step_per_collect
                            else None,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=1,
        batch_size=args.batch_size or ((args.step_per_collect or 64) * len(train_envs)),

        # use lambdas for callbacks
        test_fn=lambda epoch, _: yearly_eval_test_fn(
            epoch,
            dummy_env,
            policy_mgr,
            test_collector,
            train_collector.env,
            agents,
            logger,
            args
        ) if not args.parallel else None,
        save_best_fn=lambda ma_policy: save_best_fn(
            ma_policy,
            train_envs,
            run_name,
            args,
        ),
        save_checkpoint_fn=lambda epoch, env_step, grad_step: save_checkpoint_fn(
            epoch,
            env_step,
            grad_step,
            run_name,
            train_envs,
            test_envs,
            policy_mgr,
            args
        ),
        logger=logger,
        reward_metric=marl_reward_calculator,
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





