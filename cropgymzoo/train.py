import os
from functools import partial
import datetime
from typing import Sequence

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import numpy as np
import gymnasium as gym

from cropgymzoo.agents.networks import RecurrentGRU, MaskedActor, DictObsCritic, NetObs
from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo import _DEFAULT_LOGDIR

from pettingzoo.utils.conversions import parallel_to_aec
from pettingzoo import ParallelEnv

from cropgymzoo.utils.wrappers import VecNormObs

from cropgymzoo.utils.agent_helpers import extract_info

try:
    # ---- Tianshou imports ----
    from tianshou.data import Collector, VectorReplayBuffer, Batch
    from tianshou.data.collector import EpisodeRolloutHookProtocol
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


class CountStuffHook:  # (EpisodeRolloutHookProtocol):
    def __init__(self, agent_names):
        # cache the mapping once
        self.aid2idx = {aid: i for i, aid in enumerate(agent_names)}
    """Compute per-agent episode stats and return them as numpy arrays."""
    def __call__(self, episode_batch):
        # episode_batch.info is a list/array of `info` dicts for *every* step
        t = len(episode_batch)
        agent_ids = np.asarray(episode_batch.obs.agent_id)  # (T,) strings
        infos = episode_batch.info  # (T,) dicts
        agents = np.unique(agent_ids)
        out: dict[str, np.ndarray] = {} # {agent: metric value}
        for aid, idx in self.aid2idx.items():
            # out.setdefault(aid, {})
            mask = agent_ids == aid  # which timesteps belong to this agent
            sub = infos[mask]  # same length as #steps for that agent
            rew_tot = episode_batch.rew[:, idx].sum()

            out[f"Naction/{aid}"] = np.full(t, float(sub.Naction[-1]), dtype=np.float32)
            out[f"Yield/{aid}"] = np.full(t, float(sub.Yield[-1]), dtype=np.float32)
            out[f"Reward/{aid}"] = np.full(t, float(rew_tot), dtype=np.float32)
            out[f"Nue/{aid}"] = np.full(t, float(sub.Nue[-1]), dtype=np.float32)
            out[f"Nsurp/{aid}"] = np.full(t, float(sub.Nsurp[-1]), dtype=np.float32)

        return out

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
        actor_net = RecurrentGRU(layer_num=1, state_shape=obs_dim, action_shape=act_dim, device=device, )  # GRUBackbone(obs_dim, hidden_dim=[128, 128])
        critic_net = RecurrentGRU(layer_num=1, state_shape=obs_dim, action_shape=act_dim, device=device,)  # GRUBackbone(obs_dim, hidden_dim=[128, 128])

        actor = MaskedActor(preprocess_net=actor_net, action_dim=act_dim).to(device)
        critic = DictObsCritic(preprocess_net=critic_net).to(device)

    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    # dist = torch.distributions.Categorical  # DISCRETE!

    dist = lambda logits: torch.distributions.Categorical(logits=logits)
    # dist = torch.distributions.Categorical

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

def make_vec_env(
        parallel: bool = True,
        indep: bool = True,
        num_envs: int = 4,
        norm: bool = True,
        train: bool = True
    ) -> SubprocVectorEnv | DummyVectorEnv | VecNormObs:
    """Each subprocess builds → PettingZooEnv"""
    if parallel:
        env_fns = [partial(get_petting_zoo_env, indep, train) for _ in range(num_envs)]
        env = SubprocVectorEnv(env_fns)
    else:
        env_fns = [partial(get_petting_zoo_env, indep, train) for _ in range(1)]
        env = DummyVectorEnv(env_fns)
    if norm:
        env = VecNormObs(env, update_obs_rms=train)
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

def train_gru_ppo(hyperparams: dict):
    """
    Script to train a GRU-PPO agent
    :param hyperparams:
    :return:
    """

    # extract dict
    indep = hyperparams.get('independent', True)
    train_envs_num = hyperparams.get('train_envs', 1)
    test_envs_num = hyperparams.get('test_envs', 1)
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
    recurrent = hyperparams.get('recurrent', True)
    debug = hyperparams.get('debug', False)

    dummy_env, agents, obs_dim, act_dim = grab_spaces(seed)

    # Create vector env
    normalize = True
    train_envs = make_vec_env(parallel, indep, train_envs_num, norm=normalize, train=True)
    test_envs = make_vec_env(parallel, indep, test_envs_num, norm=normalize, train=False)

    if normalize:
        train_envs.reset(options={'year': np.random.choice(range(1951, 2024))})
        test_envs.set_obs_rms(train_envs.get_obs_rms())

    # Build policies
    if indep:
        policies = {a: make_ppo_policy(obs_dim, act_dim, lr, recurrent=recurrent) for a in agents}
    else:
        shared = make_ppo_policy(obs_dim, act_dim, lr, recurrent=recurrent)
        policies = {a: shared for a in agents}

    policy_mgr = MultiAgentPolicyManager(policies=list(policies.values()),
                                         env=PettingZooEnv(dummy_env),)

    # Buffers / collectors
    train_collector = Collector(
        policy=policy_mgr,
        env=train_envs,
        buffer=VectorReplayBuffer(
            total_size=buffer_size,
            buffer_num=1,
            stack_num=1,
        ),  # use this buffer
        exploration_noise=True
    )
    test_collector = Collector(
        policy=policy_mgr,
        env=test_envs,
    )

    logger, run_name = create_logger(logdir)

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

    def yearly_eval_test_fn(epoch, global_step):
        test_results = {}
        year_rewards = []

        reset_options_list = [
            {'year': year} for year in range(2010, 2011)
        ]

        dfs = []
        writer = logger.writer
        # per year eval
        for i, reset_opts in enumerate(reset_options_list):
            year = reset_opts["year"]

            # Collect test episode(s)
            result = test_collector.collect(
                n_episode=1,
                reset_before_collect=True,
                gym_reset_kwargs={
                    'options': reset_opts
                },
            )

            infos = test_collector.buffer._meta.info
            obs = test_collector.buffer._meta.obs
            obs_next = test_collector.buffer._meta.obs_next
            rew = test_collector.buffer._meta.rew

            agent_ids = obs_next["agent_id"]

            agent_dict = {}

            if debug:
                df = pd.DataFrame(index=agent_ids,
                                  data={
                                      'nue': infos["Nue"],
                                      'Nsurp': infos["Nsurp"],
                                      'BudgetLeft': infos["BudgetLeft"],
                                      'action': infos["Action"],
                                      'Yield': infos["Yield"]}
                                  )

                dfs.append(df)

            for a, a_id in enumerate(agents):
                agent = agent_ids == a_id

                reward = [r[a] for r in rew[agent]]
                nue = infos["Nue"][agent]
                nsurp = infos["Nsurp"][agent]
                budget_left = infos["BudgetLeft"][agent]
                yld = infos["Yield"][agent]
                n_action = infos["Action"][agent]

                agent_reward = np.sum(reward)
                agent_nue = nue[-1]
                agent_nsurp = nsurp[-1]
                agent_budget_left = budget_left[-1]
                agent_yield = yld[-1]
                agent_n_action = np.sum(n_action)

                agent_dict[a_id] = {
                    "Reward": agent_reward,
                    "Nue": agent_nue,
                    "Nsurp": agent_nsurp,
                    "BudgetLeft": agent_budget_left,
                    "Yield": agent_yield,
                    "Naction": agent_n_action,
                }

                if writer:
                    writer.add_scalar(f"test/{year}/{a_id}/reward", agent_reward, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/NUE", agent_nue, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/Nsurp", agent_nsurp, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/BudgetLeft", agent_budget_left, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/Yield", agent_yield, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/Naction", agent_n_action, epoch)

            # Store results with metadata
            test_results[year] = agent_dict
            year_reward = np.sum([v
                                  for y in test_results.values()
                                  for field in y.values()
                                  for key, v in field.items()
                                  if key == "Reward"])
            year_rewards.append(year_reward)

            # Logging intermediate results
            if writer:
                writer.add_scalar(f"test/{year}/reward", year_reward, epoch)

        # Final aggregated logging
        mean_reward = np.mean(year_rewards)
        #
        if writer:
            writer.add_scalar("test/mean_reward_all_years", mean_reward, epoch)

        writer.flush()


    result = OnpolicyTrainer(
        policy=policy_mgr,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        # step_per_collect=step_per_collect,
        episode_per_collect=episode_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=1,
        batch_size=step_per_collect * len(train_envs),
        test_fn=yearly_eval_test_fn,
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





