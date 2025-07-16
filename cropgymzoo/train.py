import os
from functools import partial
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import numpy as np
import gymnasium as gym

from cropgymzoo.agents.networks import RecurrentGRU, MaskedActor, DictObsCritic
from cropgymzoo.envs.worker_env import ParallelRLWorkers
from cropgymzoo import _DEFAULT_LOGDIR

from pettingzoo.utils.conversions import parallel_to_aec
from pettingzoo import ParallelEnv

from cropgymzoo.utils.wrappers import VecNormObs

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


# ------------------------------------------------------------------ #
# 2)  SAMPLE (dict)  →  np.ndarray
# ------------------------------------------------------------------ #


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





