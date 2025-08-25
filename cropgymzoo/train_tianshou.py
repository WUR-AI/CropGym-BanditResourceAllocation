import os
from functools import partial
import datetime
import itertools
from typing import Sequence
from argparse import Namespace
import pickle

# for comet_ml
from cropgymzoo.utils.callbacks_tianshou import (
    marl_save_checkpoint_fn,
    save_best_fn,
    create_comet_experiment,
    CometTianshouLogger,
    MultiLogger,
    run_test_callback
)

from pathlib import Path

import numpy as np
import gymnasium as gym

from cropgymzoo import _DEFAULT_MODEL_DIR
from cropgymzoo.agents.networks_tianshou import (
    RecurrentGRU,
    MaskedActor,
    IntrinsicCuriosityModuleMARL,
    ConstraintCritic,
    ObsMLP
)
from cropgymzoo.agents.marl_algorithms_tianshou import IPPOPolicy, IPPOCollector, LagrangianIPPOPolicy
from cropgymzoo.envs.multi_field_env import MultiFieldEnv

from cropgymzoo.envs.wrappers_tianshou import MultiAgentVecNormObs
from cropgymzoo.utils.curriculum import CurriculumCallbackManager


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

try:
    # ---- Tianshou imports ----
    from tianshou.data import Collector, VectorReplayBuffer, Batch, ReplayBuffer
    from tianshou.data.collector import EpisodeRolloutHookProtocol, StepHook
    from tianshou.env import PettingZooEnv, DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs, BaseVectorEnv, VectorEnvWrapper, ShmemVectorEnv
    from tianshou.utils.net.common import NetBase, RecurrentStateBatch, Net, MLP
    from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule  # will wrap our GRU core
    from tianshou.utils.net.common import Recurrent
    from tianshou.utils.logger.tensorboard import TensorboardLogger
    from tianshou.utils import tqdm_config
    from tianshou.policy import PPOPolicy, MultiAgentPolicyManager, ICMPolicy
    from tianshou.trainer import OnpolicyTrainer
    from tianshou.utils.statistics import RunningMeanStd
except ImportError:
    tianshou = None


def load_model(args: Namespace) -> dict:

    if not hasattr(args, 'model_dir'):
        args.model_dir = 'GRU_PPO'

    model_dir = Path(str(os.path.join(_DEFAULT_MODEL_DIR, args.model_dir)))

    print('Loading model from {}'.format(model_dir))
    assert model_dir.is_dir(), f"The path {str(model_dir)} is not a valid directory!"

    checkpoint = None
    for entry in model_dir.iterdir():
        checkpoint = torch.load(entry, weights_only=False) if str(entry).endswith(".pth") else None
        if checkpoint is not None:
            break
    else:
        print(f"Loaded {checkpoint}!")

    return checkpoint


def marl_reward_calculator(
        rewards: np.ndarray  # with shape (num_episode, agent_num)
) -> np.ndarray:  # with shape (num_episode,)
    avg = []
    for env_idx, reward in enumerate(rewards):
        avg_env_reward = np.mean(reward)
        avg.append(avg_env_reward)
    return np.array(avg)

def make_ppo_policy(
    obs_dim: int | tuple[int],
    act_dim: int,
    hidden: Sequence = [64, 64],
    use_icm: bool = False,
    args: Namespace = None,
    mlp_critics = True,
) -> PPOPolicy | ICMPolicy:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor_net = RecurrentGRU(
        layer_num=len(hidden),
        hidden_layer_size=hidden[0],
        state_shape=obs_dim,
        action_shape=act_dim,
        device=device,
    )

    if args.lagrangian_ppo:
        obs_constraint_dim = args.obs_constraint_dim
        obs_constraint_idx = args.constraint_indices

    if mlp_critics:
        constraint_net = ObsMLP(
            input_dim=obs_constraint_dim,
            hidden_sizes=hidden,
            activation=torch.nn.Tanh,
            device=device,
        ) if args.lagrangian_ppo else None
        critic_net = ObsMLP(
            input_dim=obs_dim[0],
            hidden_sizes=hidden,
            activation=torch.nn.Tanh,
            device=device,
        )
    else:
        constraint_net = RecurrentGRU(
            layer_num=len(hidden),
            hidden_layer_size=hidden[0],
            state_shape=obs_constraint_dim,
            action_shape=act_dim,
            device=device,
        ) if args.lagrangian_ppo else None
        critic_net = RecurrentGRU(
            layer_num=len(hidden),
            hidden_layer_size=hidden[0],
            state_shape=obs_dim,
            action_shape=act_dim,
            device=device,
        )


    actor = MaskedActor(preprocess_net=actor_net, action_dim=act_dim).to(device)
    critic = Critic(preprocess_net=critic_net).to(device)
    constraint_critic = ConstraintCritic(
        preprocess_net=constraint_net,
        constraint_indices=obs_constraint_idx,
    ) if args.lagrangian_ppo else None

    optim = Adam(
        list(actor.parameters()) + list(critic.parameters()) + list(constraint_critic.parameters()),
        lr=args.lr if args is not None else 1e-3,
        ) if args.lagrangian_ppo else (
        Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=args.lr if args is not None else 1e-3,
        )
    )
    # dist = torch.distributions.Categorical  # DISCRETE!

    dist = lambda logits: torch.distributions.Categorical(logits=logits)
    # dist = torch.distributions.Categorical

    policy_fn = IPPOPolicy if not args.lagrangian_ppo else LagrangianIPPOPolicy

    lagrangian_kwarg = {'constraint_critic': constraint_critic}

    ppo_policy = policy_fn(
        actor=actor,
        critic=critic,
        **lagrangian_kwarg if args.lagrangian_ppo else None,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma if args is not None else 0.99,
        gae_lambda=args.gae_lambda if args is not None else 0.95,
        max_grad_norm=0.5,
        vf_coef=args.vf_coef if args is not None else 0.5,
        ent_coef=args.ent_coef if args is not None else 0.01,
        eps_clip=0.2,
        value_clip=True,
        action_space=gym.spaces.Discrete(act_dim),
        action_scaling=False,
        advantage_normalization=True,
        reward_normalization=True,
    ).to(device)

    if not use_icm:
        return ppo_policy


    feature_net = MLP(
        input_dim=obs_dim[0],
        hidden_sizes=[128],
        output_dim=64,
    ).to(device)

    icm = IntrinsicCuriosityModuleMARL(
        feature_net=feature_net,
        action_dim=act_dim,
        feature_dim=64,
        hidden_sizes=[128],
    )

    icm_optim = torch.optim.Adam(
        icm.parameters(),
        lr=args.lr if args is not None else 1e-4,
    )

    icm_policy = ICMPolicy(
        policy=ppo_policy,
        model=icm,
        optim=icm_optim,
        lr_scale=1.0,
        reward_scale=0.5,
        forward_loss_weight=0.5,
        action_space=gym.spaces.Discrete(act_dim),
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim),
    )
    return icm_policy

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
        # env = ShmemVectorEnv(env_fns)
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
        training=False, # training,
        random_budget=False,
    )
    return env

def get_dummy_env():
    env = MultiFieldEnv()
    return env

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

def create_logger(args):
    logdir = args.logdir
    # Logger
    run_name = f"PPO_GRU_{'parallel' if args.parallel else 'dummy'}_{datetime.datetime.now():%d_%H%M}"
    run_name = (run_name + "_resume") if args.resume else run_name
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

    # resume model?
    saved_model = None
    if getattr(args, "resume", None) is not None:
        saved_model = load_model(args)
        # override args to match
        args = saved_model['args']
        args.resume = True

        print("Resuming training from saved model!")

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

    # For constraint critic
    obs_features = dummy_env.get_field_env_with_idx(0).unwrapped.obs_constraint_features()
    args.obs_constraint_dim = len(obs_features)
    args.constraint_indices = dummy_env.get_field_env_with_idx(0).unwrapped.get_idx_features(obs_features)

    # Build policies
    # if args.independent:
    policies = {
        a: make_ppo_policy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=args.hidden_layers,
            use_icm=args.use_icm,
            args=args,
        )
        for a in agents
    }

    print(f"Using {'LagrangianIPPO' if args.lagrangian_ppo else 'IPPO'} policy!")
    print(f"Using ICM Policy!") if args.use_icm else None

    marl_policy_manager = MultiAgentPolicyManager(
        policies=list(policies.values()),
        env=PettingZooEnv(dummy_env),
    )

    if args.resume is not None:
        for agent, policy in marl_policy_manager.policies.items():
            policy.load_state_dict(saved_model['models'][agent], strict=True)

        train_envs.set_obs_rms(saved_model['obs_rms'])
        test_envs.set_obs_rms(saved_model['obs_rms'])

    # Buffers / collectors
    train_collector = IPPOCollector(
        policy=marl_policy_manager,
        env=train_envs,
        buffer=VectorReplayBuffer(
            total_size=args.buffer_size,
            buffer_num=args.train_envs_num if args.parallel else 1,
            stack_num=1,
        ),  # use this buffer
        exploration_noise=True,
    )
    test_collector = IPPOCollector(
        policy=marl_policy_manager,
        env=test_envs,
    )

    # get tensorboard logger
    tb_logger, run_name = create_logger(args)

    # get comet logger
    comet_experiment = create_comet_experiment(run_name, args)

    comet_logger = CometTianshouLogger(
        experiment=comet_experiment,
        log_dir=args.logdir,
    ) if comet_experiment is not None else None

    # put both in the multi-logger item
    logger = MultiLogger(
        tb_logger,
        comet_logger if comet_logger else None
    )

    curriculum_manager = None
    if args.curriculum:
        curriculum_manager = CurriculumCallbackManager(
            # sample from sub-env attribute... Quite dirty this way.
            max_stage=dummy_env.get_field_env_with_idx(0).unwrapped.random_manager.get_max_stage()
        )
    print(f'Curriculum learning activated!')

    # make trainer
    result = OnpolicyTrainer(
        policy=marl_policy_manager,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect
                         if args.step_per_collect
                         else None,
        episode_per_collect=args.episode_per_collect
                            if args.episode_per_collect > args.train_envs_num
                            else args.train_envs_num
                            if not args.step_per_collect
                            else None,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_envs_num if args.parallel else 1,
        batch_size=args.batch_size or ((args.step_per_collect or 64) * len(train_envs)),

        # use lambdas for callbacks
        test_fn=lambda epoch, env_step: run_test_callback(
            epoch,
            env_step,
            dummy_env,
            marl_policy_manager,
            train_collector.env,
            agents,
            logger,
            curriculum_manager,
            args
        ),
        save_best_fn=lambda ma_policy: save_best_fn(
            ma_policy,
            train_envs,
            run_name,
            args,
        ),
        save_checkpoint_fn=lambda epoch, env_step, grad_step: marl_save_checkpoint_fn(
            epoch,
            env_step,
            grad_step,
            run_name,
            train_envs,
            test_envs,
            marl_policy_manager,
            args,
            experiment=comet_experiment,
        ),
        logger=logger,
        reward_metric=marl_reward_calculator,
    ).run()
    print(f"Training done → best avg reward: {result.best_reward:.3f}")

'''
NOTE FOR WHEN RESUMING MODEL

ckpt = torch.load("checkpoints/best.pth", map_location="cpu")

policy.load_state_dict(ckpt["model"])
policy.optim.load_state_dict(ckpt["optimizer"])        # if you saved it

train_envs.set_obs_rms(ckpt["obs_rms"])                # keep collecting
test_envs.set_obs_rms(ckpt["obs_rms"])                 # deterministic eval
'''





