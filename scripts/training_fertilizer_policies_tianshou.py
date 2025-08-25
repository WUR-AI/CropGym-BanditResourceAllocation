# tianshou_gru_marl.py

import warnings

from cropgymzoo.utils import curriculum

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from cropgymzoo.train_tianshou import train_gru_ppo
from cropgymzoo import _DEFAULT_LOGDIR
import yaml

import argparse


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=107)

    #--------- Hyperparams
    parser.add_argument("--lr", type=float, default=1e-4)

    # epoch * step_per_epoch = training steps
    parser.add_argument("--epoch", type=int, default=5_000)
    parser.add_argument("--step_per_epoch", type=int, default=1_000)


    parser.add_argument("--repeat_per_collect", type=int, default=2)
    # parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--buffer_size", type=int, default=20_000)
    parser.add_argument("--train_envs_num", type=int, default=8)
    parser.add_argument("--test_envs_num", type=int, default=1)
    parser.add_argument("--episode_per_collect", type=int, default=6)
    parser.add_argument("--step_per_collect", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)  # Not explicitly defining batch size


    # Meta stuff
    parser.add_argument("--logdir", type=str, default=_DEFAULT_LOGDIR)
    parser.add_argument("--not_recurrent", action='store_false', dest='recurrent')
    parser.add_argument("--parallel", action='store_true', dest='parallel')
    parser.add_argument("--not_independent", action="store_false",
                        help="not use independent learning (IPPO)", dest='independent')
    parser.add_argument("--debug", action='store_true', dest='debug')

    # algorithm hyperparameter stuff
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--hidden_layers", type=int, nargs="*",
                        help="Input space separated layer sizes", default=[64, 64])

    # Use curriculum learning?
    parser.add_argument("--curriculum", action='store_true', dest='curriculum')

    # Use intrinsic curiosity module
    parser.add_argument("--use_icm", action='store_true', dest='use_icm')

    # resume model
    parser.add_argument("--resume", action="store_true", dest='resume')
    parser.add_argument("--model_dir", type=str, default='GRU_PPO')

    # use lagrangian
    parser.add_argument("--lagrangian_ppo", action="store_true", dest='lagrangian_ppo')

    parser.set_defaults(
        resume=None,
        parallel=False,
        recurrent=True,
        independent=True,
        debug=False,
        use_icm=False,
        lagrangian_ppo=False,
    )
    hyperparams = parser.parse_args()

    # safeguard
    hyperparams.train_envs_num = 2 if (hyperparams.parallel is True and hyperparams.train_envs_num == 1) else hyperparams.train_envs_num

    train_gru_ppo(hyperparams)
