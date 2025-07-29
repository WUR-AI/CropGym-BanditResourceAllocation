# tianshou_gru_marl.py

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from cropgymzoo.train import train_gru_ppo
from cropgymzoo import _DEFAULT_LOGDIR
import yaml

import argparse


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=107)

    #--------- Hyperparams
    parser.add_argument("--lr", type=float, default=3e-4)

    # epoch * step_per_epoch = training steps
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step_per_epoch", type=int, default=10_000)


    parser.add_argument("--repeat_per_collect", type=int, default=2)
    # parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--buffer_size", type=int, default=20_000)
    parser.add_argument("--train_envs_num", type=int, default=8)
    parser.add_argument("--test_envs_num", type=int, default=8)
    parser.add_argument("--episode_per_collect", type=int, default=4)
    parser.add_argument("--step_per_collect", type=int, default=0)
    # parser.add_argument("--batch_size", type=int, default=64)  # Not explicitly defining batch size


    # Meta stuff
    parser.add_argument("--logdir", type=str, default=_DEFAULT_LOGDIR)
    parser.add_argument("--not_recurrent", action='store_false', dest='recurrent')
    parser.add_argument("--parallel", action='store_true', dest='parallel')
    parser.add_argument("--not_independent", action="store_false",
                        help="not use independent learning (IPPO)", dest='independent')
    parser.add_argument("--debug", action='store_true', dest='debug')
    parser.set_defaults(
        parallel=False,
        recurrent=True,
        independent=True,
        debug=False,
    )
    hyperparams = parser.parse_args()

    # with open(os.path.join(_CONFIG_PATH, 'ppo_hyperparameters.yaml'), "r") as f:
    #     hyperparams: dict = yaml.load(f, Loader=yaml.SafeLoader)

    train_gru_ppo(hyperparams)
