import os

import argparse

from cropgymzoo.eval_tianshou import run_episodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--model_dir",
                        type=str, default='GRU_PPO')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_icm", action='store_true', dest="use_icm")
    parser.set_defaults(use_icm=False)
    args = parser.parse_args()

    run_episodes(args)
