# tianshou_gru_marl.py
"""
Example training script for a PettingZoo parallel MARL environment using Tianshou with
recurrent (GRU‑based) independent PPO policies (one per agent by default).

⚠️  Replace the placeholder logic (marked TODO) with concrete code that
converts your dict observations to flat numpy arrays and handles masks the
way your environment provides them.  The rest of the pipeline is ready‑made.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from cropgymzoo.train import train_gru_ppo
from cropgymzoo import _CONFIG_PATH
import yaml


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--independent", action="store_true", help="independent learning (IPPO)")
    # parser.add_argument("--lr", type=float, default=3e-4)
    # parser.add_argument("--epoch", type=int, default=500)
    # parser.add_argument("--step-per-epoch", type=int, default=10000)
    # parser.add_argument("--collect-per-step", type=int, default=20000)
    # parser.add_argument("--repeat-per-collect", type=int, default=5)
    # parser.add_argument("--batch-size", type=int, default=3200)
    # parser.add_argument("--buffer-size", type=int, default=20000)
    # parser.add_argument("--train-num", type=int, default=8)
    # parser.add_argument("--test-num", type=int, default=8)
    # parser.add_argument("--logdir", type=str, default="./log")
    # args = parser.parse_args()

    with open(os.path.join(_CONFIG_PATH, 'ppo_hyperparameters.yaml'), "r") as f:
        hyperparams: dict = yaml.load(f, Loader=yaml.SafeLoader)

    train_gru_ppo(hyperparams)
