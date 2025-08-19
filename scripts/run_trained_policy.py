import os

import argparse

from cropgymzoo import _DEFAULT_MODEL_DIR
from cropgymzoo.eval_tianshou import load_model, run_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--model_dir",
                        type=str, default='GRU_PPO')

    args = parser.parse_args()

    model = load_model(args)

    print(model)

    run_inference(model)
