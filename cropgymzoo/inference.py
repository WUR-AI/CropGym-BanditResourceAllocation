import os
from pathlib import Path
import argparse
import torch

import pickle

from cropgymzoo import _DEFAULT_MODEL_DIR
from cropgymzoo.train import make_vec_env, grab_spaces, make_ppo_policy


def load_model(args: argparse.Namespace) -> pickle:

    model_dir = Path(str(os.path.join(_DEFAULT_MODEL_DIR, args.model_dir)))
    assert model_dir.is_dir(), f"The path {str(model_dir)} is not a valid directory!"

    checkpoint = None
    for entry in model_dir.iterdir():
        checkpoint = torch.load(entry, weights_only=False) if str(entry).endswith(".pth") else None
        if checkpoint is not None:
            break
    else:
        print(f"Load {checkpoint}!")

    return checkpoint


def continue_training():
    ...


def run_inference(model) -> None:

    env = make_vec_env(
        False,
        True,
        1,
        True,
        False,
    )

    dummy_env, agents, obs_dim, act_dim = grab_spaces(107)

    policies = {a: make_ppo_policy(obs_dim, act_dim, recurrent=True) for a in agents}

