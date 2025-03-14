import gymnasium as gym
from pcse_zoo.envs.singular_env import ParcelEnv

def register_predefined_envs() -> None:
    gym.envs.register(
        id='Parcel-1',
        entry_point=ParcelEnv,

    )

def register_eval_envs():
    ...

def register_test_envs():
    ...