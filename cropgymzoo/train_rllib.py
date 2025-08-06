import os

import ray
from gymnasium.spaces import Box, Discrete
from networkx.algorithms.approximation.ramsey import ramsey_R2
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.helpers_rllib import get_agent_ids

torch, nn = try_import_torch()

class TorchMaskedActions(PPOTorchPolicy):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space: Box,
        action_space: Discrete,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        PPOTorchPolicy.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

if __name__ == "__main__":
    ray.init()

    alg_name = "PPO"
    ModelCatalog.register_custom_model("masked_rnn", TorchMaskedActions)
    # function that outputs the environment you wish to register.

    def env_creator():
        env = MultiFieldEnv(
            training=True,
            warm_up=0,
            random_budget=True,
        )
        return env

    env_name = "cropgymzoo-train"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    config = (
        PPOConfig()
        .environment(env=env_name)
        .env_runners(num_env_runners=1)
        .training(
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            # model={"custom_model": "masked_rnn"},
        )
        .multi_agent(
            policies={
                agent_id: (None, obs_space, act_space, dict()) for agent_id in get_agent_ids()
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
        .framework(framework="torch")
        .callbacks(
            on_episode_end=(
                lambda episode, **kw: print(f"Episode done. R={episode.get_return()}")
            )
        )
    )

    tune.run(
        alg_name,
        name="PPO",
        stop={"timesteps_total": 10000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )