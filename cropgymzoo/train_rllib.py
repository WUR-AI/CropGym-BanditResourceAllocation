import os

import tree

from collections import defaultdict



from gymnasium.spaces import Box
import numpy as np

import ray

import supersuit as ss

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import (
    ActorCriticEncoderConfig, MLPEncoderConfig, RecurrentEncoderConfig
)
from ray.rllib.core.models.base import (
    Encoder,
    ActorCriticEncoder,
    StatefulActorCriticEncoder,
    ENCODER_OUT,
)
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec, MultiRLModule
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.models.utils import get_initializer_fn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env
from ray.rllib.evaluation.observation_function import ObservationFunction

from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.helpers_rllib import get_agent_ids

torch, nn = try_import_torch()

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.columns import Columns as C


class LSTMModule(TorchRLModule, ValueFunctionAPI):
    """An example TorchRLModule that contains an LSTM layer.

    .. testcode::

        import numpy as np
        import gymnasium as gym

        B = 10  # batch size
        T = 5  # seq len
        e = 25  # embedding dim
        CELL = 32  # LSTM cell size

        # Construct the RLModule.
        my_net = LSTMContainingRLModule(
            observation_space=gym.spaces.Box(-1.0, 1.0, (e,), np.float32),
            action_space=gym.spaces.Discrete(4),
            model_config={"lstm_cell_size": CELL}
        )

        # Create some dummy input.
        obs = torch.from_numpy(
            np.random.random_sample(size=(B, T, e)
        ).astype(np.float32))
        state_in = my_net.get_initial_state()
        # Repeat state_in across batch.
        state_in = tree.map_structure(
            lambda s: torch.from_numpy(s).unsqueeze(0).repeat(B, 1), state_in
        )
        input_dict = {
            Columns.OBS: obs,
            Columns.STATE_IN: state_in,
        }

        # Run through all 3 forward passes.
        print(my_net.forward_inference(input_dict))
        print(my_net.forward_exploration(input_dict))
        print(my_net.forward_train(input_dict))

        # Print out the number of parameters.
        num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
        print(f"num params = {num_all_params}")
    """

    def setup(self):
        """Use this method to create all the model components that you require.

        Feel free to access the following useful properties in this class:
        - `self.model_config`: The config dict for this RLModule class,
        which should contain flxeible settings, for example: {"hiddens": [256, 256]}.
        - `self.observation|action_space`: The observation and action space that
        this RLModule is subject to. Note that the observation space might not be the
        exact space from your env, but that it might have already gone through
        preprocessing through a connector pipeline (for example, flattening,
        frame-stacking, mean/std-filtering, etc..).
        """
        # Assume a simple Box(1D) tensor as input shape.
        in_size = self.observation_space.shape[0]

        # Get the LSTM cell size from the `model_config` attribute:
        self._lstm_cell_size = self.model_config.get("lstm_cell_size", 256)
        self._lstm = nn.LSTM(in_size, self._lstm_cell_size, batch_first=True)
        in_size = self._lstm_cell_size

        # Build a sequential stack.
        layers = []
        # Get the dense layer pre-stack configuration from the same config dict.
        dense_layers = self.model_config.get("dense_layers", [128, 128])
        for out_size in dense_layers:
            # Dense layer.
            layers.append(nn.Linear(in_size, out_size))
            # ReLU activation.
            layers.append(nn.ReLU())
            in_size = out_size

        self._fc_net = nn.Sequential(*layers)

        # Logits layer (no bias, no activation).
        self._pi_head = nn.Linear(in_size, self.action_space.n)
        # Single-node value layer.
        self._values = nn.Linear(in_size, 1)

    def get_initial_state(self):
        return {
            "h": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32),
            "c": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32),
        }

    def _forward(self, batch, **kwargs):
        # Compute the basic 1D embedding tensor (inputs to policy- and value-heads).
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)

        # Return logits as ACTION_DIST_INPUTS (categorical distribution).
        # Note that the default `GetActions` connector piece (in the EnvRunner) will
        # take care of argmax-"sampling" from the logits to yield the inference (greedy)
        # action.
        return {
            C.ACTION_DIST_INPUTS: logits,
            C.STATE_OUT: state_outs,
        }

    def _forward_train(self, batch, **kwargs):
        # Same logic as _forward, but also return embeddings to be used by value
        # function branch during training.
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)
        return {
            C.ACTION_DIST_INPUTS: logits,
            C.STATE_OUT: state_outs,
            C.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    def compute_values(
        self, batch: dict, embeddings = None
    ):
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_state_outs(batch)
        values = self._values(embeddings).squeeze(-1)
        return values

    def _compute_embeddings_and_state_outs(self, batch):
        obs = batch[C.OBS]["observation"]
        act_mask = batch[C.OBS]["action_mask"]
        state_in = batch[C.STATE_IN]
        h, c = state_in["h"], state_in["c"]
        # Unsqueeze the layer dim (we only have 1 LSTM layer).
        embeddings, (h, c) = self._lstm(obs, (h.unsqueeze(0), c.unsqueeze(0)))
        # Push through our FC net.
        embeddings = self._fc_net(embeddings)
        # Squeeze the layer dim (we only have 1 LSTM layer).
        return embeddings, {"h": h.squeeze(0), "c": c.squeeze(0)}


class MLPMaskedPPOModule(TorchRLModule, ValueFunctionAPI):
    """
    PPO-compatible, feed-forward (non-recurrent) module with action masking.

    Expects in batch[C.OBS] a dict containing:
      • "observation": (B, feat) or (B, T, feat)
      • "action_mask": (B, N)   or (B, T, N)

    Emits:
      • C.ACTION_DIST_INPUTS: (B, 1, N)
      • C.VF_PREDS:           (B, 1)
      • C.STATE_OUT:          {}
    """

    def __init__(self, *args, hidden_sizes=(128, 128), activation=nn.Tanh, **kwargs):
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        # ---- anatomy of the spaces ----
        vec_space = self.observation_space["observation"]
        self.n_act = self.action_space.n
        in_dim = int(vec_space.shape[0])

        # ---- MLP backbone & heads ----
        layers = []
        last = in_dim
        for hs in self._hidden_sizes:
            layers += [nn.Linear(last, hs), self._activation()]
            last = hs
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        self.pi = nn.Linear(last, self.n_act)
        self.vf = nn.Linear(last, 1)

        # RLlib uses this to build the distribution from the logits
        self.action_dist_cls = TorchCategorical


    def get_initial_state(self, batch_size: int = 1):
        return {}

    def _forward_masked(self, batch, **kwargs):
        obs = batch[C.OBS]
        x = obs["observation"].float()
        mask = obs["action_mask"].bool()

        # If a time axis exists, use the last step; otherwise keep (B, F)
        if x.ndim == 3:   # (B, T, F)
            x = x[:, -1, :]

        # Ensure mask is (B, N)
        if mask.ndim == 3:  # (B, T, N)
            mask = mask[:, -1, :]
        elif mask.ndim == 1:  # (N,) -> (1, N)
            mask = mask.unsqueeze(0)

        z = self.backbone(x)
        logits = self.pi(z)             # (B, N)
        # v = self.vf(z).squeeze(-1)      # (B,)

        # Safety: ensure at least one valid action per row
        none_valid = ~mask.any(dim=1)
        if none_valid.any():
            # fall back to "all valid" for those rows
            mask = mask.clone()
            mask[none_valid] = True

        logits = logits.masked_fill(~mask, float("-inf"))

        v = self.vf(z).view(-1, 1)  # (B, 1)  <- important

        self._last_vf_preds = v

        return {
            C.ACTION_DIST_INPUTS: logits,  # (B, N)
            C.VF_PREDS: v,                 # (B,)
            # Note: DO NOT include C.STATE_OUT for MLP
        }

    def value_function(self):
        """Required method for PPO to access value function predictions."""
        return self._last_vf_preds

    _forward_train       = _forward_masked
    _forward_exploration = _forward_masked
    _forward_inference   = _forward_masked


class GRUMaskedPPOModule(TorchRLModule):
    """
    PPO-compatible RLModule with:
      • a single-layer GRU encoder (batch_first=True)
      • invalid-action masking by setting logits = -inf
    Works for both single- and multi-agent rollouts.
    """

    def setup(self) -> None:
        # ---- anatomy of the spaces ----
        vec_space   = self.observation_space['observation']
        self.n_act  = self.action_space.n
        self.h_dim  = 128                       # hidden size of the GRU
        self.num_layers = 1

        # ---- encoder & heads ----
        self.gru   = nn.GRU(
            input_size=vec_space.shape[0],
            hidden_size=self.h_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.pi    = nn.Linear(self.h_dim, self.n_act)
        self.vf    = nn.Linear(self.h_dim, 1)

        # RLlib uses this to build the distribution from the logits
        self.action_dist_cls = TorchCategorical

    # --------------------------------------------------------------------- #
    #  Required helpers for recurrent modules
    # --------------------------------------------------------------------- #

    def get_initial_state(self, batch_size: int = 1):
        # shape: (num_layers, B, hidden_dim) – here (1, B, h_dim)
        return {"h": torch.zeros(self.num_layers, self.h_dim)}

    # --------------------------------------------------------------------- #
    #  Three forward passes RLlib will call
    # --------------------------------------------------------------------- #
    def _forward_masked(self, batch, **kwargs):
        """
        Handles all phases (train / exploration / inference).
        Input shapes:
            observations:  (B, T, feat) or (B, feat)
            action_mask:   (B, T, N)    or (B, N)
            state_in[0]:   (1, B, h_dim)
        """
        obs_dict = batch[C.OBS]

        x    = obs_dict["observation"].float()
        mask = obs_dict["action_mask"].bool()

        # Ensure a time dimension for one-step batches  ->  (B, 1, feat)
        if x.ndim == 2:
            x    = x.unsqueeze(1)

        if mask.ndim == 1:  # (n_act,) -> (1, n_act)
            mask = mask.unsqueeze(0)
        if mask.ndim == 2:  # (B, n_act) -> (B, 1, n_act)
            mask = mask.unsqueeze(1)

        B, T, _ = x.shape
        state_list = batch.get(C.STATE_IN)

        # ---- robust state reshape ----
        if state_list:
            h_in = state_list["h"] if isinstance(state_list, dict) else state_list[0]
            if h_in.dim() == 4:  # (B, T, L, H)
                h_in = h_in[:, 0].permute(1, 0, 2)  # -> (L, B, H)
            elif h_in.dim() == 3:
                if h_in.shape[0] == x.shape[0]:  # (B, L, H)
                    h_in = h_in.permute(1, 0, 2)  # -> (L, B, H)
                # else already (L, B, H)
            elif h_in.dim() == 2:  # (L, H)
                h_in = h_in.unsqueeze(1)  # -> (L, 1, H) for GRU
        else:
            h_in = torch.zeros(self.num_layers, B, self.h_dim,
                               device=x.device, dtype=x.dtype)

        y, h_out = self.gru(x, h_in)           # y: (B, T, h_dim)

        # Take the last timestep’s hidden state
        last = y[:, -1, :]                     # (B, h_dim)

        logits = self.pi(last)                 # (B, N)

        # Align mask for last timestep and zero out invalid actions
        mask_last = mask[:, -1, :]  # if mask.ndim == 3 else mask
        logits = logits.masked_fill(~mask_last, float("-inf"))

        state_out = {
            'h': h_out.permute(1, 0, 2)
        }

        vf = self.vf(last)
        vf = vf.unsqueeze(1) if vf.dim() == 1 else vf
        logits = logits.unsqueeze(1)

        # print("logits", logits.shape, "vf", vf.shape, "state", state_out['h'].shape)
        #
        # B = x.shape[0]
        # assert logits.shape[:2] == (B, 1)
        # assert vf.shape[:2] == (B, 1)
        # assert state_out['h'].shape[:2] == (B, 1)

        return {
            C.ACTION_DIST_INPUTS: logits,
            C.VF_PREDS:           vf,
            C.STATE_OUT:          state_out,
        }

    # Wire the shared helper into the three public paths
    _forward_train        = _forward_masked
    _forward_exploration  = _forward_masked
    _forward_inference    = _forward_masked

class ObservationFunctionGetter(ObservationFunction):
    def __call__(
            self,
            agent_obs: dict,
            *args,
            **kwargs
    ):
        return agent_obs["observations"]

class DumpExtraKeys(DefaultCallbacks):
    def on_episode_step(self, *, episode, **kwargs):
        # Fix the most recent weights_seq_no for each agent to be len(buf)-1
        for aid, sae in episode.agent_episodes.items():
            try:
                buf = sae.get_extra_model_outputs("weights_seq_no")
                if buf is None or len(buf) == 0:
                    continue
                # print(aid, "len=", len(buf), "last=", buf[-1])
                expected = len(buf) - 1
                if buf[-1] != expected:
                    # make sure it’s an int (or np.int64) – some backends care
                    buf[-1] = int(expected)
            except Exception:
                # keep rollout alive even if one agent lacks the key
                pass

class FixLength(DefaultCallbacks):
    def on_episode_step(self, *, episode, **kwargs):
        for aid, sae in episode.agent_episodes.items():
            try:
                buf = sae.get_extra_model_outputs("weights_seq_no")
                if buf is not None and len(buf):
                    buf[-1] = int(len(buf) - 1)  # make latest index monotonic
            except Exception:
                pass


class ObsPPOCatalog(PPOCatalog):
    def _get_encoder_config(
        self, observation_space, model_config_dict, action_space=None, view_requirements=None
    ):
        activation = model_config_dict["fcnet_activation"]
        output_activation = model_config_dict["fcnet_activation"]
        use_lstm = model_config_dict["use_lstm"]

        if use_lstm:
            encoder_config = RecurrentEncoderConfig(
                input_dims=observation_space.shape,
                recurrent_layer_type="lstm",
                hidden_dim=model_config_dict["lstm_cell_size"],
                hidden_weights_initializer=model_config_dict["lstm_kernel_initializer"],
                hidden_weights_initializer_config=model_config_dict[
                    "lstm_kernel_initializer_kwargs"
                ],
                hidden_bias_initializer=model_config_dict["lstm_bias_initializer"],
                hidden_bias_initializer_config=model_config_dict[
                    "lstm_bias_initializer_kwargs"
                ],
                batch_major=True,
                num_layers=1,
                tokenizer_config=None,
            )
        else:
            # TODO (Artur): Maybe check for original spaces here
            # input_space is a 1D Box
            if isinstance(observation_space, Box) and len(observation_space.shape) == 1:
                # In order to guarantee backward compatability with old configs,
                # we need to check if no latent dim was set and simply reuse the last
                # fcnet hidden dim for that purpose.
                hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
                encoder_latent_dim = model_config_dict["fcnet_hiddens"][-1]
                encoder_config = MLPEncoderConfig(
                    input_dims=observation_space.shape,
                    hidden_layer_dims=hidden_layer_dims,
                    hidden_layer_activation=activation,
                    hidden_layer_weights_initializer=model_config_dict[
                        "fcnet_kernel_initializer"
                    ],
                    hidden_layer_weights_initializer_config=model_config_dict[
                        "fcnet_kernel_initializer_kwargs"
                    ],
                    hidden_layer_bias_initializer=model_config_dict[
                        "fcnet_bias_initializer"
                    ],
                    hidden_layer_bias_initializer_config=model_config_dict[
                        "fcnet_bias_initializer_kwargs"
                    ],
                    output_layer_dim=encoder_latent_dim,
                    output_layer_activation=output_activation,
                    output_layer_weights_initializer=model_config_dict[
                        "fcnet_kernel_initializer"
                    ],
                    output_layer_weights_initializer_config=model_config_dict[
                        "fcnet_kernel_initializer_kwargs"
                    ],
                    output_layer_bias_initializer=model_config_dict[
                        "fcnet_bias_initializer"
                    ],
                    output_layer_bias_initializer_config=model_config_dict[
                        "fcnet_bias_initializer_kwargs"
                    ],
                )

            # input_space is a 3D Box
            else:
                # NestedModelConfig
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} found."
                )

        return encoder_config

if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    alg_name = "PPO"
    # ModelCatalog.register_custom_model("masked_rnn", TorchMaskedActions)
    # function that outputs the environment you wish to register.

    def env_creator():
        env = MultiFieldEnv(
            training=True,
            warm_up=0,
            random_budget=True,
        )
        # env = ss.black_death_v3(env)
        return env

    env_name = "cropgymzoo-train"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = env_creator()
    agent_ids = test_env.possible_agents
    obs_space = {agent_id: test_env.observation_space(agent_id) for agent_id in agent_ids}
    act_space = {agent_id: test_env.action_space(agent_id) for agent_id in agent_ids}

    print("Agent IDs:", test_env.possible_agents)

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            action_mask_key="action_mask",
        )
        .env_runners(
            num_env_runners=0,
            create_local_env_runner=True,
            rollout_fragment_length=8,
        )
        .training(
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            lambda_=0.95,
            gamma=0.99,
            # train_batch_size_per_learner=4096,
        )
        .multi_agent(
            policies={
                agent_id: PolicySpec(
                    observation_space=obs_space[agent_id]['observation'],
                    action_space=act_space[agent_id],
                    config={}
                )
                for agent_id in agent_ids
                # "__default__": PolicySpec(
                #     observation_space=obs_space['field-1'],
                #     action_space=act_space['field-1'],
                # )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .framework(framework="torch")
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                multi_rl_module_class=MultiRLModule,
                observation_space=obs_space['field-1'],
                action_space=act_space['field-1'],
                inference_only=False,
                rl_module_specs={
                    agent_id: RLModuleSpec(
                        LSTMModule,
                    ) for agent_id in agent_ids
                },
                # catalog_class=ObsPPOCatalog,           # your module
                # You can pass model_config={} to forward hyperparams if you want
            ),
            model_config=dict(
                max_seq_len=8,
            )
            # model_config=DefaultModelConfig(
            #     use_lstm=True,  # built-in recurrent wrapper (LSTM)
            #     max_seq_len=32,  # BPTT length
            #     fcnet_hiddens=[256, 256],  # encoder before the LSTM
            # ),

        )
        # .callbacks(
        # #     on_episode_end=(
        # #         lambda episode, **kw: print(f"Episode done. R={episode.get_return()}")
        # #     )
        #     make_multi_callbacks([DumpExtraKeys, FixLength]),
        # )
    )

    algo = config.build()
    print(algo.train())

    algo.stop()
    ray.shutdown()

    # tune.run(
    #     alg_name,
    #     name="PPO",
    #     stop={"timesteps_total": 10000000 if not os.environ.get("CI") else 50000},
    #     checkpoint_freq=10,
    #     config=config.to_dict(),
    # )