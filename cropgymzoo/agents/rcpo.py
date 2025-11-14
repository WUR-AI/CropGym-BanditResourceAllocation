from tianshou.policy.modelfree.ppo import PPOTrainingStats
from typing import cast, Any

import numpy as np
import torch
from tianshou.data import Batch, to_torch_as, to_numpy, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol, LogpOldProtocol, BatchWithAdvantagesProtocol
from tianshou.policy.base import _gae_return
from tianshou.utils import RunningMeanStd

from cropgymzoo.agents.lagppo import IPPOPolicy, ActorCriticConstraint, IPPOTrainingStats


class IRCPOPolicy(IPPOPolicy):
    """
    Reward Constrained Policy Optimization (RCPO) for this Tianshou-based codebase.

    RCPO optimizes PPO on a *penalized* reward:
        r'_t = r_t - lambda * (c_t - cost_limit)

    with a dual ascent update on the Lagrange multiplier:
        lambda <- [lambda + alpha * (J_c - cost_limit)]_+

    where J_c is the (discounted) expected cost.  We implement:
      • a separate constraint critic for the discounted cost,
      • optional running normalization of costs (like the existing RunningMeanStdSafe),
      • PPO training on the *penalized* reward stream r',
      • a projected gradient ascent update of lambda after each learn() call.

    Expected cost signal location (in priority order):
        batch.info["cost"]  -> batch.cost -> zeros if missing.

    Notes:
      - Mirrors the patterns used in LagrangianIPPOPolicy (RMS, critic calls, masking, logging hooks).
      - Keeps the standard PPO value loss on the *reward* critic. The constraint critic is used only
        for variance-reduced estimates of the discounted cost/advantage.
    """

    def __init__(
        self,
        *,
        constraint_critic: torch.nn.Module | None = None,
        cost_limit: float = 0.0,
        lambda_init: float = 0.0,
        lambda_lr: float = 5e-4,
        lambda_upper_bound: float = 5.0,
        normalize_cost: bool = True,
        logger=None,
        **kwargs,
    ):
        self.recurrent = kwargs.pop("recurrent", None)
        self.unroll_len = kwargs.pop("unroll_len", None)
        super().__init__(**kwargs)

        if constraint_critic is None:
            raise ValueError("RCPOIPPOPolicy requires a `constraint_critic` network.")

        self.constraint_critic = constraint_critic
        self.const_rms = RunningMeanStd()
        self.normalize_cost = bool(normalize_cost)

        # Dual variable (not optimized by torch; updated by projected ascent)
        # Keep on the same device as actor params.
        device = next(self.actor.parameters()).device
        self._lambda = torch.tensor(float(lambda_init), dtype=torch.float32, device=device)

        self.cost_limit = float(cost_limit)
        self.lambda_lr = float(lambda_lr)
        self.lambda_upper_bound = float(lambda_upper_bound)

        self.logger = logger
        self._warned_no_cost = False

        # re-use the combined wrapper for gradient clipping etc.
        self._actor_critic = ActorCriticConstraint(self.actor, self.critic, self.constraint_critic)

    # ---------------------------- helpers ---------------------------- #

    def _extract_cost_array(self, batch: RolloutBatchProtocol) -> np.ndarray:
        """Fetch immediate cost per step as a NumPy array aligned with batch.rew."""
        cost = None
        if hasattr(batch, "info") and isinstance(batch.info, Batch) and "TotalConstraint" in batch.info:
            cost = batch.info["TotalConstraint"]
        elif hasattr(batch, "TotalConstraint"):
            cost = batch.cost

        if cost is None:
            if not self._warned_no_cost:
                print("[RCPO] No cost signal found on batch.info['TotalConstraint'] nor batch.cost. Using zeros.")
                self._warned_no_cost = True
            cost = np.zeros_like(batch.rew)

        # Make sure it's np.ndarray
        if isinstance(cost, torch.Tensor):
            cost = cost.detach().cpu().numpy()
        elif not isinstance(cost, np.ndarray):
            cost = np.asarray(cost, dtype=np.float32)

        return cost.astype(np.float32)

    def _extract_cost_tensor(self, batch: RolloutBatchProtocol) -> torch.Tensor:
        """Fetch immediate cost per step as a torch tensor aligned with batch.rew."""
        cost = None
        # Priority order of where we look for costs
        if hasattr(batch, "info") and isinstance(batch.info, Batch) and "TotalConstraint" in batch.info:
            cost = batch.info["TotalConstraint"]
        elif hasattr(batch, "TotalConstraint"):
            cost = batch.cost

        if cost is None:
            if not self._warned_no_cost:
                print("[RCPO] No cost signal found on batch.info['TotalConstraint'] nor batch.cost. Using zeros.")
                self._warned_no_cost = True
            cost = torch.zeros_like(batch.rew)

        # Ensure tensor on same device/dtype as rewards
        if isinstance(cost, np.ndarray):
            cost = torch.as_tensor(cost, dtype=torch.float32)
        if isinstance(batch.rew, np.ndarray):
            batch.rew = torch.as_tensor(batch.rew, dtype=torch.float32)
        cost = to_torch_as(cost, batch.rew)
        return cost

    def _compute_cost_values_and_gae(
            self,
            batch: RolloutBatchProtocol,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (const_returns, const_adv) using the constraint critic and GAE."""

        # 1) Get constraint values V_c(s) and V_c(s')
        c_s, c_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                c_s.append(self.constraint_critic(minibatch.obs, info=minibatch.info))
                c_s_.append(self.constraint_critic(minibatch.obs_next, info=minibatch.info))

        batch.c_s = torch.cat(c_s, dim=0).flatten()
        c_s_np = batch.c_s.detach().cpu().numpy()
        c_s_next_np = torch.cat(c_s_, dim=0).flatten().cpu().numpy()

        # 2) Immediate costs (same shape as rewards)

        cost_np = self._extract_cost_array(batch)

        # Optional normalization (RMS over *costs*)
        if self.normalize_cost:
            self.const_rms.update(cost_np)
            cost_np = self.const_rms.norm(cost_np)

        # 3) End flags for GAE (done / terminated / truncated)
        if hasattr(batch, "terminated") and hasattr(batch, "truncated"):
            end_flag = np.logical_or(
                to_numpy(batch.terminated),
                to_numpy(batch.truncated),
            )
        else:
            end_flag = to_numpy(batch.done)

        # 4) GAE on the cost stream
        gamma = self.gamma
        gae_lmb = self.gae_lambda if hasattr(self, "gae_lambda") else 0.95

        # _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        const_adv = _gae_return(
            c_s_np,
            c_s_next_np,
            cost_np,
            end_flag,
            gamma,
            gae_lmb,
        )

        const_returns = const_adv + c_s_np
        return const_returns, const_adv

    # ------------------------ PPO integration ------------------------ #

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        # 0) Align done/terminated/truncated with agent Alive flags (same as IPPO)
        if "Alive" in batch.info:
            bat_done = batch.info["Alive"] == False
            bat_term = batch.info["Alive"] == False
            batch.done = bat_done
            batch.terminated = bat_term
            batch.truncated = bat_term
        if "Alive" in buffer.info:
            buf_done = buffer.info["Alive"] == False
            buf_term = buffer.info["Alive"] == False
            buffer._meta.done = buf_done
            buffer._meta.terminated = buf_term
            buffer._meta.truncated = buf_term

        # 1) Compute cost critic targets and store const_returns/const_adv
        const_returns_np, const_adv_np = self._compute_cost_values_and_gae(batch)
        batch.const_returns = to_torch_as(const_returns_np, batch.c_s)
        batch.const_adv = to_torch_as(const_adv_np, batch.c_s)

        # 2) Form the RCPO penalized reward and temporarily replace batch.rew
        cost_t = self._extract_cost_array(batch)
        # r' = r - lambda * (cost - cost_limit); note that subtracting lambda*cost_limit
        # is a constant shaping that doesn't change the policy gradient, but it stabilizes
        # the value targets. Keeping it here to mirror the paper.
        lam = float(self._lambda.detach().item())
        penalized_rew = batch.rew - lam * (cost_t - self.cost_limit)

        # Keep a copy for debugging
        batch.policy.orig_rew = batch.rew.copy()
        batch.rew = penalized_rew

        # 3) Now run the standard PPO processing to compute v_s, returns, adv and logp_old
        if self.recompute_adv:
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)  # reward-side v_s/returns/adv
        batch.act = to_torch_as(batch.act, batch.v_s)

        # Compute logp_old without touching the networks' grads
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        batch: LogpOldProtocol
        return batch

    def _compute_returns(  # identical to PPO/A2C reward-side logic
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs, info=minibatch.info))
                v_s_.append(self.critic(minibatch.obs_next, info=minibatch.info))
        batch.v_s = torch.cat(v_s, dim=0).flatten()
        v_s_np = batch.v_s.detach().cpu().numpy()
        v_s_next_np = torch.cat(v_s_, dim=0).flatten().cpu().numpy()

        # Unnormalize if reward normalization is enabled (mirror PPO/A2C)
        if self.rew_norm:
            v_s_np = v_s_np * np.sqrt(self.ret_rms.var + self._eps)
            v_s_next_np = v_s_next_np * np.sqrt(self.ret_rms.var + self._eps)

        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_next_np,
            v_s_np,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if self.rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    # ----------------------- training & lambda ----------------------- #

    def learn(  # type: ignore[override]
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> PPOTrainingStats:
        """
        Train with standard PPO loss on the *penalized* reward computed in process_fn().
        After PPO updates, perform a projected gradient ascent step on lambda using
        the mean discounted cost return from this batch.
        """
        # Run standard PPO optimization (clip loss, vf loss, entropy)
        stats: IPPOTrainingStats = super().learn(batch, batch_size, repeat, *args, **kwargs)  # type: ignore[assignment]

        # Dual ascent: lambda <- [lambda + lr * (E[J_c] - cost_limit)]_+
        # Use the batch's discounted cost returns as a proxy for E[J_c].
        if not hasattr(batch, "const_returns"):
            # Safety: if costs are missing, skip the update.
            return stats

        mean_cost_return = float(batch.const_returns.mean().item())
        delta = mean_cost_return - self.cost_limit
        new_lambda = float(self._lambda.detach().item() + self.lambda_lr * delta)
        new_lambda = float(np.clip(new_lambda, 0.0, self.lambda_upper_bound))

        # Keep device consistent
        self._lambda = self._lambda.detach()  # just in case
        self._lambda.data = torch.tensor(new_lambda, dtype=self._lambda.dtype, device=self._lambda.device)

        if self.logger is not None:
            name = self._get_agent_name(batch)
            try:
                more_data = {
                    f"rcpo/lambda/{name}": self._mean(new_lambda),
                    f"cost/total_cost/{name}": self._mean(batch.info["TotalConstraint"]),
                    f"rcpo/mean_cost_return/{name}": mean_cost_return,
                    f"rcpo/cost_limit/{name}": self.cost_limit,
                    f"cost/lagrangian_multiplier/{name}": self.lagrange.lagrangian_multiplier,
                }
                self.logger.write("training/train", step=self._update_step, data=more_data)
            except Exception:
                pass

        return stats

    @property
    def lambda_value(self) -> float:
        return float(self._lambda.detach().item())
