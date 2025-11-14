from tianshou.policy.modelfree.ppo import PPOTrainingStats
from torch import nn as nn
from typing import Any

import numpy as np
import torch
from tianshou.data.types import RolloutBatchProtocol

from cropgymzoo.agents.rcpo import IRCPOPolicy


class IPCPOPolicy(IRCPOPolicy):
    """
    Projection-based Constrained Policy Optimization (PCPO) for this Tianshou-based codebase.

    Workflow:
      1) PPO update on task reward (by default with lambda=0; i.e., no penalty).
      2) If mean discounted cost (from `batch.const_returns`) exceeds `cost_limit + projection_tol`,
         run a projection loop: PPO-style clipped update using cost advantages to reduce expected cost,
         with an approximate-KL guard to enforce a small trust region.

    Options:
      - Set `use_lambda_update=True` to combine PCPO with a Lagrange multiplier update (hybrid).
      - Otherwise, lambda is set to 0 during the reward step and restored afterward.

    This class reuses your batching, masking, logging, and advantage handling.
    """

    def __init__(
        self,
        *,
        use_lambda_update: bool = False,
        projection_steps: int = 3,
        projection_batch_size: int | None = None,
        projection_tol: float = 0.0,
        max_proj_kl: float = 0.01,
        projection_lr_scale: float = 1.0,
        constraint_critic: torch.nn.Module | None = None,
        cost_limit: float = 0.0,
        lambda_init: float = 0.0,
        lambda_lr: float = 5e-4,
        lambda_upper_bound: float = 5.0,
        normalize_cost: bool = True,
        logger=None,
        **kwargs,
    ):
        super().__init__(
            constraint_critic=constraint_critic,
            cost_limit=cost_limit,
            lambda_init=lambda_init,
            lambda_lr=lambda_lr,
            lambda_upper_bound=lambda_upper_bound,
            normalize_cost=normalize_cost,
            logger=logger,
            **kwargs,
        )
        self.use_lambda_update = bool(use_lambda_update)
        self.projection_steps = int(projection_steps)
        self.projection_batch_size = projection_batch_size
        self.projection_tol = float(projection_tol)
        self.max_proj_kl = float(max_proj_kl)
        self.projection_lr_scale = float(projection_lr_scale)

    def learn(  # type: ignore[override]
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> PPOTrainingStats:
        # 1) Pure PPO on r' with lambda=0 by default (unless hybrid mode requested)
        backup_lambda = self._lambda.detach().clone()
        if not self.use_lambda_update:
            self._lambda.data = torch.tensor(0.0, dtype=self._lambda.dtype, device=self._lambda.device)

        stats: PPOTrainingStats = super().learn(batch, batch_size, repeat, *args, **kwargs)  # type: ignore[assignment]

        # 2) Lambda update (only if hybrid PCPO+RCPO requested)
        if self.use_lambda_update:
            # lambda was already updated inside RCPO.learn() above
            pass
        else:
            # restore lambda for subsequent calls
            self._lambda.data = backup_lambda

        # 3) Projection loop if still violating constraint
        if not hasattr(batch, "const_returns"):
            return stats

        mean_cost_return = float(batch.const_returns.mean().item())
        violation = mean_cost_return - (self.cost_limit + self.projection_tol)
        if violation <= 0.0:
            if self.logger is not None:
                try:
                    self.logger.write("pcpo/feasible_after_ppo", 1.0)
                    self.logger.write("pcpo/mean_cost_return", mean_cost_return)
                except Exception:
                    pass
            return stats

        if self.logger is not None:
            try:
                self.logger.write("pcpo/feasible_after_ppo", 0.0)
                self.logger.write("pcpo/pre_projection_cost", mean_cost_return)
            except Exception:
                pass

        split_batch_size = self.projection_batch_size or (batch_size or -1)
        proj_steps_taken = 0

        for _ in range(self.projection_steps):
            proj_steps_taken += 1
            approx_kls = []
            for mb in batch.split(split_batch_size, merge_last=True, shuffle=True if getattr(self, "recurrent", False) else False):
                dist = self(mb).dist
                act = mb.act
                logp = dist.log_prob(act)
                if logp.ndim > 1:
                    logp = logp.sum(-1)

                logratio = logp - mb.logp_old
                ratios = logratio.exp().float()

                const_adv = mb.const_adv
                if getattr(self, "norm_const_adv", False):
                    cmean, cstd = const_adv.mean(), const_adv.std()
                    const_adv = (const_adv - cmean) / (cstd + self._eps)

                # PPO-style clipped surrogate on cost advantage.
                # We MINIMIZE this to reduce the expected discounted cost.
                surr1 = ratios * const_adv
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * const_adv
                proj_clip_loss = torch.min(surr1, surr2).mean()

                ent = dist.entropy().mean()
                loss = proj_clip_loss - self.ent_coef * ent

                self.optim.zero_grad()
                # clear grads
                for p in self._actor_critic.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self.max_grad_norm)
                # conservative step
                for p in self._actor_critic.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(self.projection_lr_scale)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = ((ratios - 1) - logratio).mean().item()
                    approx_kls.append(float(approx_kl))

            # trust-region guard
            if len(approx_kls) > 0 and float(np.mean(approx_kls)) > self.max_proj_kl:
                if self.logger is not None:
                    try:
                        self.logger.write("pcpo/projection_early_stop_kl", 1.0)
                        self.logger.write("pcpo/approx_kl_last", float(np.mean(approx_kls)))
                    except Exception:
                        pass
                break

        if self.logger is not None:
            try:
                self.logger.write("pcpo/projection_steps", float(proj_steps_taken))
            except Exception:
                pass

        return stats
