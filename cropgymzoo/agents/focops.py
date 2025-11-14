from tianshou.policy.modelfree.ppo import TPPOTrainingStats
from torch import nn as nn
from typing import Any

import numpy as np
import torch
from tianshou.data import ReplayBuffer, to_torch_as
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils import RunningMeanStd

from cropgymzoo.agents.lagppo import LagrangianIPPOPolicy, ActorCriticConstraint, _last1d, IPPOPolicy, IPPOTrainingStats


class IFOCOPSPolicy(LagrangianIPPOPolicy):
    """FOCOPS (First-Order Constrained Optimization in Policy Space) for IPPO.

    This implements the SafePo FOCOPS update rule on top of your IPPO/Lagrangian
    infrastructure, using:
      - reward critic  : self.critic
      - cost critic    : self.constraint_critic
      - Lagrange multip: self.lagrange
    """

    def __init__(
        self,
        *,
        constraint_critic: torch.nn.Module,
        # Lagrangian hyper-parameters
        cost_limit: float = 0.0,
        initial_lagrangian_multiplier: float = 0.001,
        lagrangian_learning_rate: float = 5e-4,
        lagrangian_upper_bound: float = 2.0,  # FOCOPS_NU
        # FOCOPS hyper-parameters
        focops_lambda: float = 1.5,          # FOCOPS_LAM
        target_kl: float = 0.02,
        use_critic_norm: bool = False,
        critic_l2_coeff: float = 1e-3,
        use_value_coefficient: bool = False,  # 2*V_r + V_c like safepo
        # logging
        logger=None,
        # recurrent support inherited from LagrangianIPPOPolicy
        recurrent: bool = False,
        unroll_len: int = 32,
        burn_in: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            constraint_critic=constraint_critic,
            constraint_loss_coefficient=1.0,  # not used in FOCOPS loss
            initial_lagrangian_multiplier=initial_lagrangian_multiplier,
            lagrangian_learning_rate=lagrangian_learning_rate,
            lagrangian_upper_bound=lagrangian_upper_bound,
            const_norm=False,
            norm_const_adv=False,
            logger=logger,
            recurrent=recurrent,
            unroll_len=unroll_len,
            burn_in=burn_in,
            **kwargs,
        )
        # Override Lagrange cost limit & upper bound specifically for FOCOPS
        self.lagrange.cost_limit = cost_limit
        self.lagrange.lagrangian_upper_bound = lagrangian_upper_bound

        self.focops_lambda = float(focops_lambda)
        self.target_kl = float(target_kl)
        self.use_critic_norm = bool(use_critic_norm)
        self.critic_l2_coeff = float(critic_l2_coeff)
        self.use_value_coefficient = bool(use_value_coefficient)

        # separate RMS for normalising costs if needed later
        self.const_rms = RunningMeanStd()

        # combined actor + two critics
        self._actor_critic = ActorCriticConstraint(self.actor, self.critic, self.constraint_critic)

    # ---- Advantage computation (reward + cost) ---------------------------

    def _compute_cost_gae(
        self,
        batch: RolloutBatchProtocol,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute cost returns and advantages via GAE on the constraint critic.

        Assumes per-step cost is stored in batch.info["cost"].
        Works in flat (non-RNN) form, using done flags from batch.done.
        """
        # Extract scalar cost per step (N,)
        if "TotalConstraint" not in batch.info:
            raise KeyError(
                "FOCOPSIPPOPolicy expects per-step cost in batch.info['TotalConstraint']."
            )
        cost = _last1d(batch.info["TotalConstraint"])  # np.ndarray, shape [N]

        # Critic values for cost V_c(s_t) and V_c(s_{t+1})
        v_c, v_c_next = [], []
        with torch.no_grad():
            for mb in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                v_c.append(self.constraint_critic(mb.obs, info=mb.info))
                v_c_next.append(self.constraint_critic(mb.obs_next, info=mb.info))
        v_c = torch.cat(v_c, dim=0).flatten().cpu().numpy()
        v_c_next = torch.cat(v_c_next, dim=0).flatten().cpu().numpy()

        # Done flags
        done = np.asarray(batch.done).astype(bool)
        if done.ndim > 1:
            done = done[..., -1]
        end_flag = done.astype(np.bool_)

        # GAE for costs (classic scalar implementation, equivalent to _gae_return)
        T = len(cost)
        adv_c = np.zeros_like(cost, dtype=np.float64)
        last_gae = 0.0
        gamma = float(self.gamma)
        lam = float(self.gae_lambda)

        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - float(end_flag[t])
            delta = cost[t] + gamma * v_c_next[t] * nonterminal - v_c[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv_c[t] = last_gae

        returns_c = adv_c + v_c
        return returns_c, adv_c

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        """Pre-process batch for FOCOPS.

        1) Use IPPO's process_fn to compute reward returns & advantages
           + logp_old, v_s, etc.
        2) Compute cost returns & advantages with the constraint critic.
        3) Build combined advantage A = (A_r - λ * A_c) / (1 + λ).
        """
        # First: IPPO reward-side preprocessing (GAE on reward, logp_old ...)
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

        # Call the "raw" IPPO version explicitly, not Lagrangian's
        batch = IPPOPolicy.process_fn(self, batch, buffer, indices)

        # Compute logp_old from current policy (as in IPPOPolicy)
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                lp = self(minibatch).dist.log_prob(minibatch.act)
                if lp.ndim > 1:
                    lp = lp.sum(-1)
                logp_old.append(lp)
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()

        # Second: cost-side GAE using constraint_critic
        cost_returns, cost_adv = self._compute_cost_gae(batch)
        batch.cost_returns = to_torch_as(cost_returns, batch.v_s)
        batch.cost_adv = to_torch_as(cost_adv, batch.v_s)

        # Optional normalization of cost advantages for stability (off by default)
        if self.const_norm:
            self.const_rms.update(cost_adv)
            cost_adv_norm = self.const_rms.norm(cost_adv)
            batch.cost_adv = to_torch_as(cost_adv_norm, batch.v_s)

        # Third: combined FOCOPS advantage: Ã = (A_r - λ A_c) / (1 + λ)
        lambda_c = float(self.lagrange.lagrangian_multiplier)
        denom = lambda_c + 1.0 + 1e-8
        combined_adv = (batch.adv - lambda_c * batch.cost_adv) / denom
        batch.focops_adv = combined_adv

        return batch

    # ---- Learn / update step ---------------------------------------------

    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TPPOTrainingStats:
        """FOCOPS update.

        Uses:
          loss_pi = E[ ( KL(π || π_old) - (1/λ_F) * ρ * Ã ) * 1[KL <= target_kl] ]
        where ρ = π(a|s) / π_old(a|s), and Ã is the combined advantage.
        """
        # Update Lagrange multiplier from average cost (rough approximation)
        with torch.no_grad():
            if hasattr(batch, "cost_returns"):
                ep_cost = batch.cost_returns.mean().item()
            elif "TotalConstraint" in batch.info:
                ep_cost = float(np.mean(_last1d(batch.info["TotalConstraint"])))
            else:
                ep_cost = 0.0
        self.lagrange.update_lagrange_multiplier(ep_cost)

        # Prepare old policy distribution parameters (logits) for KL
        with torch.no_grad():
            old_out = self(batch)
            if self.action_type == "discrete":
                old_logits = old_out.dist.logits.detach()
            else:
                # For continuous, store generic "params" (e.g., loc/scale)
                # You can adapt this as needed if you use continuous actions.
                old_logits = old_out.logits.detach()
        batch.old_logits = old_logits

        losses, vf_losses, cost_vf_losses, ent_losses = [], [], [], []
        approx_kls, clipfracs, explained_variances = [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1

        for _ in range(repeat):
            # We follow the SafePo idea: multiple passes over the same data,
            # but break early when KL exceeds target_kl.
            # At each outer repeat, recompute KL wrt the original old policy.
            new_out_full = self(batch)
            if self.action_type == "discrete":
                new_full_dist = new_out_full.dist
                old_full_dist = torch.distributions.Categorical(logits=batch.old_logits)
            else:
                new_full_dist = new_out_full.dist
                # For continuous, you may need to reconstruct the old dist here.
                old_full_dist = new_full_dist.__class__(**new_full_dist.__dict__)  # placeholder

            full_kl = torch.distributions.kl_divergence(new_full_dist, old_full_dist)
            if full_kl.ndim > 1:
                full_kl = full_kl.sum(-1)
            mean_kl = full_kl.mean().item()
            approx_kls.append(mean_kl)

            if mean_kl > self.target_kl:
                break

            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1

                # Rebuild old & new distributions for minibatch
                if self.action_type == "discrete":
                    new_dist = self(minibatch).dist
                    old_dist = torch.distributions.Categorical(logits=minibatch.old_logits)
                else:
                    new_dist = self(minibatch).dist
                    old_dist = new_dist.__class__(**new_dist.__dict__)  # placeholder

                # log prob and ratio
                logp = new_dist.log_prob(minibatch.act)
                if logp.ndim > 1:
                    logp = logp.sum(-1)
                ratio = torch.exp(logp - minibatch.logp_old)

                # KL per sample
                kls = torch.distributions.kl_divergence(new_dist, old_dist)
                if kls.ndim > 1:
                    kls = kls.sum(-1)

                # FOCOPS combined advantage
                adv = minibatch.focops_adv
                if self.norm_adv:
                    mean, std = adv.mean(), adv.std()
                    adv = (adv - mean) / (std + self._eps)

                # Indicator 1[KL <= target_kl]
                kl_mask = (kls.detach() <= self.target_kl).float()

                # Policy loss
                lam_f = float(self.focops_lambda)
                loss_pi_term = kls - (1.0 / lam_f) * ratio * adv
                loss_pi = (loss_pi_term * kl_mask).mean()

                # Value loss for reward
                value_r = self.critic(minibatch.obs, info=minibatch.info).flatten()
                vf_loss = (minibatch.returns - value_r).pow(2).mean()

                # Value loss for cost
                value_c = self.constraint_critic(minibatch.obs, info=minibatch.info).flatten()
                cost_vf_loss = (minibatch.cost_returns - value_c).pow(2).mean()

                if self.use_critic_norm:
                    l2_reg_r = sum(p.pow(2).sum() for p in self.critic.parameters())
                    l2_reg_c = sum(p.pow(2).sum() for p in self.constraint_critic.parameters())
                    vf_loss = vf_loss + self.critic_l2_coeff * l2_reg_r
                    cost_vf_loss = cost_vf_loss + self.critic_l2_coeff * l2_reg_c

                # Entropy (for logging only)
                ent_loss = new_dist.entropy().mean()

                # Total loss (with optional 2*V_r + V_c coefficient)
                if self.use_value_coefficient:
                    total_loss = loss_pi + 2.0 * vf_loss + cost_vf_loss
                else:
                    total_loss = loss_pi + vf_loss + cost_vf_loss

                self.optim.zero_grad()
                total_loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.optim.step()

                losses.append(total_loss.item())
                vf_losses.append(vf_loss.item())
                cost_vf_losses.append(cost_vf_loss.item())
                ent_losses.append(ent_loss.item())

                # These are not meaningful in FOCOPS but we fill them for stats compatibility
                clipfracs.append(0.0)

                # Reward critic explained variance
                def _explained_var(true, pred) -> float:
                    y_pred = pred.detach().cpu().numpy()
                    y_true = true.detach().cpu().numpy()
                    var_y = np.var(y_true)
                    return np.nan if var_y == 0.0 else 1.0 - np.var(y_true - y_pred) / var_y

                explained_variances.append(_explained_var(minibatch.returns, value_r))

        # Build training stats compatible with IPPOTrainingStats
        # For clip_loss we can just reuse loss_pi; for cost_vf_loss we don't
        # have a dedicated field, but you can extend IPPOTrainingStats if wanted.
        clip_losses = vf_losses  # placeholder, or you can make a separate list

        stats = IPPOTrainingStats.from_sequences(  # type: ignore[return-value]
            losses=losses,
            clip_losses=clip_losses,
            vf_losses=vf_losses,
            ent_losses=ent_losses,
            approx_kl=approx_kls,
            clipfrac=clipfracs,
            explained_variance=explained_variances,
            gradient_steps=gradient_steps,
        )

        self._log_learn_stats(batch, losses, clip_losses, vf_losses, ent_losses, approx_kls,
                              clipfracs, explained_variances)

        return stats
