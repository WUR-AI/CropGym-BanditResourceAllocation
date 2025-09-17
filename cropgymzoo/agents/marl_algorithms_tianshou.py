from typing import Any, cast, Sequence, Self
from copy import copy
import time

import logging

import torch
import torch.nn as nn
import numpy as np

from tianshou.data import to_torch_as, to_numpy, ReplayBuffer, Batch, SequenceSummaryStats
from tianshou.data.types import BatchWithAdvantagesProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import _gae_return
from tianshou.policy.modelfree.ppo import PPOPolicy, TPPOTrainingStats, PPOTrainingStats
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.data.collector import (
    Collector,
    TCollectStats,
    _nullable_slice,
    CollectStepBatchProtocol,
    EpisodeBatchProtocol,
    MalformedBufferError,
)
from tianshou.utils.determinism import TraceLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.statistics import RunningMeanStd

from dataclasses import dataclass


class ActorCriticConstraint(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module, constraint_critic: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.constraint_critic = constraint_critic


class IPPOPolicy(PPOPolicy):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def process_fn(self, batch, buffer, indices):
        # build per-step done that includes agent deaths
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
        if self.recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                lp = self(minibatch).dist.log_prob(minibatch.act)
                # if lp.ndim > 1:
                #     lp = lp.sum(-1)
                logp_old.append(lp)
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        batch: LogpOldProtocol
        return batch

class LagrangianIPPOPolicy(IPPOPolicy):
    def __init__(
            self,
            constraint_critic: torch.nn.Module = None,
            constraint_loss_coefficient: float = 0.5,
            initial_lagrangian_multiplier: float = 0.001,
            lagrangian_learning_rate: float = 0.0005,
            lagrangian_upper_bound: float = 3.0,
            const_norm: bool = False,
            logger = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.constraint_critic = constraint_critic
        self.const_rms = RunningMeanStd()
        self.cf_coef = constraint_loss_coefficient
        self.lagrange = Lagrange(
            cost_limit=0.0,
            lagrangian_multiplier_init=initial_lagrangian_multiplier,
            lagrangian_multiplier_lr = lagrangian_learning_rate,
            lagrangian_upper_bound = lagrangian_upper_bound,
        )
        self.const_norm = const_norm
        self._actor_critic = ActorCriticConstraint(self.actor, self.critic, self.constraint_critic)

        self.logger = logger

    def _compute_returns(
            self,
            batch: RolloutBatchProtocol,
            buffer: ReplayBuffer,
            indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        """
        Adding the constraint critic calculation here
        """
        v_s, v_s_ = [], []
        c_s, c_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs))
                v_s_.append(self.critic(minibatch.obs_next))
                c_s.append(self.constraint_critic(minibatch.obs))
                c_s_.append(self.constraint_critic(minibatch.obs_next))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        batch.c_s = torch.cat(c_s, dim=0).flatten()

        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()

        c_s = batch.c_s.cpu().numpy()
        c_s_ = torch.cat(c_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Empirical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        # TODO: see todo in PGPolicy.process_fn
        if self.rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        if self.const_norm:
            c_s = c_s * np.sqrt(self.const_rms.var + self._eps)
            c_s_ = c_s_ * np.sqrt(self.const_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        const_returns, constraint_advantages = self.compute_episodic_cost(
            batch,
            buffer,
            indices,
            c_s_,
            c_s,
            gamma=1.0,
            gae_lambda=self.gae_lambda,
        )

        if self.rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        if self.const_norm:
            batch.const_returns = const_returns / np.sqrt(self.const_rms.var + self._eps)
            self.const_rms.update(const_returns)
        else:
            batch.const_returns = const_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)

        batch.const_returns = to_torch_as(batch.const_returns, batch.c_s)
        batch.const_adv = to_torch_as(constraint_advantages, batch.c_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    @staticmethod
    def compute_episodic_cost(
            batch: RolloutBatchProtocol,
            buffer: ReplayBuffer,
            indices: np.ndarray,
            v_s_: np.ndarray | torch.Tensor | None = None,
            v_s: np.ndarray | torch.Tensor | None = None,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:

        cost = batch.info['TotalConstraint']
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(cost)
        else:
            v_s_ = to_numpy(v_s_.flatten())
            v_s_ = v_s_ * BasePolicy.value_mask(buffer, indices)
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s.flatten())

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        if len(end_flag.shape) > 1:
            end_flag = end_flag[:, -1]
        advantage = _gae_return(v_s, v_s_, cost, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage


    @staticmethod
    def compute_episodic_return(
            batch: RolloutBatchProtocol,
            buffer: ReplayBuffer,
            indices: np.ndarray,
            v_s_: np.ndarray | torch.Tensor | None = None,
            v_s: np.ndarray | torch.Tensor | None = None,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:

        rew = batch.rew
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(rew)
        else:
            v_s_ = to_numpy(v_s_.flatten())
            v_s_ = v_s_ * BasePolicy.value_mask(buffer, indices)
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s.flatten())

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        if len(end_flag.shape) > 1:
            end_flag = end_flag[:, -1]
        advantage = _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage

    def learn(  # type: ignore
            self,
            batch: RolloutBatchProtocol,
            batch_size: int | None,
            repeat: int,
            *args: Any,
            **kwargs: Any,
    ) -> TPPOTrainingStats:
        losses, clip_losses, vf_losses, ent_losses, cf_losses = [], [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1

        total_constraint = batch.info["TotalEpisodicConstraint"]
        done = batch.done

        # lagrangian stuff
        # final_constraint_values = [
        #     float(tc)
        #     for tc, d in zip(total_constraint, done)
        #     if d and tc is not None
        # ]
        # mean_ep_constraint_values = float(np.mean(final_constraint_values)) if final_constraint_values else 0.0
        if total_constraint.ndim > 1 and total_constraint.shape[-1] == 1:
            total_constraint = np.squeeze(total_constraint, axis=-1)
        final_tc = total_constraint[done]
        mean_ep_constraint_values = float(np.mean(final_tc)) if final_tc.size > 0 else 0.0
        lagrangian_multiplier = float(self.lagrange.lagrangian_multiplier)

        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                # calculate loss for actor
                advantages = minibatch.adv

                constraint_advantages = minibatch.const_adv

                dist = self(minibatch).dist
                if self.norm_adv:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)  # per-batch norm

                    const_mean, const_std = constraint_advantages.mean(), constraint_advantages.std()
                    constraint_advantages = (constraint_advantages - const_mean) / (const_std + self._eps)

                # start lagrangian constraint
                combined_advantages = advantages - lagrangian_multiplier * constraint_advantages

                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)

                surr1 = ratios * combined_advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * combined_advantages
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * combined_advantages)
                    clip_loss = -torch.where(combined_advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                constraint_value = self.constraint_critic(minibatch.obs).flatten()

                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip,
                        self.eps_clip,
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                # calculate constraint returns
                cf_loss = (minibatch.const_returns - constraint_value).pow(2).mean()

                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self.vf_coef * vf_loss + self.cf_coef * cf_loss - self.ent_coef * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                cf_losses.append(cf_loss.item())
                losses.append(loss.item())

        self.lagrange.update_lagrange_multiplier(mean_ep_constraint_values)

        return IPPOTrainingStats.from_sequence(  # type: ignore[return-value]
            losses=losses,
            clip_losses=clip_losses,
            vf_losses=vf_losses,
            cf_losses=cf_losses,
            ent_losses=ent_losses,
            gradient_steps=gradient_steps,
        )

log = logging.getLogger(__name__)

class IPPOCollector(Collector):

    @staticmethod
    def _assign_obs_next_row(buffer: ReplayBuffer, idx: int, value: Batch) -> None:
        """Overwrite obs_next of a single transition row with a nested Batch `value`."""
        # Use a 1-length index to get a 1-row Batch view
        row_idx = np.array([int(idx)])
        original_stack = buffer.stack_num
        buffer.stack_num = 1
        row = buffer[row_idx]  # -> Batch of length 1
        buffer.stack_num = original_stack
        # if isinstance(value, dict):
        #     for k, v in value.items():
        #         if not isinstance(v, ndarray):
        row.obs_next = value  # assign nested Batch directly
        buffer._meta[row_idx] = row  # write back to buffer

    def _collect(  # noqa: C901
            self,
            n_step: int | None = None,
            n_episode: int | None = None,
            random: bool = False,
            render: float | None = None,
            gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> TCollectStats:
        """This method is currently very complex, but it's difficult to break it down into smaller chunks.

        Please read the block-comment of the class to understand the notation
        in the implementation.

        It does the collection by executing the following logic:

        0. Keep track of n_step and n_episode for being able to stop the collection.
        1.  Create a CollectStats instance to store the statistics of the collection.
        2.  Compute actions (with policy or sampling from action space) for the R currently active envs.
        3.  Perform a step in these R envs.
        4.  Perform on-step hook on the result
        5.  Update the CollectStats (using `update_at_step_batch`) and the internal counters after the step
        6.  Add the resulting R transitions to the buffer
        7.  Find the D envs that reached done in the current iteration
        8.  Reset the envs that reached done
        9.  Extract episodes for the envs that reached done from the buffer
        10. Perform on-episode-done hook. If it has a return, modify the transitions belonging to the episodes inside the buffer inplace
        11. Update the CollectStats instance with the episodes from 9. by using `update_on_episode_done`
        12. Prepare next step in while loop by saving the last observations and infos
        13. Remove S surplus envs from collection mechanism, thereby reducing R to R-S, to increase performance
        14. Update instance-level collection counters (contrary to counters with a lifetime of the collect execution)
        15. Prepare for the next call of collect (save last observations and info to collector state)

        You can search for Step <n> to find where it happens
        """
        # TODO: can't do it init since AsyncCollector is currently a subclass of Collector
        if self.env.is_async:
            raise ValueError(
                f"Please use AsyncCollector for asynchronous environments. "
                f"Env class: {self.env.__class__.__name__}.",
            )

        ready_env_ids_R: np.ndarray[Any, np.dtype[np.signedinteger]]
        """provides a mapping from local indices (indexing within `1, ..., R` where `R` is the number of ready envs)
         to global ones (indexing within `1, ..., num_envs`). So the entry i in this array is the global index of the i-th ready env."""
        if n_step is not None:
            ready_env_ids_R = np.arange(self.env_num)
        elif n_episode is not None:
            if self.env_num > n_episode:
                log.warning(
                    f"Number of episodes ({n_episode}) is smaller than the number of environments "
                    f"({self.env_num}). This means that {self.env_num - n_episode} "
                    f"environments (or, equivalently, parallel workers) will not be used!",
                )
            ready_env_ids_R = np.arange(min(self.env_num, n_episode))
        else:
            raise RuntimeError("Input validation failed, this is a bug and shouldn't have happened")

        if self._pre_collect_obs_RO is None or self._pre_collect_info_R is None:
            raise ValueError(
                "Initial obs and info should not be None. "
                "Either reset the collector (using reset or reset_env) or pass reset_before_collect=True to collect.",
            )

        # --- NEW: pending map for aligning AEC obs_next to same-agent next obs ---
        # Maps global_env_id -> { agent_id -> last transition index in buffer that still needs obs_next }
        if not hasattr(self, "_pending_idx_by_env_agent"):
            self._pending_idx_by_env_agent: dict[int, dict[Any, int]] = {i: {} for i in range(self.env_num)}
        # Maps agent_id -> { global_env_id -> state_for_that_agent_in_that_env }
        if not hasattr(self, "_hs_bank_by_agent_env"):
            self._hs_bank_by_agent_env: dict[Any, dict[int, Any]] = {}

        # Step 0
        # get the first obs to be the current obs in the n_step case as
        # episodes as a new call to collect does not restart trajectories
        # (which we also really don't want)
        step_count = 0
        num_collected_episodes = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        # Step 1
        collect_stats = self.collect_stats_class()

        # in case we select fewer episodes than envs, we run only some of them
        last_obs_RO = _nullable_slice(self._pre_collect_obs_RO, ready_env_ids_R)
        last_info_R = _nullable_slice(self._pre_collect_info_R, ready_env_ids_R)
        last_hidden_state_RH = _nullable_slice(
            self._pre_collect_hidden_state_RH,
            ready_env_ids_R,
        )

        while True:
            # todo check if we need this when using cur_rollout_batch
            # if len(cur_rollout_batch) != len(ready_env_ids):
            #     raise RuntimeError(
            #         f"The length of the collected_rollout_batch {len(cur_rollout_batch)}) is not equal to the length of ready_env_ids"
            #         f"{len(ready_env_ids)}. This should not happen and could be a bug!",
            #     )
            # restore the state: if the last state is None, it won't store

            state_filtered_RH = self._build_filtered_state_for_forward(
                ready_env_ids_R=ready_env_ids_R,
                last_obs_RO=last_obs_RO,
            )

            # Step 2
            # get the next action and related stats from the previous observation
            collect_action_computation_batch_R = self._compute_action_policy_hidden(
                random=random,
                ready_env_ids_R=ready_env_ids_R,
                last_obs_RO=last_obs_RO,
                last_info_R=last_info_R,
                last_hidden_state_RH=state_filtered_RH  # last_hidden_state_RH,
            )

            # if len(collect_action_computation_batch_R.act.shape) > 1:
            #     collect_action_computation_batch_R.act = collect_action_computation_batch_R.act[:, -1]
            #     collect_action_computation_batch_R.act_normalized = collect_action_computation_batch_R.act_normalized[:, -1]

            self._update_hs_bank_from_forward(
                collect_action_computation_batch_R.hidden_state,
                ready_env_ids_R,
                last_obs_RO,
            ) if collect_action_computation_batch_R.hidden_state else None

            TraceLogger.log(log, lambda: f"Action: {collect_action_computation_batch_R.act}")

            # Step 3
            obs_next_RO, rew_R, terminated_R, truncated_R, info_R = self.env.step(
                collect_action_computation_batch_R.act_normalized,
                ready_env_ids_R,
            )
            if isinstance(info_R, dict):  # type: ignore[unreachable]
                # This can happen if the env is an envpool env. Then the info returned by step is a dict
                info_R = _dict_of_arr_to_arr_of_dicts(info_R)  # type: ignore[unreachable]
            done_R = np.logical_or(terminated_R, truncated_R)

            current_step_batch_R = cast(
                CollectStepBatchProtocol,
                Batch(
                    obs=last_obs_RO,
                    dist=collect_action_computation_batch_R.dist,
                    act=collect_action_computation_batch_R.act,
                    policy=collect_action_computation_batch_R.policy_entry,
                    obs_next=obs_next_RO,
                    rew=rew_R,
                    terminated=terminated_R,
                    truncated=truncated_R,
                    done=done_R,
                    info=info_R,
                ),
            )

            # TODO: only makes sense if render_mode is human.
            #  Also, doubtful whether it makes sense at all for true vectorized envs
            if render:
                self.env.render()
                if not np.isclose(render, 0):
                    time.sleep(render)

            # Step 4
            self.run_on_step_hook(
                collect_action_computation_batch_R,
                current_step_batch_R,
            )

            # --- NEW
            # Step 4a
            # check if policy Batches are of size `num_env`; broadcast from cache if not
            temp_current_step_batch_R = copy(current_step_batch_R)
            temp_collect_action_computation_batch_R = copy(collect_action_computation_batch_R)
            temp_agent_hs = {}
            for _agent, _hs_agent in temp_current_step_batch_R['policy']['hidden_state'].items():
                if not _hs_agent:
                    continue
                if len(_hs_agent.shape) == 3 and _hs_agent['hidden'].shape[0] < len(ready_env_ids_R):
                    hidden_agent = torch.cat(
                        [
                            self._hs_bank_by_agent_env[_agent][i]['hidden']
                            for i in ready_env_ids_R
                        ],
                        dim=0
                    )
                    temp_agent_hs[_agent] = hidden_agent

            if temp_agent_hs:
                for _agent, _hs_agent in temp_agent_hs.items():
                    temp_current_step_batch_R['policy']['hidden_state'][_agent]['hidden'] = _hs_agent
                    temp_collect_action_computation_batch_R['hidden_state'][_agent]['hidden'] = _hs_agent
                    temp_collect_action_computation_batch_R['policy_entry']['hidden_state'][_agent]['hidden'] = _hs_agent
                current_step_batch_R = copy(temp_current_step_batch_R)
                collect_action_computation_batch_R = copy(temp_collect_action_computation_batch_R)


            # Step 5, collect statistics
            collect_stats.update_at_step_batch(current_step_batch_R)
            num_episodes_done_this_iter = np.sum(done_R)
            num_collected_episodes += num_episodes_done_this_iter
            step_count += len(ready_env_ids_R)

            # Step 6
            # add data into the buffer. Since the buffer is essentially an array, we don't want
            # to add the dist. One should not have arrays of dists but rather a single, batch-wise dist.
            # Tianshou already implements slicing of dists, but we don't yet implement merging multiple
            # dists into one, which would be necessary to make a buffer with dists work properly
            batch_to_add_R = copy(current_step_batch_R)
            batch_to_add_R.pop("dist")
            batch_to_add_R = cast(RolloutBatchProtocol, batch_to_add_R)
            insertion_idx_R, ep_return_R, ep_len_R, ep_start_idx_R = self.buffer.add(
                batch_to_add_R,
                buffer_ids=ready_env_ids_R,
            )

            # -_-_-_-

            # --- Step 6a (NEW): Realign obs_next to the next time the *same* agent acts ---
            # For each ready env, we just inserted one transition at insertion_idx_R[local_i]
            # whose obs belongs to the *current* agent. The next time this same agent appears,
            # we want to close the *previous* transition by writing obs_next = current obs.
            for local_i, global_env_id in enumerate(ready_env_ids_R):
                # agent_id of the actor that produced last_obs_RO[local_i]
                agent_id = last_obs_RO[local_i]["agent_id"]

                # 1) Close previous pending transition for (env, agent)
                prev_idx = self._pending_idx_by_env_agent.get(int(global_env_id), {}).pop(agent_id, None)
                if prev_idx is not None:
                    # Write obs_next = current *same-agent* obs (i.e., what the agent sees on its next turn)
                    if prev_idx is not None:
                        self._assign_obs_next_row(
                            self.buffer,
                            prev_idx,
                            copy(last_obs_RO[local_i])
                        )

                # 2) Register the just-inserted transition as the new pending one
                self._pending_idx_by_env_agent.setdefault(int(global_env_id), {})[agent_id] = int(
                    insertion_idx_R[local_i]
                )

            # -_-_-_-

            # preparing for the next iteration
            # obs_next, info and hidden_state will be modified inplace in the code below,
            # so we copy to not affect the data in the buffer
            last_obs_RO = copy(obs_next_RO)
            last_info_R = copy(info_R)

            # preserve agent hidden state
            if last_hidden_state_RH is None:
                last_hidden_state_RH = copy(collect_action_computation_batch_R.hidden_state)
            else:
                _temp_last_hidden_state_RH: Batch = copy(collect_action_computation_batch_R.hidden_state)
                _temp_last_hidden_state_RH.replace_empty_batches_by_none()
                for agent_id, _state in _temp_last_hidden_state_RH.items():
                    if _state is not None:
                        last_hidden_state_RH[agent_id] = _temp_last_hidden_state_RH[agent_id]

            # Preparing last_obs_RO, last_info_R, last_hidden_state_RH for the next while-loop iteration
            # Resetting envs that reached done, or removing some of them from the collection if needed (see below)
            if num_episodes_done_this_iter > 0:
                # TODO: adjust the whole index story, don't use np.where, just slice with boolean arrays
                # D - number of envs that reached done in the rollout above
                # local_idx - see block comment on class level
                # Step 7
                env_done_local_idx_D = np.where(done_R)[0]
                """Indexes which episodes are done within the ready envs, so it can be used for selecting from `..._R` arrays.
                Stands in contrast to the "global" index, which counts within all envs and is unsuitable for selecting from `..._R` arrays."""
                episode_lens_D = ep_len_R[env_done_local_idx_D]
                episode_returns_D = ep_return_R[env_done_local_idx_D]
                episode_start_indices_D = ep_start_idx_R[env_done_local_idx_D]

                episode_lens.extend(episode_lens_D)
                episode_returns.extend(episode_returns_D)
                episode_start_indices.extend(episode_start_indices_D)

                # Step 8
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                gym_reset_kwargs = gym_reset_kwargs or {}
                # The index env_done_idx_D was based on 0, ..., R
                # However, each env has an index in the context of the vectorized env and buffer. So the env 0 being done means
                # that some env of the corresponding "global" index was done. The mapping between "local" index in
                # 0,...,R and this global index is maintained by the ready_env_ids_R array.
                # See the class block comment for more details
                env_done_global_idx_D = ready_env_ids_R[env_done_local_idx_D]
                """Indexes which episodes are done within all envs, i.e., within the index `1, ..., num_envs`. It can be
                used to communicate with the vector env, where env ids are selected from this "global" index.
                Is not suited for selecting from the ready envs (`..._R` arrays), use the local counterpart instead.
                """
                obs_reset_DO, info_reset_D = self.env.reset(
                    env_id=env_done_global_idx_D,
                    **gym_reset_kwargs,
                )

                # Set the hidden state to zero or None for the envs that reached done
                # TODO: does it have to be so complicated? We should have a single clear type for hidden_state instead of
                #  this complex logic
                self._reset_hidden_state_based_on_type(env_done_local_idx_D, last_hidden_state_RH)

                # --- Step 8b (NEW): Flush any pending transitions for envs that just finished ---
                for local_done_i, global_env_id in enumerate(env_done_global_idx_D):
                    pendings = self._pending_idx_by_env_agent.get(int(global_env_id), {})
                    if not pendings:
                        continue

                    # Any placeholder with correct shape is fine because done=True prevents bootstrapping.
                    # Use the current last_obs_RO[local_done_i] for shape consistency.
                    terminal_obs_like = copy(last_obs_RO[local_done_i])

                    for _, prev_idx in list(pendings.items()):
                        self._assign_obs_next_row(
                            self.buffer,
                            prev_idx,
                            terminal_obs_like
                        )
                    # Clear all pendings for this env now that the episode ended
                    self._pending_idx_by_env_agent[int(global_env_id)] = {}

                # Try not popping the bank
                # for g in env_done_global_idx_D:
                #     for a in list(self._hs_bank_by_agent_env.keys()):
                #         self._hs_bank_by_agent_env[a].pop(int(g), None)

                # Step 9
                # execute episode hooks for those envs which emitted 'done'
                for local_done_idx, cur_ep_return in zip(
                        env_done_local_idx_D,
                        episode_returns_D,
                        strict=True,
                ):
                    # retrieve the episode batch from the buffer using the episode start and stop indices
                    ep_start_idx, ep_stop_idx = (
                        int(ep_start_idx_R[local_done_idx]),
                        int(insertion_idx_R[local_done_idx] + 1),
                    )

                    ep_index_array = self.buffer.get_buffer_indices(ep_start_idx, ep_stop_idx)
                    ep_batch = cast(EpisodeBatchProtocol, self.buffer[ep_index_array])

                    # Step 10
                    episode_hook_additions = self.run_on_episode_done(ep_batch)
                    if episode_hook_additions is not None:
                        if n_episode is None:
                            raise ValueError(
                                "An on_episode_done_hook with non-empty returns is not supported for n_step collection."
                                "Such hooks should only be used when collecting full episodes. Got a on_episode_done_hook "
                                f"that would add the following fields to the buffer: {list(episode_hook_additions)}.",
                            )

                        for key, episode_addition in episode_hook_additions.items():
                            self.buffer.set_array_at_key(
                                episode_addition,
                                key,
                                index=ep_index_array,
                            )
                            # executing the same logic in the episode-batch since stats computation
                            # may depend on the presence of additional fields
                            ep_batch.set_array_at_key(
                                episode_addition,
                                key,
                            )
                    # Step 11
                    # Finally, update the stats
                    collect_stats.update_at_episode_done(
                        episode_batch=ep_batch,
                        episode_return=cur_ep_return,
                    )

                # Step 12
                # preparing for the next iteration
                last_obs_RO[env_done_local_idx_D] = obs_reset_DO
                last_info_R[env_done_local_idx_D] = info_reset_D

                # Step 13
                # Handling the case when we have more ready envs than desired and are not done yet
                #
                # This can only happen if we are collecting a fixed number of episodes
                # If we have more ready envs than there are remaining episodes to collect,
                # we will remove some of them for the next rollout
                # One effect of this is the following: only envs that have completed an episode
                # in the last step can ever be removed from the ready envs.
                # Thus, this guarantees that each env will contribute at least one episode to the
                # collected data (the buffer). This effect was previous called "avoiding bias in selecting environments"
                # However, it is not at all clear whether this is actually useful or necessary.
                # Additional naming convention:
                # S - number of surplus envs
                # TODO: can the whole block be removed? If we have too many episodes, we could just strip the last ones.
                #   Changing R to R-S highly increases the complexity of the code.
                if n_episode:
                    remaining_episodes_to_collect = n_episode - num_collected_episodes
                    surplus_env_num = len(ready_env_ids_R) - remaining_episodes_to_collect
                    if surplus_env_num > 0:
                        # R becomes R-S here, preparing for the next iteration in while loop
                        # Everything that was of length R needs to be filtered and become of length R-S.
                        # Note that this won't be the last iteration, as one iteration equals one
                        # step and we still need to collect the remaining episodes to reach the breaking condition.

                        # creating the mask
                        env_to_be_ignored_ind_local_S = env_done_local_idx_D[:surplus_env_num]
                        env_should_remain_R = np.ones_like(ready_env_ids_R, dtype=bool)
                        env_should_remain_R[env_to_be_ignored_ind_local_S] = False
                        # stripping the "idle" indices, shortening the relevant quantities from R to R-S
                        ready_env_ids_R = ready_env_ids_R[env_should_remain_R]
                        last_obs_RO = last_obs_RO[env_should_remain_R]
                        last_info_R = last_info_R[env_should_remain_R]
                        if collect_action_computation_batch_R.hidden_state is not None:
                            last_hidden_state_RH = last_hidden_state_RH[env_should_remain_R]  # type: ignore[index]

            if (n_step and step_count >= n_step) or (
                    n_episode and num_collected_episodes >= n_episode
            ):
                break

        # Check if we screwed up somewhere
        if self.raise_on_nan_in_buffer and self.buffer.hasnull():
            nan_batch = self.buffer.isnull().apply_values_transform(np.sum)

            raise MalformedBufferError(
                "NaN detected in the buffer. You can drop them with `buffer.dropnull()`. "
                "This error is most often caused by an incorrect use of `EpisodeRolloutHooks`"
                "together with the `n_steps` (instead of `n_episodes`) option, or by "
                "an incorrect implementation of `StepHook`."
                "Here an overview of the number of NaNs per field: \n"
                f"{nan_batch}",
            )

        # Step 14
        # update instance-lifetime counters, different from collect_stats
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes

        # Step 15
        if n_step:
            # persist for future collect iterations
            self._pre_collect_obs_RO = last_obs_RO
            self._pre_collect_info_R = last_info_R
            self._pre_collect_hidden_state_RH = last_hidden_state_RH
        elif n_episode:
            # reset envs and the _pre_collect fields
            self.reset_env(gym_reset_kwargs)  # todo still necessary?
            # --- NEW: clear pendings since we started fresh episodes
            self._pending_idx_by_env_agent = {i: {} for i in range(self.env_num)}
            self._hs_bank_by_agent_env = {}
        return collect_stats


    def _build_filtered_state_for_forward(
            self,
            ready_env_ids_R: np.ndarray,
            last_obs_RO: Batch,
    ) -> Batch:
        """
        Returns a Batch mapping agent_id -> state whose batch dim equals the number of
        alive samples for that agent in THIS step, in the SAME order as last_obs_RO.
        If any alive slot for an agent has no cached state yet, we pass None for that agent
        so the policy will initialize it (safe & simple).
        """
        agent_ids_R = np.asarray([env_obs["agent_id"] for env_obs in last_obs_RO])
        agent_ids_dict_R = {e: ag for e, ag in zip(ready_env_ids_R, agent_ids_R)}
        result = Batch()

        # make sure all agents seen so far have a dict in the bank
        for a in np.unique(agent_ids_R):
            self._hs_bank_by_agent_env.setdefault(a, {})

        for a in np.unique(agent_ids_R):
            # local positions in this step for this agent and alive
            # local_idx = np.nonzero((agent_ids_R == a) & (alive_R == True))[0]
            # local_idx = np.flatnonzero((agent_ids_dict_R.values() == a))
            local_idx = np.asarray([k for k, v in agent_ids_dict_R.items() if v == a])
            if len(local_idx) == 0:
                continue

            if self._hs_bank_by_agent_env[a]:
                sliced_hs = {i: self._hs_bank_by_agent_env[a][i]['hidden'] for i in local_idx if i in self._hs_bank_by_agent_env[a]}
                hs = torch.cat([sliced_hs[i] for i in local_idx], dim=0)

                result[a] = Batch({'hidden': hs})

        if not result:
            result = None

        return result

    def _update_hs_bank_from_forward(
            self,
            hidden_state_out: Batch,
            ready_env_ids_R: np.ndarray,
            last_obs_RO: Batch,
    ) -> None:
        """
        hidden_state_out is a Batch mapping agent_id -> batched state for alive samples.
        We must map each row back to the correct global_env_id.
        """
        agent_ids_R = np.asarray([env_obs["agent_id"] for env_obs in last_obs_RO])

        for a, batched_state in hidden_state_out.items():
            # positions where we forwarded this agent
            # local_idx = np.nonzero((agent_ids_R == a) & (alive_R == True))[0]
            local_idx = np.flatnonzero((agent_ids_R == a))
            if len(local_idx) == 0:
                continue

            # slice per-sample state from batched_state and put into bank
            for j, li in enumerate(local_idx):
                g = int(ready_env_ids_R[li])
                self._hs_bank_by_agent_env.setdefault(a, {})
                self._hs_bank_by_agent_env[a].setdefault(g, Batch())
                self._hs_bank_by_agent_env[a][g]['hidden'] = batched_state['hidden'][j:j + 1, :, :]


# Lagrange class taken from https://github.com/PKU-Alignment/safety-gymnasium,
# paper Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark
class Lagrange:
    """Lagrange multiplier for constrained optimization.

    Args:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier

    Attributes:
        cost_limit: the cost limit
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier
        _lagrangian_multiplier: the lagrangian multiplier
        lambda_range_projection: the projection function of the lagrangian multiplier
        lambda_optimizer: the optimizer of the lagrangian multiplier
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
            self,
            cost_limit: float,
            lagrangian_multiplier_init: float,
            lagrangian_multiplier_lr: float,
            lagrangian_upper_bound: float | None = None,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.cost_limit: float = cost_limit
        self.lagrangian_multiplier_lr: float = lagrangian_multiplier_lr
        self.lagrangian_upper_bound: float | None = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 0.0)
        self._lagrangian_multiplier: nn.Parameter = nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        self.lambda_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [
                self._lagrangian_multiplier,
            ],
            lr=lagrangian_multiplier_lr,
        )

    @property
    def lagrangian_multiplier(self) -> torch.Tensor:
        """The lagrangian multiplier.

        Returns:
            the lagrangian multiplier
        """
        return self.lambda_range_projection(self._lagrangian_multiplier).detach().item()

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        """Compute the loss of the lagrangian multiplier.

        Args:
            mean_ep_cost: the mean episode cost

        Returns:
            the loss of the lagrangian multiplier
        """
        return -self._lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        """Update the lagrangian multiplier.

        Args:
            Jc: the mean episode cost

        Returns:
            the loss of the lagrangian multiplier
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self._lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]


@dataclass(kw_only=True)
class IPPOTrainingStats(PPOTrainingStats):
    cf_loss: SequenceSummaryStats

    @classmethod
    def from_sequence(
        cls,
        *,
        losses: Sequence[float],
        clip_losses: Sequence[float],
        vf_losses: Sequence[float],
        cf_losses: Sequence[float],
        ent_losses: Sequence[float],
        gradient_steps: int = 0,
    ) -> Self:
        return cls(
            loss=SequenceSummaryStats.from_sequence(losses),
            clip_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            cf_loss=SequenceSummaryStats.from_sequence(cf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            gradient_steps=gradient_steps,
        )
