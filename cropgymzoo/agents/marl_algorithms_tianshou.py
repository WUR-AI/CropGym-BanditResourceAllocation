from typing import Any, cast
from copy import copy
import time

import logging

import torch
import torch.nn as nn
import numpy as np

from tianshou.data import to_torch_as, to_numpy, ReplayBuffer, Batch
from tianshou.policy import BasePolicy
from tianshou.policy.modelfree.ppo import PPOPolicy, TPPOTrainingStats, PPOTrainingStats
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.data.collector import (
    Collector,
    TCollectStats,
    _nullable_slice,
    CollectStepBatchProtocol,
    EpisodeBatchProtocol,
    MalformedBufferError
)
from tianshou.utils.determinism import TraceLogger

class IPPOPolicy(PPOPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def process_fn(self, batch, buffer, indices):
        # build per-step done that includes agent deaths
        if "Alive" in batch.info:
            bat_done = batch.info["Alive"] == False
            bat_term = batch.info["Alive"] == False
            batch.done = bat_done
            batch.terminated = bat_term
        if "Alive" in buffer.info:
            buf_done = buffer.info["Alive"] == False
            buf_term = buffer.info["Alive"] == False
            buffer._meta.done = buf_done
            buffer._meta.terminated = buf_term
        if self.recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        batch: LogpOldProtocol
        return batch

log = logging.getLogger(__name__)

class IPPOCollector(Collector):

    @staticmethod
    def _assign_obs_next_row(buffer: ReplayBuffer, idx: int, value: Batch) -> None:
        """Overwrite obs_next of a single transition row with a nested Batch `value`."""
        # Use a 1-length index to get a 1-row Batch view
        row_idx = np.array([int(idx)])
        row = buffer[row_idx]  # -> Batch of length 1
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

            self._update_hs_bank_from_forward(
                collect_action_computation_batch_R.hidden_state,
                ready_env_ids_R,
                last_obs_RO,
            )

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
                for g in env_done_global_idx_D:
                    for a in list(self._hs_bank_by_agent_env.keys()):
                        self._hs_bank_by_agent_env[a].pop(int(g), None)

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
