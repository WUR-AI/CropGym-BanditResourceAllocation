from dataclasses import dataclass

from tianshou.policy.base import _gae_return
from tianshou.policy.modelfree.ppo import TPPOTrainingStats, PPOTrainingStats
from typing import Any, cast, Sequence, Self

import numpy as np
import torch
from tianshou.data import to_torch_as, Batch, ReplayBuffer, to_numpy, SequenceSummaryStats
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol, BatchWithAdvantagesProtocol
from tianshou.policy import PPOPolicy, BasePolicy
from torch import nn as nn


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


def _last1d(x) -> np.ndarray:
    # numpy array, last time slice if stacked, flattened to 1-D
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[..., -1]
    return x.reshape(-1)


class RunningMeanStdSafe:
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    :param mean: the initial mean estimation for data array. Default to 0.
    :param std: the initial standard error estimation for data array. Default to 1.
    :param clip_max: the maximum absolute value for data array. Default to
        10.0.
    :param epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        clip_max: float | None = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        # NOTE: the "std" argument is actually variance in Tianshou's RMS usage
        self.var = np.asarray(std, dtype=np.float64)
        self.clip_max = clip_max
        self.count = int(0)
        self.eps = float(epsilon)
        # Safety bounds to avoid exploding/vanishing scales
        self._min_var = np.float64(1e-12)
        self._max_var = np.float64(1e12)

    def norm(self, data_array: float | np.ndarray) -> float | np.ndarray:
        # data_array = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        data_array = (np.asarray(data_array, dtype=np.float64) - self.mean) / np.sqrt(np.clip(self.var, self._min_var, self._max_var) + self.eps)
        if self.clip_max:
            data_array = np.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        data_array = np.asarray(data_array, dtype=np.float64)
        if data_array.ndim == 0:
            data_array = data_array[None]
        batch_mean = np.mean(data_array, axis=0)
        batch_var = np.var(data_array, axis=0)
        batch_count = int(data_array.shape[0])

        if batch_count <= 0:
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / total_count)
        new_var = m_2 / total_count

        # Safety clamps
        new_var = np.clip(new_var, self._min_var, self._max_var)

        self.mean, self.var = new_mean, new_var
        self.count = total_count


class IPPOPolicy(PPOPolicy):
    def __init__(
            self,
            logger = None,
            **kwargs
    ):
        self.logger = logger
        self._update_step = 0
        self.constraint_critic = kwargs.pop('constraint_critic', None)
        self.recurrent = kwargs.pop('recurrent', False)
        self.unroll_len = kwargs.pop('unroll_len', 1)
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
        # batch.logp_old = to_torch_as(
        #     batch.policy.logp_old[batch.obs.agent_id[0]].astype(np.float32),
        #     batch.v_s
        # )
        # batch: LogpOldProtocol
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        batch: LogpOldProtocol
        return batch

    def _compute_returns(
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
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Empirical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        # TODO: see todo in PGPolicy.process_fn
        if self.rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
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

    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TPPOTrainingStats:
        losses, clip_losses, vf_losses, ent_losses, clipfracs, approx_kls, explained_variances = [], [], [], [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                # calculate loss for actor
                advantages = minibatch.adv
                dist = self(minibatch).dist
                if self.norm_adv:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)  # per-batch norm
                act = minibatch.act
                logp = dist.log_prob(act)
                if logp.ndim > 1: logp = logp.sum(-1)

                logratio = logp - minibatch.logp_old
                ratios = logratio.exp().float()  # [b, T]
                # ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                # ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > self.eps_clip).float().mean().item()]

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs, info=minibatch.info).flatten()
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

                def get_critic_prediction(true, pred) -> float | int | Any:
                    y_pred, y_true = pred.detach().cpu().numpy(), true.detach().cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                    return explained_var

                explained_vars = get_critic_prediction(minibatch.returns, value)

                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
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
                approx_kls.append(approx_kl.item())
                explained_variances.append(explained_vars.item())
                losses.append(loss.item())

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

    @staticmethod
    def _get_agent_name(batch):
        try:
            return str(batch.obs.agent_id[0])
        except Exception:
            return batch.obs.agent_id[0]


    @staticmethod
    def _mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else float(xs)

    def _log_learn_stats(self, batch, losses, clip_losses, vf_losses, ent_losses,
                         approx_kls, clipfracs, explained_variances, additional_data: dict = None):
        if self.logger is None:
            return

        name = self._get_agent_name(batch)

        log_data = {
            f"loss/total/{name}": self._mean(losses),
            f"loss/clip/{name}": self._mean(clip_losses),
            f"loss/value/{name}": self._mean(vf_losses),
            f"loss/entropy/{name}": self._mean(ent_losses),
            f"ppo/approx_kl/{name}": self._mean(approx_kls),
            f"ppo/clipfrac/{name}": self._mean(clipfracs),
            f"ppo/explained_var/{name}": self._mean(explained_variances),
        }

        if additional_data is not None:
            log_data = {**log_data, **additional_data}

        # optional: episode returns
        try:
            rew_np = np.asarray(batch.rew)
            done_np = np.asarray(batch.done)
            ep_returns = []
            cur = 0.0
            for r, d in zip(rew_np, done_np):
                cur += r
                if d:
                    ep_returns.append(cur)
                    cur = 0.0
            if ep_returns:
                log_data[f"train/ep_return_mean/{name}"] = float(np.mean(ep_returns))
                log_data[f"train/ep_return_std/{name}"] = float(np.std(ep_returns))
        except:
            pass

        step = self._update_step
        self.logger.write("training/train", step=step, data=log_data)
        self._update_step += 1


class LagrangianIPPOPolicy(IPPOPolicy):
    def __init__(
            self,
            constraint_critic: torch.nn.Module = None,
            constraint_loss_coefficient: float = 0.5,
            initial_lagrangian_multiplier: float = 0.001,
            lagrangian_learning_rate: float = 0.0005,
            lagrangian_upper_bound: float = 3.0,
            const_norm: bool = True,
            norm_const_adv: bool = False,
            logger = None,
            recurrent: bool = False,
            unroll_len: int = 32,
            burn_in: int = 0,  # optional: use 0 to keep it simple
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.constraint_critic = constraint_critic
        self.const_rms = RunningMeanStdSafe()
        self.ret_rms = RunningMeanStdSafe()
        self.cf_coef = constraint_loss_coefficient
        self.lagrange = Lagrange(
            cost_limit=0.0,
            lagrangian_multiplier_init=initial_lagrangian_multiplier,
            lagrangian_multiplier_lr = lagrangian_learning_rate,
            lagrangian_upper_bound = lagrangian_upper_bound,
        )
        self.const_norm = const_norm
        self.norm_const_adv = norm_const_adv
        self._actor_critic = ActorCriticConstraint(self.actor, self.critic, self.constraint_critic)

        self.recurrent = recurrent
        self.unroll_len = int(unroll_len)
        self.burn_in = int(burn_in)

        self.logger = logger

    def _as_sequences(self, flat: Batch, T: int):
        """
        Convert a flat rollout Batch into sequences of length T (with padding).
        Returns: (seq_batch, h0, valid_mask, learn_mask)
          - seq_batch: Batch with fields shaped [B, T, ...]
          - h0: RecurrentStateBatch with key 'hidden' of shape [B, L, H]
          - valid_mask: [B, T] (True where real data, False for padding)
          - learn_mask: [B, T] (False on pads and optional burn-in steps)
        """
        obs = flat.obs
        aid = np.asarray(obs.agent_id)
        done = np.asarray(flat.done).astype(bool)
        if done.ndim > 1:
            done = done[..., -1]

        env_id = np.asarray(getattr(flat.info, 'env_id', flat.info["env_id"]))  # shape [N]

        # 1) Segment indices where episode or agent changes
        N = aid.shape[0]
        cuts = [0]
        for i in range(1, N):
            if done[i - 1] or (aid[i] != aid[i - 1]) or (env_id[i] != env_id[i - 1]):
                cuts.append(i)
        cuts.append(N)

        # 2) Fixed windows of length T inside each segment (non-overlapping for simplicity)
        windows = []  # (start, end)
        for s, e in zip(cuts[:-1], cuts[1:]):
            Lseg = e - s
            if Lseg <= 0:
                continue
            for off in range(0, Lseg, T):
                ws, we = s + off, min(s + off + T, e)
                assert (aid[ws:we] == aid[ws]).all(), "Window mixes agents"
                assert (env_id[ws:we] == env_id[ws]).all(), "Window mixes envs"
                windows.append((ws, we))

        B = len(windows)


        # 3) Stack/pad fields to [B, T, ...]
        def slice_field(field, slc):
            return field[slc]

        def pad_to_T(x, tlen):
            # x is torch.Tensor or np.ndarray with leading dim tlen
            if isinstance(x, torch.Tensor):
                if tlen < T:
                    pad = torch.zeros((T - tlen, *x.shape[1:]), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=0)
            else:  # np
                if tlen < T:
                    pad = np.zeros((T - tlen, *x.shape[1:]), dtype=x.dtype)
                    x = np.concatenate([x, pad], axis=0)
            return x

        # Helper: more general padding that respects various dtypes
        def pad_to_T_general(x, tlen):
            # x has leading time dim of length tlen
            if isinstance(x, torch.Tensor):
                if tlen < T:
                    pad = torch.zeros((T - tlen, *x.shape[1:]), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=0)
                return x

            # Convert lists/tuples to np.ndarray
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)

            if tlen >= T:
                return x

            pad_len = T - tlen
            lead_shape = (pad_len,)
            tail_shape = x.shape[1:]
            pad_shape = lead_shape + tail_shape

            kind = x.dtype.kind  # 'b' bool, 'i' int, 'u' uint, 'f' float, 'c' complex, 'O' object, 'U' unicode, 'S' bytes, 'M' datetime64, 'm' timedelta64

            if kind in ('b',):
                pad = np.zeros(pad_shape, dtype=x.dtype)  # False
            elif kind in ('i', 'u', 'f', 'c'):
                pad = np.zeros(pad_shape, dtype=x.dtype)  # 0
            elif kind in ('U', 'S'):
                pad = np.full(pad_shape, '', dtype=x.dtype)  # empty string
            elif kind == 'O':
                pad = np.empty(pad_shape, dtype=x.dtype)
                pad.fill(None)  # fill with None for object arrays
            elif kind in ('M', 'm'):
                # datetime64/timedelta64 NaT padding
                pad = np.empty(pad_shape, dtype=x.dtype)
                pad[...] = np.datetime64('NaT') if kind == 'M' else np.timedelta64('NaT')
            else:
                # Fallback: attempt zeros
                pad = np.zeros(pad_shape, dtype=x.dtype)

            return np.concatenate([x, pad], axis=0)

        # obs: keep Batch semantics but ensure obs.obs ends up [B, T, H]
        obs_list = []
        valid_list = []
        for ws, we in windows:
            o = slice_field(flat.obs, slice(ws, we))  # Batch of length t
            t = we - ws

            # obs: [t, H] -> [T, H]
            if torch.is_tensor(o.obs):
                o_obs = pad_to_T(o.obs, t)
            else:
                o_obs = pad_to_T(torch.as_tensor(o.obs), t)

            entry = Batch(obs=o_obs)

            # carry over agent_id if you need it later (optional)
            if hasattr(o, 'agent_id') and o.agent_id is not None:
                # agent_id is typically length-t; padding is optional (unused by nets), but safe:
                entry.agent_id = pad_to_T(o.agent_id, t)

            # IMPORTANT: carry over action mask and pad it to [T, ...]
            if hasattr(o, 'mask') and o.mask is not None:
                m = o.mask
                if not torch.is_tensor(m):
                    # preserve dtype: bool or float
                    m = torch.as_tensor(m)
                entry.mask = pad_to_T(m, t)
                # Optionally, if padded positions should be invalid, ensure zeros in the padded tail.
                # pad_to_T with zeros already accomplishes that.

            obs_list.append(entry)

            val = np.zeros((T,), dtype=bool)
            val[:t] = True
            valid_list.append(val)

        valid_mask = torch.as_tensor(np.stack(valid_list, axis=0))  # [B, T]

        # actions, advantages, returns, old logp
        def stack_time(field_name):
            xs = []
            for ws, we in windows:
                x = slice_field(getattr(flat, field_name), slice(ws, we))
                t = we - ws
                if not torch.is_tensor(x):
                    x = torch.as_tensor(
                        x, device=getattr(flat, field_name).device
                        if hasattr(getattr(flat, field_name), 'device')
                        else None
                    )
                # Squeeze trailing singleton dims for discrete fields that should be scalar per-step
                if field_name in ("act", "logp_old") and x.ndim >= 2 and x.shape[-1] == 1:
                    x = x.squeeze(-1)
                xs.append(pad_to_T_general(x, t))
            return torch.stack(xs, dim=0)  # [B, T, ...]

        act = stack_time('act')
        adv = stack_time('adv')
        ret = stack_time('returns')
        v_s = stack_time('v_s')
        logp_old = stack_time('logp_old')
        dones = stack_time('done')

        # Build info as Batch over windows without converting to tensors
        info_list = []
        for ws, we in windows:
            sub = flat.info[slice(ws, we)]
            t = we - ws
            entry = {}
            for k, v in sub.items():
                entry[k] = pad_to_T_general(v, t)
            info_list.append(Batch(entry))
        info = Batch.stack(info_list, axis=0)

        const_adv = getattr(flat, 'const_adv', None)
        const_returns = getattr(flat, 'const_returns', None)
        if const_adv is not None:
            const_adv = stack_time('const_adv')
        if const_returns is not None:
            const_returns = stack_time('const_returns')

        # 4) Initial hidden (and optional LSTM cell) state for each window from the first element’s stored state
        # Assumes collector saved pre-action state per step in flat.policy.hidden_state[agent_id],
        # with keys 'hidden' (and optionally 'cell' for LSTM).
        # Each can be one of: [N, H], [N, L, H], [N, E, L, H]

        def _get_local_env_id(self, flat: Batch, row: int) -> int:
            if 'env_id' in getattr(flat.info, '__dict__', {}) and flat.info.env_id is not None:
                return int(flat.info.env_id[row])
            # Fallback: derive from buffer and indices (works for VectorReplayBuffer)
            if hasattr(self, '_buffer') and hasattr(self, '_indices') and self._buffer is not None:
                buf = self._buffer
                idx = int(self._indices[row])
                if hasattr(buf, 'buffer_num') and buf.buffer_num > 0 and buf.maxsize % buf.buffer_num == 0:
                    per_env = buf.maxsize // buf.buffer_num
                    return idx // per_env
            # Last resort: assume single env
            return 0

        h0_list: list[torch.Tensor] = []
        c0_list: list[torch.Tensor] = []
        hs_any = None
        cs_any = None
        for ws, _ in windows:
            agent_id = aid[ws]
            st = flat.policy.hidden_state[agent_id]  # Batch or tensor-like

            # Get the hidden tensor regardless of wrapping
            if isinstance(st, Batch):
                hs = st.get('hidden', None)
                cs = st.get('cell', None)
            else:
                hs = st
                cs = None
            if hs is None:
                raise RuntimeError('policy.hidden_state[agent_id] does not contain a "hidden" tensor')

            # Hidden shapes:
            # [N, H] -> add layer dim -> [1, H]
            # [N, L, H] -> pick time row -> [L, H]
            # [N, E, L, H] -> pick env then time -> [L, H]
            if hs.ndim == 2:
                h_ws = hs[ws].unsqueeze(0)
            elif hs.ndim == 3:
                h_ws = hs[ws]
            elif hs.ndim == 4:
                env_local = _get_local_env_id(self, flat, ws)
                h_ws = hs[ws, env_local]
            else:
                raise RuntimeError(f'Unsupported hidden shape: {tuple(hs.shape)}')
            h0_list.append(h_ws)
            hs_any = hs  # remember device/dtype

            # Cell shapes mirror hidden; only if present
            if cs is not None:
                if cs.ndim == 2:
                    c_ws = cs[ws].unsqueeze(0)
                elif cs.ndim == 3:
                    c_ws = cs[ws]
                elif cs.ndim == 4:
                    env_local = _get_local_env_id(self, flat, ws)
                    c_ws = cs[ws, env_local]
                else:
                    raise RuntimeError(f'Unsupported cell shape: {tuple(cs.shape)}')
                c0_list.append(c_ws)
                cs_any = cs

        # Stack to [B, L, H]
        h0 = torch.stack(h0_list, dim=0)
        if isinstance(hs_any, torch.Tensor):
            h0 = h0.to(hs_any.device, dtype=hs_any.dtype)
        if len(c0_list) > 0:
            c0 = torch.stack(c0_list, dim=0)
            if isinstance(cs_any, torch.Tensor):
                c0 = c0.to(cs_any.device, dtype=cs_any.dtype)
        else:
            c0 = None

        seq = Batch(
            obs=Batch.stack(obs_list, axis=0),  # .obs will be [B, T, H]
            act=act,
            adv=adv,
            returns=ret,
            v_s=v_s,
            logp_old=logp_old,
            info=info,
            done=dones,
        )
        if const_adv is not None:
            seq.const_adv = const_adv
        if const_returns is not None:
            seq.const_returns = const_returns

        learn_mask = valid_mask.clone()
        if self.burn_in > 0:
            learn_mask[:, :self.burn_in] = False

        # Wrap h0 as RecurrentStateBatch (include LSTM cell if available)
        h0_fields = {"hidden": h0}
        if c0 is not None:
            h0_fields["cell"] = c0
        h0_batch = Batch(h0_fields)
        return seq, h0_batch, valid_mask, learn_mask

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
                v_s.append(self.critic(minibatch.obs, info=minibatch.info))
                v_s_.append(self.critic(minibatch.obs_next, info=minibatch.info))
                c_s.append(self.constraint_critic(minibatch.obs, info=minibatch.info))
                c_s_.append(self.constraint_critic(minibatch.obs_next, info=minibatch.info))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        batch.c_s = torch.cat(c_s, dim=0).flatten()

        # === Add safety checks for NaNs here ===
        if torch.isnan(batch.v_s).any() or torch.isnan(batch.c_s).any():
            print("NaNs detected in value or constraint predictions!")
            print("v_s:", batch.v_s)
            print("c_s:", batch.c_s)
        assert not torch.isnan(batch.v_s).any(), "NaN detected in batch.v_s"
        assert not torch.isnan(batch.c_s).any(), "NaN detected in batch.c_s"

        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()

        c_s = batch.c_s.cpu().numpy()
        c_s_ = torch.cat(c_s_, dim=0).flatten().cpu().numpy()

        # Also check after conversion to numpy, just in case
        assert not np.isnan(v_s).any(), "NaN detected in v_s (numpy)"
        assert not np.isnan(v_s_).any(), "NaN detected in v_s_ (numpy)"
        assert not np.isnan(c_s).any(), "NaN detected in c_s (numpy)"
        assert not np.isnan(c_s_).any(), "NaN detected in c_s_ (numpy)"
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
            # Guard the running variance and compute a bounded scale
            # var = np.asarray(self.const_rms.var, dtype=np.float64)
            # if not np.all(np.isfinite(var)):
            #     var = np.float64(1.0)
            # var = np.clip(var, 1e-12, 1e6)
            # scale = np.sqrt(var + float(self._eps))
            # # Bound the effective scale to avoid overflow on multiplication
            # scale = float(np.clip(scale, 1e-3, 1e3))
            # # Promote to float64 during scaling to reduce overflow, then keep as float64 for downstream np ops
            # c_s = (c_s.astype(np.float64) * scale)
            # c_s_ = (c_s_.astype(np.float64) * scale)
            # # Final magnitude clamp (acts as a last-resort safety net)
            # c_s = np.clip(c_s, -1e12, 1e12)
            # c_s_ = np.clip(c_s_, -1e12, 1e12)
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

        # Sanity guard against any remaining numerical issues
        if not np.all(np.isfinite(const_returns)):
            const_returns = np.nan_to_num(const_returns, nan=0.0, posinf=1e12, neginf=-1e12)
        if not np.all(np.isfinite(constraint_advantages)):
            constraint_advantages = np.nan_to_num(constraint_advantages, nan=0.0, posinf=1e12, neginf=-1e12)

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
        if torch.isnan(batch.adv).any() or torch.isnan(batch.const_adv).any():
            print("NaNs detected in value or constraint predictions!")
            print("adv:", batch.adv)
            print("const_adv:", batch.const_adv)
        assert not torch.isnan(batch.adv).any(), "NaN detected in batch.adv"
        assert not torch.isnan(batch.const_adv).any(), "NaN detected in batch.const_adv"
        return cast(BatchWithAdvantagesProtocol, batch)

    @staticmethod
    def compute_episodic_cost(
            batch: RolloutBatchProtocol,
            buffer: ReplayBuffer,
            indices: np.ndarray,
            c_s_: np.ndarray | torch.Tensor | None = None,
            c_s: np.ndarray | torch.Tensor | None = None,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:

        cost = _last1d(batch.info['TotalConstraint'])
        if c_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            c_s_ = np.zeros_like(cost)
        else:
            c_s_ = to_numpy(c_s_.flatten())
            c_s_ = c_s_ * BasePolicy.value_mask(buffer, indices)
        c_s = np.roll(c_s_, 1) if c_s is None else to_numpy(c_s.flatten())

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        if len(end_flag.shape) > 1:
            end_flag = end_flag[:, -1]
        advantage = _gae_return(c_s, c_s_, cost, end_flag, gamma, gae_lambda)
        returns = advantage + c_s
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

    @staticmethod
    def _get_hidden_state(
            batch: Batch,
    ):
        agent_id = batch.obs.agent_id[0]
        st = batch.policy.hidden_state[agent_id]

        # If the collector stored a Batch with keys, pass them through.
        if isinstance(st, Batch):
            h = st.get("hidden", None)
            c = st.get("cell", None)
            out_fields = {}
            if h is not None:
                out_fields["hidden"] = h
            if c is not None:
                out_fields["cell"] = c
            return Batch(out_fields) if len(out_fields) > 0 else None

        # Otherwise, assume a single tensor = GRU hidden and wrap it.
        return Batch({"hidden": st})

    def learn(  # type: ignore
            self,
            batch: RolloutBatchProtocol,
            batch_size: int | None,
            repeat: int,
            *args: Any,
            **kwargs: Any,
    ) -> TPPOTrainingStats:
        (losses, clip_losses, vf_losses, ent_losses, cf_losses, clipfracs, approx_kls,
         explained_variances, constraint_predictions) = [], [], [], [], [], [], [], [], []
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

        self.lagrange.update_lagrange_multiplier(mean_ep_constraint_values)

        lagrangian_multiplier = float(self.lagrange.lagrangian_multiplier)

        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            if not self.recurrent:
                gradient_steps, clip_losses, vf_losses, ent_losses, cf_losses, approx_kls, explained_variances, constraint_predictions, losses = self.flat_learn_ppo(
                    batch, cf_losses, clip_losses, ent_losses, gradient_steps,
                    lagrangian_multiplier, losses, split_batch_size, vf_losses, clipfracs, approx_kls, explained_variances,
                constraint_predictions)
            else:
                # Recurrent sequence path
                # T = self.burn_in + self.unroll_len
                # seq_batch, h0, valid_mask, learn_mask = self._as_sequences(batch, T)
                T = self.burn_in + self.unroll_len
                seq_batch, h0_collected, valid_mask, learn_mask = self._as_sequences(batch, T)

                # Zero-init h0 to avoid stale behavior-state; we will rebuild under *current* weights.
                # Keep device/dtype from collected tensors.
                h0_fields = {}
                hid0 = h0_collected.hidden
                h0_fields["hidden"] = torch.zeros_like(hid0)
                if hasattr(h0_collected, "cell") and getattr(h0_collected, "cell") is not None:
                    cel0 = h0_collected.cell
                    h0_fields["cell"] = torch.zeros_like(cel0)
                h0 = Batch(h0_fields)
                # h0 = h0_collected
                B = seq_batch.adv.shape[0]
                bs = batch_size or B

                # loop whole batch by slicing batch size
                for s in range(0, B, bs):
                    gradient_steps += 1

                    # get end index
                    e = min(s + bs, B)

                    # slice batch to get minibatch
                    mb = seq_batch[s:e]
                    mb_valid = valid_mask[s:e]

                    # mask for learning
                    mb_learn = learn_mask[s:e]

                    # get fields for hidden states
                    mb_h0_fields = {"hidden": h0.hidden[s:e]}
                    if hasattr(h0, "cell") and getattr(h0, "cell") is not None:
                        mb_h0_fields["cell"] = h0.cell[s:e]
                    # mb_h0 = Batch(mb_h0_fields)

                    mb.obs['detach_state'] = False


                    # Forward the whole [B,T,H] sequence once. Start from mb_h0 for this window.
                    out_seq = self(Batch(obs=mb.obs, info=mb.info), state=None)
                    dist_seq = out_seq.dist
                    # state = out_seq.state  # carry the updated hidden state if you need it later

                    # Compute log-prob and entropy for ALL steps at once
                    act_bt = mb.act  # may be [B,T] or [B,T,1] for discrete
                    if act_bt.ndim == 3 and act_bt.shape[-1] == 1:
                        act_bt = act_bt.squeeze(-1)
                    logp = dist_seq.log_prob(act_bt)
                    ent = dist_seq.entropy()

                    # If action space has extra dims, sum across the last dim
                    if logp.ndim > 2:
                        logp = logp.sum(dim=-1)
                    if ent.ndim > 2:
                        ent = ent.sum(dim=-1)

                    # Ensure [B,T] layout
                    bsz, tlen = mb.obs.obs.shape[:2]
                    if logp.ndim == 1:
                        logp = logp.view(bsz, tlen)
                    if ent.ndim == 1:
                        ent = ent.view(bsz, tlen)

                    # logratio = logp - mb.logp_old
                    logp_old_bt = mb.logp_old
                    if logp_old_bt.ndim == 3 and logp_old_bt.shape[-1] == 1:
                        logp_old_bt = logp_old_bt.squeeze(-1)
                    logratio = logp - logp_old_bt
                    ratios = logratio.exp().float()  # [b, T]

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratios - 1) - logratio).mean()
                        clipfracs += [((ratios - 1.0).abs() > self.eps_clip).float().mean().item()]

                    adv = mb.adv
                    if self.norm_adv:
                        mean, std = adv[mb_learn].mean(), adv[mb_learn].std()
                        adv = (adv - mean) / (std + self._eps)

                    if hasattr(mb, 'const_adv'):
                        cadv = mb.const_adv
                        if self.norm_const_adv:
                            cmean, cstd = cadv[mb_learn].mean(), cadv[mb_learn].std()
                            cadv = (cadv - cmean) / (cstd + self._eps)
                        combined_adv = adv - float(self.lagrange.lagrangian_multiplier) * cadv
                        combined_adv /= (self.lagrange.lagrangian_multiplier + 1)
                    else:
                        combined_adv = adv

                    surr1 = ratios * combined_adv
                    surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * combined_adv
                    if self.dual_clip:
                        clip1 = torch.minimum(surr1, surr2)
                        clip2 = torch.maximum(clip1, self.dual_clip * combined_adv)
                        pg = -torch.where(combined_adv < 0, clip2, clip1)
                    else:
                        pg = -torch.minimum(surr1, surr2)

                    # Critic values: if critic is non-recurrent, flatten [b, T, ...] -> [b*T, ...]
                    # mb.obs.obs: [b, T, H]
                    flat_obs = mb.obs.obs.reshape(-1, mb.obs.obs.shape[-1])
                    # mask_from_obs = mb.obs.mask.reshape(-1, mb.obs.mask.shape[-1])
                    # flat_info = mb.info.reshape(-1, mb.info.shape[-1])
                    # v = self.critic(Batch(obs=flat_obs, mask=mask_from_obs), info=Batch(info=flat_info)).reshape(mb.returns.shape)  # [b, T]
                    # Robust flatten for mask (2D or >2D)
                    if mb.obs.mask.ndim >= 2:
                        mask_from_obs = mb.obs.mask.reshape(-1, *mb.obs.mask.shape[2:])
                    else:
                        mask_from_obs = mb.obs.mask.reshape(-1)

                    # helper: flatten a Batch of info fields from [B, T, ...] -> [B*T, ...] per key
                    def _flatten_info_batch(info_batch: Batch) -> Batch:
                        flat_dict = {}
                        for k, v in info_batch.items():
                            if k in ['CropName', 'Date']:
                                continue
                            t = v if torch.is_tensor(v) else torch.as_tensor(v)
                            if t.ndim >= 2:
                                flat_dict[k] = t.reshape(-1, *t.shape[2:])
                            else:
                                flat_dict[k] = t.reshape(-1)
                        return Batch(flat_dict)

                    flat_info = _flatten_info_batch(mb.info)
                    v = self.critic(Batch(obs=flat_obs, mask=mask_from_obs), info=flat_info).reshape(mb.returns.shape)  # [b, T]

                    if self.value_clip:
                        v_clip = mb.v_s + (v - mb.v_s).clamp(-self.eps_clip, self.eps_clip)
                        vf1 = (mb.returns - v).pow(2)
                        vf2 = (mb.returns - v_clip).pow(2)
                        vfloss = torch.maximum(vf1, vf2)
                    else:
                        vfloss = (mb.returns - v).pow(2)

                    # Constraint critic similarly
                    if self.constraint_critic is not None and hasattr(mb, 'const_returns'):
                        cv = self.constraint_critic(Batch(obs=flat_obs, mask=mask_from_obs), info=flat_info).reshape(
                            mb.const_returns.shape)
                        cfloss = (mb.const_returns - cv).pow(2)
                    else:
                        cfloss = None

                    def get_critic_prediction(true, pred) -> float | int | Any:
                        y_pred, y_true = pred.detach().cpu().numpy(), true.detach().cpu().numpy()
                        var_y = np.var(y_true)
                        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                        return explained_var

                    explained_var = get_critic_prediction(mb.returns, v)
                    constraint_prediction = get_critic_prediction(mb.const_returns, cv)

                    # Masked reductions
                    def masked_mean(x):
                        m = mb_learn
                        return (x * m).sum() / m.sum().clamp_min(1)

                    clip_loss = masked_mean(pg)
                    vf_loss = masked_mean(vfloss)
                    ent_loss = masked_mean(ent)
                    cf_loss = masked_mean(cfloss) if isinstance(cfloss, torch.Tensor) else torch.tensor(0.0,
                                                                                                        device=ent.device)
                    approx_kl = masked_mean(approx_kl)
                    explained_vars = masked_mean(explained_var)
                    constraint_prediction = masked_mean(constraint_prediction)

                    loss = clip_loss + self.vf_coef * vf_loss + self.cf_coef * cf_loss - self.ent_coef * ent_loss
                    self.optim.zero_grad()
                    loss.backward()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self.max_grad_norm)
                    self.optim.step()
                    clip_losses.append(clip_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(ent_loss.item())
                    cf_losses.append(cf_loss.item())
                    approx_kls.append(approx_kl.item())
                    explained_variances.append(explained_vars.item())
                    constraint_predictions.append(constraint_prediction.item())
                    losses.append(loss.item())

        stats = LagIPPOTrainingStats.from_sequences(  # type: ignore[return-value]
            losses=losses,
            clip_losses=clip_losses,
            vf_losses=vf_losses,
            cf_losses=cf_losses,
            ent_losses=ent_losses,
            approx_kl=approx_kls,
            clipfrac=clipfracs,
            explained_variance=explained_variances,
            constraint_prediction=constraint_predictions,
            gradient_steps=gradient_steps,
        )

        name = self._get_agent_name(batch)
        more_data = {
            f"loss/constraint/{name}": self._mean(cf_losses),
            f"cost/total_cost/{name}": self._mean(batch.info["TotalConstraint"]),
            f"cost/lagrangian_multiplier/{name}": self.lagrange.lagrangian_multiplier,
            f"ppo/constraint_prediction/{name}": self._mean(constraint_predictions),
        }
        self._log_learn_stats(batch, losses, clip_losses, vf_losses, ent_losses, approx_kls,
                              clipfracs, explained_variances, additional_data=more_data)

        return stats

    def flat_learn_ppo(self, batch: RolloutBatchProtocol | BatchWithAdvantagesProtocol, cf_losses: list[Any],
                       clip_losses: list[Any], ent_losses: list[Any], gradient_steps: int, lagrangian_multiplier: float,
                       losses: list[Any], split_batch_size: int | None, vf_losses: list[Any], clipfracs: list, approx_kls: list,
                       explained_variances: list, constraint_predictions: list) -> tuple:
        for minibatch in batch.split(split_batch_size, merge_last=True, shuffle=True if self.recurrent else False):
            gradient_steps += 1
            # calculate loss for actor
            advantages = minibatch.adv

            constraint_advantages = minibatch.const_adv

            # learn
            dist = self(minibatch).dist
            if self.norm_adv:
                mean, std = advantages.mean(), advantages.std()
                advantages = (advantages - mean) / (std + self._eps)  # per-batch norm
            if self.norm_const_adv:
                const_mean, const_std = constraint_advantages.mean(), constraint_advantages.std()
                constraint_advantages = (constraint_advantages - const_mean) / (const_std + self._eps)

            # start lagrangian constraint
            combined_advantages = advantages - lagrangian_multiplier * constraint_advantages
            combined_advantages /= (lagrangian_multiplier + 1)

            act = minibatch.act
            logp = dist.log_prob(act)
            if logp.ndim > 1: logp = logp.sum(-1)

            logratio = logp - minibatch.logp_old
            ratios = logratio.exp().float()  # [b, T]

            # ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # old_approx_kl = (-logratio).mean()
                approx_kl = ((ratios - 1) - logratio).mean()
                clipfracs += [((ratios - 1.0).abs() > self.eps_clip).float().mean().item()]

            surr1 = ratios * combined_advantages
            surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * combined_advantages
            if self.dual_clip:
                clip1 = torch.min(surr1, surr2)
                clip2 = torch.max(clip1, self.dual_clip * combined_advantages)
                clip_loss = -torch.where(combined_advantages < 0, clip2, clip1).mean()
            else:
                clip_loss = -torch.min(surr1, surr2).mean()

            # calculate loss for critic
            value = self.critic(minibatch.obs, info=minibatch.info).flatten()
            constraint_value = self.constraint_critic(minibatch.obs, info=minibatch.info).flatten()

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

            def get_critic_prediction(true, pred) -> float | int | Any:
                y_pred, y_true = pred.detach().cpu().numpy(), true.detach().cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                return explained_var

            explained_vars = get_critic_prediction(minibatch.returns, value)
            constraint_prediction = get_critic_prediction(minibatch.const_returns, constraint_value)

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
            approx_kls.append(approx_kl.item())
            explained_variances.append(explained_vars.item())
            constraint_predictions.append(constraint_prediction.item())
            losses.append(loss.item())
        return gradient_steps, clip_losses, vf_losses, ent_losses, cf_losses, approx_kls, explained_variances, constraint_predictions, losses


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
    approx_kl: SequenceSummaryStats
    clipfrac: SequenceSummaryStats
    explained_variance: SequenceSummaryStats

    @classmethod
    def from_sequences(
        cls,
        *,
        losses: Sequence[float],
        clip_losses: Sequence[float],
        vf_losses: Sequence[float],
        ent_losses: Sequence[float],
        approx_kl: Sequence[float],
        clipfrac: Sequence[float],
        explained_variance: Sequence[float],
        gradient_steps: int = 0,
    ) -> Self:
        return cls(
            loss=SequenceSummaryStats.from_sequence(losses),
            clip_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            approx_kl=SequenceSummaryStats.from_sequence(approx_kl),
            clipfrac=SequenceSummaryStats.from_sequence(clipfrac),
            explained_variance=SequenceSummaryStats.from_sequence(explained_variance),
            gradient_steps=gradient_steps,
        )


@dataclass(kw_only=True)
class LagIPPOTrainingStats(IPPOTrainingStats):
    cf_loss: SequenceSummaryStats
    constraint_prediction: SequenceSummaryStats

    @classmethod
    def from_sequences(
        cls,
        *,
        losses: Sequence[float],
        clip_losses: Sequence[float],
        vf_losses: Sequence[float],
        cf_losses: Sequence[float],
        ent_losses: Sequence[float],
        approx_kl: Sequence[float],
        clipfrac: Sequence[float],
        explained_variance: Sequence[float],
        constraint_prediction: Sequence[float],
        gradient_steps: int = 0,
    ) -> Self:
        return cls(
            loss=SequenceSummaryStats.from_sequence(losses),
            clip_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            cf_loss=SequenceSummaryStats.from_sequence(cf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            approx_kl=SequenceSummaryStats.from_sequence(approx_kl),
            clipfrac=SequenceSummaryStats.from_sequence(clipfrac),
            explained_variance=SequenceSummaryStats.from_sequence(explained_variance),
            constraint_prediction=SequenceSummaryStats.from_sequence(constraint_prediction),
            gradient_steps=gradient_steps,
        )
