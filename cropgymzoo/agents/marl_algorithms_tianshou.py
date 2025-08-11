import torch
import numpy as np

from numba import njit

from tianshou.data import to_torch_as, to_numpy, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.policy.modelfree.ppo import PPOPolicy
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol

class IPPOPolicy(PPOPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def process_fn(self, batch, buffer, indices):
        if self.recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        # build per-step done that includes agent deaths
        if "Alive" in batch.info:
            done  = batch.info["Alive"] == False
            term = batch.info["Alive"] == False
            batch.done = done
            batch.terminated = term
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        batch: LogpOldProtocol
        return batch

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
        r"""Compute returns over given batch.

        Use Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
        to calculate q/advantage value of given batch. Returns are calculated as
        advantage + value, which is exactly equivalent to using :math:`TD(\lambda)`
        for estimating returns.

        Setting `v_s_` and `v_s` to None (or all zeros) and `gae_lambda` to 1.0 calculates the
        discounted return-to-go/ Monte-Carlo return.

        :param batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param buffer: the corresponding replay buffer.
        :param indices: tells the batch's location in buffer, batch is equal
            to buffer[indices].
        :param v_s_: the value function of all next states :math:`V(s')`.
            If None, it will be set to an array of 0.
        :param v_s: the value function of all current states :math:`V(s)`. If None,
            it is set based upon `v_s_` rolled by 1.
        :param gamma: the discount factor, should be in [0, 1].
        :param gae_lambda: the parameter for Generalized Advantage Estimation,
            should be in [0, 1].

        :return: two numpy arrays (returns, advantage) with each shape (bsz, ).
        """
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
        advantage = _modified_gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage

@njit
def _modified_gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    # mask for non-terminal transitions at time t
    nonterm = 1.0 - end_flag.astype(np.float32)

    # mask V_{t+1} at terminals
    delta = rew + gamma * nonterm * v_s_ - v_s

    adv = np.zeros_like(rew, dtype=np.float32)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + gamma * gae_lambda * nonterm[i] * gae
        adv[i] = gae
    return adv

    # returns = np.zeros(rew.shape)
    # delta = rew + v_s_ * gamma - v_s
    # discount = (1.0 - end_flag) * (gamma * gae_lambda)
    # gae = 0.0
    # for i in range(len(rew) - 1, -1, -1):
    #     gae = delta[i] + discount[i] * gae
    #     returns[i] = gae
    # return returns