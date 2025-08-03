import torch
import numpy as np

from tianshou.data import to_torch_as
from tianshou.policy.modelfree.ppo import PPOPolicy
from tianshou.data.types import LogpOldProtocol

class IPPOPolicy(PPOPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def process_fn(self, batch, buffer, indices):
        if self.recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        # build per-step done that includes agent deaths
        if "Alive" in batch.info:
            done = batch.info["Alive"] == False
            batch.done = done
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        batch: LogpOldProtocol
        return batch