import argparse
import os


from typing import Any

import numpy as np
import pandas as pd
import torch

from tianshou.data import Collector, Batch
from tianshou.env import BaseVectorEnv
from tianshou.policy import MultiAgentPolicyManager, BasePolicy

from cropgymzoo.envs.wrappers_tianshou import MultiAgentVecNormObs
from cropgymzoo.envs.multi_field_env import MultiFieldEnv
from cropgymzoo.utils.eval_tianshou import evaluate_policy


def yearly_eval_test_fn(
        epoch,
        raw_env: MultiFieldEnv,
        policy_mgr: MultiAgentPolicyManager,
        train_env: BaseVectorEnv,
        agents,
        logger,
        args
):

    reset_options_list = [
        year for year in range(2010, 2011)
    ]
    # get writer
    writer = logger.writer

    # align normalizer
    obs_rms = train_env.get_obs_rms()

    info_dict = {}
    for i, year in enumerate(reset_options_list):
        info_dict[year] = {}

        next_states = {
            ag: None for ag in agents
        }

        raw_env.reset(year)

        for agent in raw_env.agent_iter():
            obs, rew, term, trunc, info = raw_env.last()

            # get appropriate info shape for policy
            processed_info = Batch({k: [i[-1]] for k, i in info.items()})
            processed_info['env_id'] = [0]

            with torch.no_grad():
                out = policy_mgr.policies[agent](
                    batch = Batch(
                        {
                            'obs': {
                                'obs': obs_rms.norm(obs['observation']),
                                'mask': raw_env._get_mask(agent),
                            },
                            'info': processed_info,
                        }
                    ),
                    state=Batch(next_states[agent]),
                )

            action = out.act.item()
            state = None if not hasattr(out, 'state') else out.state

            next_states[agent] = state

            if raw_env.terminations[agent]:
                info_dict[year][agent] = info  # grab info before agent dies
                raw_env.step(None)
            else:
                raw_env.step(action)

        # log results to tensorboard
        across_years_reward = {}
        for year, agent_info in info_dict.items():
            across_years_reward[year] = []
            reward_year = []
            for a_id, full_info in agent_info.items():
                agent_reward = np.sum(full_info['Reward'])
                agent_nue = full_info['Nue'][-1]
                agent_nsurp = full_info['Nsurp'][-1]
                agent_budget_left = full_info['BudgetLeft'][-1]
                agent_yield = full_info['Yield'][-1]
                agent_n_action = full_info['Naction'][-1]

                # put into year reward
                reward_year.append(agent_reward)

                if writer:
                    writer.add_scalar(f"test/{year}/{a_id}/Reward", agent_reward, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/NUE", agent_nue, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/Nsurp", agent_nsurp, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/BudgetLeft", agent_budget_left, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/Yield", agent_yield, epoch)
                    writer.add_scalar(f"test/{year}/{a_id}/Naction", agent_n_action, epoch)
            else:
                across_years_reward[year].append(np.sum(reward_year))
                # Logging intermediate results
                if writer:
                    writer.add_scalar(f"test/{year}/total_reward", np.sum(reward_year), epoch)
        else:
            # Final aggregated logging
            mean_reward = np.mean(list(across_years_reward.values()))

            if writer:
                writer.add_scalar("test/mean_reward_all_years", mean_reward, epoch)

    writer.flush()

def save_checkpoint_fn(
        epoch: int,
        env_step: int,
        grad_step: int,
        run_name: str,
        train_envs: MultiAgentVecNormObs,
        test_envs: MultiAgentVecNormObs,
        policy_mgr: MultiAgentPolicyManager,
        args: argparse.Namespace,
) -> None | str:
    # copy running statistics into the frozen eval envs *once per epoch*
    test_envs.set_obs_rms(train_envs.get_obs_rms())
    if epoch % 20 == 0:
        torch.save(
            {
                "model": {
                    aid: p.state_dict()  # one file for every agent
                    for aid, p in policy_mgr.policies.items()
                },
                "obs_rms": train_envs.get_obs_rms(),
            },
            os.path.join(args.logdir, run_name, "checkpoints", f"check_{epoch:04d}.pth")
        )

def save_best_fn(
        ma_policy: MultiAgentPolicyManager,
        train_envs: MultiAgentVecNormObs,
        run_name: str,
        args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "models": {
                aid: p.state_dict()  # one file for every agent
                for aid, p in ma_policy.policies.items()
            },
            "obs_rms": train_envs.get_obs_rms(),
        },
        os.path.join(args.logdir, run_name, "best", "best.pth")
    )