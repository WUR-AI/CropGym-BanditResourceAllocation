import argparse
import os

import numpy as np
import pandas as pd
import torch

from tianshou.data import Collector
from tianshou.env import BaseVectorEnv
from tianshou.policy import MultiAgentPolicyManager

from cropgymzoo.envs.wrappers_tianshou import MultiAgentVecNormObs

def yearly_eval_test_fn(epoch, test_collector: Collector, train_env: BaseVectorEnv, agents, logger, args):
    test_results = {}
    year_rewards = []

    reset_options_list = [
        {'year': year} for year in range(2010, 2011)
    ]

    dfs = []
    writer = logger.writer

    test_collector.env.set_obs_rms(train_env.get_obs_rms())
    # per year eval
    for i, reset_opts in enumerate(reset_options_list):
        year = reset_opts["year"]

        # Collect test episode(s)
        result = test_collector.collect(
            n_episode=1,
            reset_before_collect=True,
            gym_reset_kwargs={
                'options': reset_opts
            },
        )

        infos = test_collector.buffer._meta.info
        obs = test_collector.buffer._meta.obs
        obs_next = test_collector.buffer._meta.obs_next
        rew = test_collector.buffer._meta.rew

        agent_ids = obs_next["agent_id"]

        agent_dict = {}

        if args.debug:
            df = pd.DataFrame(index=agent_ids,
                              data={
                                  'nue': infos["Nue"],
                                  'Nsurp': infos["Nsurp"],
                                  'BudgetLeft': infos["BudgetLeft"],
                                  'action': infos["Action"],
                                  'Yield': infos["Yield"]}
                              )

            dfs.append(df)

        for a, a_id in enumerate(agents):
            agent = agent_ids == a_id

            reward = [r[a] for r in rew[agent]]
            nue = infos["Nue"][agent]
            nsurp = infos["Nsurp"][agent]
            budget_left = infos["BudgetLeft"][agent]
            yld = infos["Yield"][agent]
            n_action = infos["Action"][agent]

            agent_reward = np.sum(reward)
            agent_nue = nue[-1]
            agent_nsurp = nsurp[-1]
            agent_budget_left = budget_left[-1]
            agent_yield = yld[-1]
            agent_n_action = np.sum(n_action)

            agent_dict[a_id] = {
                "Reward": agent_reward,
                "Nue": agent_nue,
                "Nsurp": agent_nsurp,
                "BudgetLeft": agent_budget_left,
                "Yield": agent_yield,
                "Naction": agent_n_action,
            }

            if writer:
                writer.add_scalar(f"test/{year}/{a_id}/reward", agent_reward, epoch)
                writer.add_scalar(f"test/{year}/{a_id}/NUE", agent_nue, epoch)
                writer.add_scalar(f"test/{year}/{a_id}/Nsurp", agent_nsurp, epoch)
                writer.add_scalar(f"test/{year}/{a_id}/BudgetLeft", agent_budget_left, epoch)
                writer.add_scalar(f"test/{year}/{a_id}/Yield", agent_yield, epoch)
                writer.add_scalar(f"test/{year}/{a_id}/Naction", agent_n_action, epoch)

        # Store results with metadata
        test_results[year] = agent_dict
        year_reward = np.sum([v
                              for y in test_results.values()
                              for field in y.values()
                              for key, v in field.items()
                              if key == "Reward"])
        year_rewards.append(year_reward)

        # Logging intermediate results
        if writer:
            writer.add_scalar(f"test/{year}/reward", year_reward, epoch)

    # Final aggregated logging
    mean_reward = np.mean(year_rewards)
    #
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