import argparse
import os
import glob
import re

from pathlib import Path

import json

import comet_ml
from comet_ml import Experiment

from typing import Any

import numpy as np
import torch

from tianshou.data import Collector, Batch
from tianshou.env import BaseVectorEnv
from tianshou.policy import MultiAgentPolicyManager, BasePolicy
from tianshou.utils.logger.base import BaseLogger

from cropgymzoo import _SOURCE_PATH, _BASE_PATH
from cropgymzoo.envs.wrappers_tianshou import MultiAgentVecNormObs
from cropgymzoo.envs.multi_field_env import MultiFieldEnv


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
    if hasattr(logger, 'writer'):
        writer = logger.writer
    elif hasattr(logger, 'loggers'):
        writer = logger.loggers[0].writer

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

def marl_save_checkpoint_fn(
        epoch: int,
        env_step: int,
        grad_step: int,
        run_name: str,
        train_envs: MultiAgentVecNormObs,
        test_envs: MultiAgentVecNormObs,
        policy_mgr: MultiAgentPolicyManager,
        args: argparse.Namespace,
        log_every_epochs: int=5,
        experiment: Experiment=None,
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

    if experiment is not None and epoch % log_every_epochs == 0:
        log_weights_and_grads_marl(experiment, policy_mgr, step=grad_step, log_grads=False)


def log_weights_and_grads_marl(experiment, model, step: int, log_grads: bool = True):
    for aid, policy in model.policies.items():
        modules = []
        for attr in ("actor", "critic"):  # cover common layouts
            if hasattr(policy, attr) and getattr(policy, attr) is not None:
                modules.append((attr, getattr(policy, attr)))

        # Log histograms (prefix by agent + role)
        for role, module in modules:
            for pname, p in module.named_parameters():
                experiment.log_histogram_3d(
                    p.detach().cpu().numpy(),
                    name=f"{aid}/{role}/weights/{pname}",
                    step=step,
                )
                if log_grads and (p.grad is not None):
                    experiment.log_metric(
                        f"{aid}/{role}/grads/norm/{pname}",
                        p.grad.detach().data.norm().item(),
                        step=step,
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

def create_comet_experiment(
        name: str,
        args: argparse.Namespace,
):
    if not os.path.isdir(os.path.join(_BASE_PATH, 'comet')):
        print("Not using comet!")
        return

    with open(os.path.join(_BASE_PATH, 'comet', 'api'), 'r') as f:
        api_key = f.readline()
    comet_experiment = Experiment(
        api_key=api_key,
        project_name='cropgymzoo_policy_experiments',
        workspace="cropgymzoo",
        log_code=True,
        auto_metric_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_param_logging=True,
        auto_histogram_tensorboard_logging=True
    )

    comet_experiment.log_code(folder=_SOURCE_PATH)
    comet_experiment.set_name(name)

    args_dict = vars(args)
    comet_experiment.log_parameters(args_dict)

    print(f'Using comet! Logged to {name}')

    return comet_experiment


class CometTianshouLogger(BaseLogger):
    """
    Minimal Comet logger that matches Tianshou's BaseLogger API.
    - Logs numeric scalars to Comet via `log_metrics`.
    - Optionally uploads checkpoints in `save_data`.
    - Persists resume metadata to `log_dir` so `restore_data` works offline.
    """

    def __init__(
        self,
        experiment: Experiment,
        log_dir: str | os.PathLike | None = None,
        train_interval: int = 2000,
        test_interval: int = 1,
        update_interval: int = 2000,
        info_interval: int = 1,
        exclude_arrays: bool = True,
        upload_checkpoints: bool = True,
        checkpoint_glob: str = "checkpoint*.pth",
    ) -> None:
        super().__init__(
            train_interval=train_interval,
            test_interval=test_interval,
            update_interval=update_interval,
            info_interval=info_interval,
            exclude_arrays=exclude_arrays,
        )
        self.experiment = experiment
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.upload_checkpoints = upload_checkpoints
        self.checkpoint_glob = checkpoint_glob

        name = self.experiment.get_name()
        self.dir_best = os.path.join(self.log_dir, name, "best")
        self.dir_checkpoint = os.path.join(self.log_dir, name, "checkpoints")

    # -------- BaseLogger required methods --------

    def prepare_dict_for_logging(self, log_data: dict) -> dict[str, Any]:
        """Keep only scalar-ish numerics (drop arrays/tensors if exclude_arrays=True)."""
        prepared: dict[str, Any] = {}
        for k, v in log_data.items():
            if _is_scalar(v):
                prepared[k] = _to_py_scalar(v)
            else:
                if not self.exclude_arrays:
                    # reduce arrays/tensors to a mean
                    if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                        prepared[k + "/mean"] = _to_py_scalar(v)
                    # else: ignore non-numeric or complex types
        return prepared

    def write(self, step_type: str, step: int, data: dict[str, Any]) -> None:
        # Prefix keys with the step_type namespace (e.g., "train/env_step")
        namespaced = {f"{step_type}/{k}": v for k, v in data.items()}
        if namespaced:
            # log in one batch to keep steps aligned
            self.experiment.log_metrics(namespaced, step=step)



    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn=None,
    ) -> None:
        # Persist resume metadata locally (optional but useful)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            meta_path = self.log_dir / "resume.json"
            with meta_path.open("w") as f:
                json.dump(
                    {"epoch": int(epoch), "env_step": int(env_step), "gradient_step": int(gradient_step)},
                    f,
                )

        # Run user save hook (returns path usually) and optionally upload to Comet
        if save_checkpoint_fn is not None:
            ckpt_path = save_checkpoint_fn(epoch, env_step, gradient_step)
            if self.upload_checkpoints and ckpt_path and os.path.exists(ckpt_path):
                self.experiment.log_asset(ckpt_path, file_name=os.path.basename(ckpt_path))
        else:
            # If user already saved to log_dir, optionally upload the newest match
            if self.upload_checkpoints and self.log_dir is not None:
                for p in sorted(self.log_dir.glob(self.checkpoint_glob)):
                    self.experiment.log_asset(str(p), file_name=p.name)

        if self.dir_best:
            self.experiment.log_asset(
                os.path.join(self.dir_best, "best.pth"),
                file_name=f'best_epoch_{epoch}.pth',
            )

        # Also log these as metrics so they’re visible in the UI at this moment
        self.experiment.log_metrics(
            {"info/epoch": epoch, "info/env_step": env_step, "info/gradient_step": gradient_step},
            step=env_step,
        )

    def restore_data(self) -> tuple[int, int, int]:
        """Try to read resume metadata written in save_data; fall back to zeros."""
        if self.log_dir is None:
            return 0, 0, 0
        meta_path = self.log_dir / "resume.json"
        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
            return int(meta.get("epoch", 0)), int(meta.get("env_step", 0)), int(meta.get("gradient_step", 0))
        except Exception:
            return 0, 0, 0

    @staticmethod
    def restore_logged_data(log_path: str) -> dict:
        """Offline: read the saved resume.json (if present)."""
        meta_path = Path(log_path) / "resume.json"
        try:
            with meta_path.open("r") as f:
                return json.load(f)
        except Exception:
            return {}

    def finalize(self) -> None:
        try:
            # flush any buffered metrics
            self.experiment.end()
        except Exception:
            pass


def _is_scalar(x: Any) -> bool:
    if isinstance(x, (bool, int, float, np.number)):
        return True
    if isinstance(x, torch.Tensor) and x.ndim == 0:
        return True
    return False

def _to_py_scalar(x: Any) -> float | int | bool:
    if isinstance(x, torch.Tensor):
        return x.item() if x.ndim == 0 else float(x.detach().mean().cpu().numpy())
    if isinstance(x, np.ndarray):
        return float(x.mean())
    if isinstance(x, np.generic):
        return x.item()
    return x  # already py scalar


class MultiLogger(BaseLogger):
    def __init__(self, *loggers: BaseLogger):
        # pick conservative intervals (min) so nothing is skipped
        super().__init__(
            train_interval=min(l.train_interval for l in loggers),
            test_interval=min(l.test_interval for l in loggers),
            update_interval=min(l.update_interval for l in loggers),
            info_interval=min(l.info_interval for l in loggers),
            exclude_arrays=all(l.exclude_arrays for l in loggers),
        )
        self.loggers: tuple[BaseLogger, ...] = tuple(loggers)

    # We forward original data so each child logger can prepare it its own way.
    def log_train_data(self, log_data: dict, step: int) -> None:
        for lg in self.loggers:
            lg.log_train_data(log_data, step)

    def log_test_data(self, log_data: dict, step: int) -> None:
        for lg in self.loggers:
            lg.log_test_data(log_data, step)

    def log_update_data(self, log_data: dict, step: int) -> None:
        for lg in self.loggers:
            lg.log_update_data(log_data, step)

    def log_info_data(self, log_data: dict, step: int) -> None:
        for lg in self.loggers:
            lg.log_info_data(log_data, step)

    # Abstracts we still need to satisfy (not used directly because we override the log_* above)
    def prepare_dict_for_logging(self, log_data: dict) -> dict:
        return log_data

    def write(self, step_type: str, step: int, data: dict) -> None:
        for lg in self.loggers:
            lg.write(step_type, step, data)

    def save_data(self, epoch: int, env_step: int, gradient_step: int, save_checkpoint_fn=None) -> None:
        for lg in self.loggers:
            lg.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)

    def restore_data(self) -> tuple[int, int, int]:
        # Return the max across children (best-effort)
        vals = [lg.restore_data() for lg in self.loggers]
        if not vals:
            return (0, 0, 0)
        epoch = max(v[0] for v in vals)
        env_step = max(v[1] for v in vals)
        grad_step = max(v[2] for v in vals)
        return (epoch, env_step, grad_step)

    @staticmethod
    def restore_logged_data(log_path: str) -> dict:
        # No unified on-disk format; return empty (or implement your own merger if you like)
        return {}

    def finalize(self) -> None:
        for lg in self.loggers:
            lg.finalize()

def get_checkpoint(path):
    files = glob.glob(os.path.join(path, "check_*.pth"))

    if files:
        # Extract the number from the filename using regex
        def extract_num(fname):
            match = re.search(r"check_(\d+)\.pth", os.path.basename(fname))
            return int(match.group(1)) if match else -1

        latest_file = max(files, key=extract_num)
        print("Latest file:", latest_file)
    else:
        latest_file = None
        print("No matching files found.")

    return latest_file