import os
from datetime import datetime

from comet_ml import Experiment

from cropgym import _BASE_PATH, _SOURCE_PATH


def _setup_bandit_comet(args, region=None, farm_id=None):
    if not os.path.isdir(os.path.join(_BASE_PATH, "comet")):
        print("Not using comet!")
        return

    with open(os.path.join(_BASE_PATH, "comet", "api"), "r") as f:
        api_key = f.readline()

    experiment = Experiment(
        api_key=api_key,
        project_name="cropgym_allocation_experiments_paper",
        workspace="cropgym",
        log_code=True,
        auto_metric_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_param_logging=True,
        auto_histogram_tensorboard_logging=True,
    )

    experiment.log_code(folder=_SOURCE_PATH)

    baseline = getattr(args, "baseline", getattr(args, "model_dir", "ROT"))
    name = f"Bandit_{baseline}_{args.method}_{datetime.now():%m%d}"
    if region is not None:
        name = f"Bandit_{baseline}_{region}_{farm_id}_{datetime.now():%m%d}"
    experiment.set_name(name)
    experiment.log_parameters({k: v for k, v in vars(args).items()})
    return experiment
