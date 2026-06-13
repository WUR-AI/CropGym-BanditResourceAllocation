# CropGym Bandit Resource Allocation

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-4B32C3.svg)](https://docs.astral.sh/uv/)

Research code for nitrogen resource allocation experiments with CropGym crop simulations, WOFOST/PCSE model dynamics, LP baselines, and neural adaptive Gaussian-process bandit allocators.

The Python package is named `cropgym`.

## Project Layout

```text
.
├── cropgym/                  # Importable Python package
│   ├── envs/                 # Crop simulation and allocation environments
│   ├── agents/               # Bandit models and neural-network utilities
│   ├── utils/                # Rewards, plotting, callbacks, scenario helpers, evaluation tools
│   ├── utils_soil/           # SoilGrids extraction and soil-file generation utilities
│   ├── configs/              # Crop, soil, site, weather, scenario, price, and PCSE config assets
│   ├── train_allocator.py    # Bandit allocator training loops
│   ├── run_scenarios.py      # Regional and farm-level simulation/evaluation runner
│   ├── fit_lp.py             # LP and SPSA baseline precomputation
│   └── fit_nue_response.py   # NUE/N-surplus response fitting
├── scripts/                  # Command-line entry points for reproduction workflows
├── tests/                    # Unit and environment smoke tests
├── notebooks/                # Result exploration and plotting notebooks
└── pyproject.toml            # Package metadata and uv dependency specification
```

## Installation

This project was developed with Python 3.11.9 and is configured for Python `>=3.11,<3.13`.

The `pyproject.toml` expects a compatible editable PCSE checkout at `../pcse`:

```text
parent-directory/
├── pcse/
└── CropGym-BanditResourceAllocation/
```

Install with `uv`:

```bash
cd CropGym-BanditResourceAllocation
uv python install 3.11
uv sync
```

If you want to use Comet logging, create a local `comet/api` file containing your API key. Without that file, the code falls back to local logging.

## Quick Checks

Run the focused unit tests:

```bash
uv run pytest tests
```

Check that the package imports and registers the predefined single-field environments:

```bash
uv run python -c "import cropgym; print(cropgym._CONFIG_PATH)"
```

## Reproducing The Experiments

The core reproduction workflow is:

1. Prepare or verify scenario and soil configuration files.
2. Run crop simulations for regional farms and years.
3. Fit response models and LP/SPSA allocation baselines.
4. Train and evaluate bandit resource allocators.
5. Inspect results with the notebook in `notebooks/`.

### Scenario And Soil Assets

Scenario YAML files are stored under `cropgym/configs/scenarios/`. Soil assets are stored under `cropgym/configs/soil/`, and the repository already includes the scenario and SoilGrids files used by the current experiments.

For new coordinate-based scenarios, generate one SoilGrids soil file with:

```bash
uv run python scripts/generate_soilgrids_soil_file.py --longitude 6.656 --latitude 52.966
```

The helper below can generate missing soil files from scenario YAMLs when those YAMLs are organized as `<region>/<year>/farmer_*.yaml`:

```bash
uv run python scripts/pull_scenarios_soils.py --year 2020
```

### Run Regional Simulations

Run the rule-of-thumb baseline across the default test years, with optional aggregation:

```bash
uv run python -m cropgym.run_scenarios \
  --regions all \
  --years 0 \
  --baseline ROT \
  --scenario full_budget \
  --num_workers 1 \
  --aggregate
```

Run only one global farm:

```bash
uv run python -m cropgym.run_scenarios \
  --farm 0 \
  --baseline ROT \
  --scenario full_budget \
  --aggregate
```

### Fit LP Baselines


Fit offline LP responses from an evaluation pickle:

```bash
uv run python -m cropgym.fit_lp \
  --mode offline_fit \
  --file_path results/ROT/<evaluation-file>.pkl
```

### Fit NUE And N-Surplus Responses (Demeter)

```bash
uv run python -m cropgym.fit_nue_response \
  --file_path results/ROT/<evaluation-file>.pkl \
  --train_years 2015,2016,2017,2018,2019 \
  --tag ROT
```

### Train Bandit Allocators

Train the default resource allocation bandit:

```bash
uv run python scripts/training_resource_allocation.py \
  --baseline ROT \
  --rounds 300 \
  --years 2 \
  --eval_steps 10 \
  --method ucb \
  --bandit_action_mode factored \
  --no_comet
```

Train for a single farm:

```bash
uv run python scripts/training_resource_allocation.py \
  --farm 0 \
  --baseline ROT \
  --rounds 300 \
  --method ucb \
  --no_comet
```

## Outputs

Generated artifacts are written to local experiment folders:

- `models/`: trained allocator checkpoints
- `results/`: scenario outputs, LP outputs, fitted responses, and evaluation pickles
- `logs/`: local training logs
- `plots/`: generated diagnostic figures
- `resume/`: resumed training state

These folders are intentionally not required for importing `cropgym`; they are produced by reproduction runs.
