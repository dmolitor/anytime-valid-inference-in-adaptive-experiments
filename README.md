# Anytime-Valid Inference in Adaptive Experiments: Covariate Adjustment and Balanced Power

This repository replicates figures/tables for [Anytime-Valid Inference in Adaptive Experiments: Covariate Adjustment and Balanced Power](https://arxiv.org/abs/2506.20523) (Molitor and Gold, 2025).

## Install dependencies

First, ensure [uv is installed](https://docs.astral.sh/uv/getting-started/installation/).
Then, install required dependencies:
```
uv sync
```

## Replicate figures/tables

All figures and tables (there's only one table) will be output in the `./figures` directory. To replicate:
```
uv run simulation_results.py
uv run rct_results.py
```

This will likely take a while.