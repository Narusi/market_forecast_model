# market-forecast

[![CI](https://github.com/your-org/market_forecast_model/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/market_forecast_model/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-%3E%3D80%25-brightgreen)](https://github.com/your-org/market_forecast_model/actions/workflows/ci.yml)

`market-forecast` is a Python package for 1 to 12 week market movement forecasting with reusable time-series core components.

## Features in this MVP
- Single and multi-asset tabular ingestion (`pandas`)
- Feature generation for trend, seasonality, volatility, and technical indicators
- Model wrappers for ARIMA, EGARCH (via `arch`), and SVR
- Ensemble combiner with performance-weighted averaging
- Signal conversion to buy/hold/sell with tiered technical + ARIMA confirmation
- Walk-forward backtesting engine
- Regime-aware indicator scoring and signal selection
- Plotly dashboard builders for single and multi-asset outputs
- Colab-ready notebook in `notebooks/market_forecast_colab_demo.ipynb`

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[market,dashboard,dev]
```


## Install from GitHub

Install directly from a GitHub repository (replace with your repo URL):

```bash
pip install "git+https://github.com/your-org/market_forecast_model.git"
```

Install a specific branch:

```bash
pip install "git+https://github.com/your-org/market_forecast_model.git@main"
```

Install a specific tag or commit:

```bash
pip install "git+https://github.com/your-org/market_forecast_model.git@v0.1.0"
# or
pip install "git+https://github.com/your-org/market_forecast_model.git@<commit_sha>"
```

If your repository is private, authenticate with a personal access token or SSH URL and ensure your environment has access rights.

## Quick start

```python
from market_forecast.pipelines.forecast_pipeline import ForecastPipeline
from market_forecast.config.schemas import ForecastConfig

config = ForecastConfig()
pipeline = ForecastPipeline(config=config)
pipeline.fit(prices_df)
pred = pipeline.predict(horizons=[1, 2, 4, 8, 12])
signals = pipeline.generate_signals(pred)
state = pipeline.summarize_current_state()
```

## CLI

```bash
mforecast --help
```

## Colab
Open and run `notebooks/market_forecast_colab_demo.ipynb` in Google Colab.


## CI

GitHub Actions runs the following checks on Python 3.11:
- `ruff check .`
- `black --check .`
- `mypy src`
- `pytest` with coverage enforcement (`--cov-fail-under=80`)

A notebook smoke test job is also available as an optional `workflow_dispatch` input (`run_notebook_smoke`).

## Local pre-commit hooks

Install and enable local hooks:

```bash
pip install -e .[dev]
pre-commit install
```

Run all hooks locally:

```bash
pre-commit run --all-files
```

The configured hooks run `ruff`, `black`, `mypy`, and `pytest` via `.pre-commit-config.yaml`.
