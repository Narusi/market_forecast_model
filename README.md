# market-forecast

`market-forecast` is a Python package for 1 to 12 week market movement forecasting with reusable time-series core components.

## Features in this MVP
Default input mode is **single asset / single market index**. Multi-asset analysis is optional.

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

Install from pip-style requirements file:

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e .[market,dashboard,dev]
```


## Install from Git clone

Clone the repository and install from local source:

```bash
git clone https://github.com/market-forecast/market_forecast_model.git
cd market_forecast_model
pip install -e .
```

Install pinned runtime dependencies only:

```bash
pip install -r requirements.txt
```

Install runtime + development tools:

```bash
pip install -r requirements.txt -r requirements/dev.txt
```

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
