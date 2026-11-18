# Smart Energy Forecasting with an AI planning assistant

Day-ahead forecast of the German electricity price (ct/kWh) and solar generation (MWh/h) with a low/middle/high band,
three model families (LSTM, 1D-CNN, PatchTST) against naive baselines and the official SMARD solar forecast, a planning
assistant and a Streamlit dashboard.

**Protocol:** at 12:00 of day D (auction cut-off) forecast day D+1 from the last 168 h; known-future inputs of the
target day (calendar, official SMARD day-ahead forecasts) are used. Train 2019-2024, validation 2025, test 2026 (scored once).

## Setup (Python 3.10-3.12)
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data (SMARD, Germany/Luxembourg, region DE-LU, CC BY 4.0, "Bundesnetzagentur | SMARD.de")
Put the CSV downloads in `data/`: day-ahead prices (**quarter-hour** file; averaged to hours by the code), actual
generation, actual consumption, forecasted generation day-ahead, forecasted consumption (hour resolution).

```
python src/fetch_weather_dwd.py        # DWD observations (cached in data/weather_raw)
python src/fetch_weather_openmeteo.py  # fills the latest weeks from Open-Meteo
python src/data_build.py               # -> data/dataset_hourly.csv
```

## Train and evaluate
```
python src/run_experiments.py ablate   # validation-only input ablation (seed 0)
python src/run_experiments.py tune     # lr x dropout grid per model
python src/run_experiments.py final    # 3 seeds x 3 models x with/without weather
python src/final_eval.py               # scores the test set ONCE (needs --force to repeat), writes results/REPORT.md
python src/check_report.py             # verifies the report against results/test_results.json
```

## App
```
streamlit run app.py
```
Demo mode compares forecast and reality for any 2025/2026 day. "Real tomorrow" needs newer SMARD downloads in `data/`
(rebuilt on the fly). LLM explanation: put `ANTHROPIC_API_KEY` (or `NVIDIA_API_KEY`) in `.streamlit/secrets.toml`; without a
key a text summary is shown. Never commit the secrets file.

## Tests
`python -m pytest -q tests`

## Deviations from the project guide
PatchTST is a native PyTorch implementation (neuralforecast is univariate); plots use Altair; see `results/REPORT.md`.
