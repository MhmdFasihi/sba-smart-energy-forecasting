# Smart Energy Forecasting with an AI planning assistant

Day-ahead forecast of the German electricity price (ct/kWh) and solar generation (MWh/h) with a low/middle/high band,
three model families (LSTM, 1D-CNN, PatchTST) against naive baselines and the official SMARD solar forecast, a planning
assistant and a Streamlit dashboard.

**Protocol (v3):** at 12:00 of day D (auction cut-off) forecast day D+1 from the last 168 h plus known-future inputs of
the target day. Two input families:
- **noon-safe** (primary result): calendar, official day-ahead load forecast, and the day-D auction prices (all public at 12:00);
- **all fc** (upper bound): additionally the official wind/solar forecasts, which SMARD only requires "by 18:00 of the
  previous day", so their version at noon is unknown.

Train 2019-2024, validation Jan-Aug 2025 (all choices), calibration Sep-Dec 2025 (band recalibration only), test 2026.
The 2026 test year was also scored by earlier versions; see the version history in `results/REPORT.md`.

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
python src/run_experiments.py ablate   # weather on/off per family, 3 models x 2 seeds (validation only)
python src/run_experiments.py tune     # lr x dropout grid per family and model, 2 seeds each
python src/run_experiments.py final    # tuned setting, 5 seeds; writes the selection rule to results/experiment_plan.json
python src/final_eval.py               # scores the test set (refuses to run twice without --force)
python src/calibrate.py                # per-hour split-conformal band widening, fitted on Sep-Dec 2025 only
python src/final_eval.py --force       # regenerates the report with the recalibration table (same numbers)
python src/check_report.py             # checks the report AND recomputes the test metrics from the raw files
```
`--jobs N` runs N trainings in parallel. Training uses CUDA, Apple MPS or CPU automatically. `colab_training.ipynb`
runs the whole chain on a Google Colab GPU (not tested on Colab by the author).

## App
```
streamlit run app.py
```
Demo mode compares forecast and reality for any 2025/2026 day (recalibrated bands by default, raw bands with a toggle).
"Real tomorrow" needs newer SMARD downloads in `data/` (rebuilt on the fly). LLM explanation: put `ANTHROPIC_API_KEY`
(or `NVIDIA_API_KEY`) in `.streamlit/secrets.toml` (git-ignored); without a key a text summary is shown.

## Deploy on Streamlit Community Cloud
Push the repo to GitHub (the models, results and dataset needed by the app are included), then on share.streamlit.io choose
the repo, branch `main` and `app.py`. Add the optional LLM key under *Settings > Secrets*. Not deployed by the author yet.

## Tests
`python -m pytest -q tests`

## Deviations from the project guide
PatchTST is a native PyTorch implementation (neuralforecast is univariate); plots use Altair instead of plotly; the
forecast is made at 12:00 of day D for D+1 with known-future inputs; validation is Jan-Aug 2025 because Sep-Dec 2025 is
reserved for band recalibration. All deviations and limitations: `results/REPORT.md`.
