"""Fill gaps at the end of data/weather_hourly.csv with Open-Meteo values.

DWD publishes hourly solar radiation with a lag (to 2026-10-31); Open-Meteo (free, no key,
CC BY 4.0) covers the last weeks. Only NaN cells are filled, DWD values are never overwritten.
Open-Meteo "preceding hour" values (shortwave_radiation, sunshine_duration) are labelled at the END of
the hour -> shifted back 1 h so the label is the interval start like SMARD/DWD. temperature_2m and
cloud_cover are instantaneous values at the label time and are NOT shifted.

Run:  python src/fetch_weather_openmeteo.py
"""
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
POINTS = [(53.63, 9.99), (52.38, 13.06), (52.29, 10.45), (51.46, 7.26), (50.79, 12.87),
          (49.77, 9.96), (48.83, 9.20), (48.40, 11.69), (49.21, 7.11), (49.50, 11.05)]
VARS = "shortwave_radiation,sunshine_duration,cloud_cover,temperature_2m"


def fetch_recent(past_days: int = 60) -> pd.DataFrame:
    frames = []
    for lat, lon in POINTS:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
            "latitude": lat, "longitude": lon, "hourly": VARS, "past_days": past_days,
            "forecast_days": 1, "timezone": "Europe/Berlin"})
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h.index = pd.to_datetime(h.pop("time"))
        for c in ("shortwave_radiation", "sunshine_duration"):  # preceding-hour values: label end -> start
            h[c] = h[c].shift(-1)
        frames.append(h)
    m = sum(frames) / len(frames)
    return pd.DataFrame({
        "radiation": m["shortwave_radiation"],
        "sunshine": m["sunshine_duration"] / 60.0,   # s per hour -> min per hour
        "cloud_cover": m["cloud_cover"] / 100 * 8,   # % -> eighths
        "temp": m["temperature_2m"],
    })


def fill_gaps(w: pd.DataFrame) -> pd.DataFrame:
    om = fetch_recent()
    idx = pd.date_range(w.index.min(), max(w.index.max(), om.dropna(how="all").index.max()), freq="h", name=w.index.name)
    om = om.reindex(idx)  # also extends the table past the last DWD hour
    filled = w.reindex(idx)
    for c in filled.columns:
        n = filled[c].isna().sum()
        filled[c] = filled[c].fillna(om[c])
        print(f"{c}: filled {n - filled[c].isna().sum()} of {n} NaN")
    return filled


if __name__ == "__main__":
    w = pd.read_csv(DATA / "weather_hourly.csv", index_col=0, parse_dates=True)
    fill_gaps(w).to_csv(DATA / "weather_hourly_filled.csv")
