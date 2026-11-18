"""Step 2: build data/dataset_hourly.csv from the SMARD CSV files.

Run:  python src/data_build.py
Data: SMARD (Bundesnetzagentur | SMARD.de), CC BY 4.0.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

# file prefix -> {original column prefix: short name}
SOURCES = {
    "Actual_generation": {
        "Photovoltaics [": "solar",
        "Wind onshore [": "wind_onshore",
        "Wind offshore [": "wind_offshore",
    },
    "Actual_consumption": {"grid load [": "load", "Residual load [": "residual_load"},
    # day-ahead forecasts (extra input / benchmark), kept with prefix fc_
    "Forecasted_generation_Day-Ahead": {"Photovoltaics [": "fc_solar", "Wind onshore [": "fc_wind_onshore"},
    "Forecasted_consumption": {"grid load [": "fc_load"},
}


def read_smard(path: Path, rename: dict) -> pd.DataFrame:
    """SMARD export: ';' separated, '1,234.5' numbers, '-' = missing, local wall-clock time."""
    df = pd.read_csv(path, sep=";", thousands=",", na_values="-", encoding="utf-8-sig")
    ts = pd.to_datetime(df["Start date"], format="%b %d, %Y %I:%M %p")
    out = {}
    for prefix, short in rename.items():
        col = next(c for c in df.columns if c.startswith(prefix))
        out[short] = pd.to_numeric(df[col], errors="coerce").to_numpy()
    out = pd.DataFrame(out, index=pd.DatetimeIndex(ts, name="timestamp"))
    # October clock change: the repeated hour appears twice -> keep the first
    out = out[~out.index.duplicated(keep="first")]
    return out


def read_price_quarterhour(path: Path) -> pd.DataFrame:
    """Day-ahead price in its native 15-min resolution (since Oct 2025 the market clears quarter-hourly)
    -> hourly mean (guide step 3) plus the within-hour min/max of the four quarters."""
    df = pd.read_csv(path, sep=";", thousands=",", na_values="-", encoding="utf-8-sig")
    ts = pd.to_datetime(df["Start date"], format="%b %d, %Y %I:%M %p")
    col = next(c for c in df.columns if c.startswith("Germany/Luxembourg ["))
    q = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=pd.DatetimeIndex(ts, name="timestamp"))
    q = q[~q.index.duplicated(keep="first")]          # October clock change: keep the first repeated hour
    g = q.groupby(q.index.floor("h"))
    full = g.count() == 4                              # only hours with all four quarters are valid
    out = pd.DataFrame({"price": g.mean().where(full), "price_qmin": g.min().where(full),
                        "price_qmax": g.max().where(full)})
    out.index.name = "timestamp"
    return out


def build(intrahour: bool = False, trim: bool = True) -> pd.DataFrame:
    """intrahour=True also keeps the within-hour range of the 15-min prices (price_range_q).
    trim=False (real-time mode) keeps the trailing rows that have day-ahead forecasts / known prices but no
    actuals yet, so the target day can be forecast."""
    frames = [read_price_quarterhour(next(DATA.glob("Day-ahead_prices_*_Quarterhour.csv")))]
    for prefix, rename in SOURCES.items():
        path = next(DATA.glob(f"{prefix}_*.csv"))
        frames.append(read_smard(path, rename))
    df = pd.concat(frames, axis=1)
    wpath = DATA / "weather_hourly_filled.csv"
    if wpath.exists():  # DWD observations (+ Open-Meteo for the latest weeks)
        weather = pd.read_csv(wpath, index_col=0, parse_dates=True)
        df = df.join(weather)

    # full hourly grid -> March clock change leaves one missing hour, make it explicit
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", name="timestamp"))

    for c in ("price", "price_qmin", "price_qmax"):
        df[c] = df[c] / 10.0  # EUR/MWh -> ct/kWh
    df["price_range_q"] = df["price_qmax"] - df["price_qmin"]
    df = df.drop(columns=["price_qmin", "price_qmax"])
    if not intrahour:
        df = df.drop(columns=["price_range_q"])

    # drop the trailing rows without actual values (future hours)
    actual = ["price", "solar", "wind_onshore", "wind_offshore", "load", "residual_load"]
    last = df[actual].dropna(how="all").index.max()
    last_complete = df.loc[:last, actual].dropna().index.max()
    if trim:
        df = df.loc[:last_complete]

    # fill small gaps (<= 3 h), drop rows that are still incomplete (day-ahead forecasts may stay NaN)
    df = df.interpolate(limit=3, limit_area="inside")
    if trim:
        df = df.dropna(subset=actual)

    df = add_calendar_features(df)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    import holidays  # lazy: only needed here

    idx = df.index
    de_holidays = holidays.country_holidays("DE", years=sorted(set(idx.year)))
    df["is_weekend"] = (idx.weekday >= 5).astype(int)
    df["is_holiday"] = pd.Series(idx.date, index=idx).map(lambda d: d in de_holidays).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    doy = idx.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


if __name__ == "__main__":
    df = build(intrahour=True)
    df.to_csv(DATA / "dataset_hourly.csv")
    print(df.shape, df.index.min(), "->", df.index.max())
    print(df.isna().sum()[lambda s: s > 0])
    print(df[["price", "solar"]].describe().round(2))
