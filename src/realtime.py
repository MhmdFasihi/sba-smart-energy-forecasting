"""Real-tomorrow mode: forecast the next day from the newest SMARD files in data/.

At 12:00 of day D the input is the last 168 h before D 12:00 and the known-future inputs of D+1 (calendar and the
official SMARD day-ahead forecasts, which must be in the downloaded files). The same pipeline as training is used
(windows.py scaling, model checkpoints from models/).
"""
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch

import data_build
import windows as W
from evaluate import to_units
from models import build


class NotEnoughData(RuntimeError):
    pass


@dataclass
class Forecast:
    day: pd.Timestamp          # forecast day D+1 (00:00)
    origin: pd.Timestamp       # D 12:00, the moment the forecast is made
    q: np.ndarray              # (24, 2, 3) price/solar x low/mid/high, ct/kWh and MWh
    known_price_today: pd.Series   # auction prices of D 12:00-23:00 (already known at the origin)


def load_checkpoint(run: str):
    ck = torch.load(W.ROOT / "models" / f"{run}.pt", map_location="cpu", weights_only=False)
    model = build(ck["model"], len(ck["features"]), len(ck["future"]), **ck["hp"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck


def forecast_next_day(run: str, df: pd.DataFrame | None = None, origin_day: pd.Timestamp | None = None) -> Forecast:
    """df: output of data_build.build(trim=False). origin_day = D (midnight); default = the newest usable day."""
    df = data_build.build(intrahour=True, trim=False) if df is None else df
    model, ck = load_checkpoint(run)
    sc = joblib.load(W.ROOT / "models" / ck["scaler"])
    cols = sc["cols"]
    actual = ["price", "solar", "wind_onshore", "wind_offshore", "load", "residual_load"]
    complete = df.loc[:, actual].dropna().index
    if origin_day is None:
        t = complete.max()
        origin_day = (t if t.hour >= 11 else t - pd.Timedelta(days=1)).normalize()
    origin = origin_day + pd.Timedelta(hours=12)
    t0 = origin + pd.Timedelta(hours=W.GAP)                       # D+1 00:00
    hist = df.loc[origin - pd.Timedelta(hours=W.LOOKBACK):origin - pd.Timedelta(hours=1)]
    tgt = df.loc[t0:t0 + pd.Timedelta(hours=W.HORIZON - 1)]
    if len(hist) < W.LOOKBACK or hist[actual].isna().any().any():
        raise NotEnoughData(f"actual data up to {origin - pd.Timedelta(hours=1)} is incomplete")
    if len(tgt) < W.HORIZON:
        raise NotEnoughData(f"no data rows for {t0.date()}: download newer SMARD files (day-ahead forecasts) into data/")
    fut = ck["future"]
    if fut and tgt[fut].isna().any().any():
        raise NotEnoughData(f"missing {', '.join(c for c in fut if tgt[c].isna().any())} for {t0.date()}")
    feats = ck["features"]
    if hist[feats].isna().any().any():
        raise NotEnoughData("weather inputs are missing for the last 168 h (refresh the weather files)")
    scaled = lambda frame, c: sc["scaler"].transform(frame.reindex(columns=cols).to_numpy())[:, [cols.index(x) for x in c]]
    X = scaled(hist, feats).astype(np.float32)[None]
    F = scaled(tgt, fut).astype(np.float32)[None] if fut else np.zeros((1, W.HORIZON, 0), np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(X), torch.from_numpy(F)).numpy()
    q = np.sort(to_units(pred, sc["scaler"], cols), axis=-1)[0]
    known = df.loc[origin:origin + pd.Timedelta(hours=11), "price"]
    return Forecast(t0, origin, q, known)


def default_run() -> str:
    return json.load(open(W.ROOT / "results" / "test_results.json"))["winner_run"]
