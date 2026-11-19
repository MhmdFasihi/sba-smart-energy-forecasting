"""Baselines for the day-ahead task (inputs known at 12:00 of day D, target = D+1 00:00-23:00).

 - naive_lastday : same hour of the last fully observed day (D-1), i.e. target - 48 h       (guide: "yesterday")
 - naive_lastweek: same hour and weekday one week before the target day (target - 168 h)    (guide: "last week")
 - naive_dayD    : price only, same hour of day D (target - 24 h; day D's auction prices are public at 12:00 of D)
 - smard_fc_solar: the official SMARD day-ahead solar forecast for the target day (benchmark, solar only)

They give a point forecast only; to score them like the neural nets, the 10%/90% band is the baseline's own
residual quantile per target hour of day, estimated on the training days.
Run:  python src/baselines.py
"""
import json
import warnings

import numpy as np
import pandas as pd

import windows as W
from evaluate import metrics, table

QS = (0.1, 0.9)
SPLITS = ("val", "cal", "test")
TARGETS_OF = {"naive_lastday": ["price", "solar"], "naive_lastweek": ["price", "solar"], "naive_dayD": ["price"],
              "smard_fc_solar": ["solar"]}


def _lagged(df: pd.DataFrame, starts: pd.DatetimeIndex, col: str, lag: int) -> np.ndarray:
    """(N, 24) values of `col` at target hour - lag."""
    s = df[col]
    return np.stack([s.loc[t - pd.Timedelta(hours=lag):t - pd.Timedelta(hours=lag - 23)].to_numpy() for t in starts])


def _truth(df, starts) -> np.ndarray:
    return np.stack([df.loc[t:t + pd.Timedelta(hours=23), W.TARGETS].to_numpy() for t in starts])


def point(df: pd.DataFrame, starts: pd.DatetimeIndex, name: str) -> np.ndarray:
    """(N, 24, 2) point forecast in units; targets a baseline does not cover are NaN."""
    p = np.full((len(starts), 24, 2), np.nan)
    if name in ("naive_lastday", "naive_lastweek"):
        lag = 48 if name == "naive_lastday" else 168
        for k, t in enumerate(W.TARGETS):
            p[:, :, k] = _lagged(df, starts, t, lag)
    elif name == "naive_dayD":
        p[:, :, 0] = _lagged(df, starts, "price", 24)
    elif name == "smard_fc_solar":
        p[:, :, 1] = _lagged(df, starts, "fc_solar", 0)
    return p


def predictions(df: pd.DataFrame | None = None) -> dict:
    """name -> split -> (y (N,24,2), q (N,24,2,3)) in units. Band = training residual 10/90 % per target hour."""
    df = W.load_dataset() if df is None else df
    data, _ = W.prepare(df, W.BASE, W.FUTURE)
    starts = {s: data[s].start for s in ("train",) + SPLITS}
    tr_days = starts["train"][starts["train"].hour == 0]          # one training example per day
    y_tr = _truth(df, tr_days)
    out = {}
    for name in TARGETS_OF:
        pt_tr = point(df, tr_days, name)
        with warnings.catch_warnings():                          # targets a baseline does not cover are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanquantile(y_tr - pt_tr, QS, axis=0)    # (24, 2) each
        out[name] = {}
        for s in SPLITS:
            pt = point(df, starts[s], name)
            out[name][s] = (_truth(df, starts[s]), np.stack([pt + lo, pt, pt + hi], axis=-1))
    return out


def run(df: pd.DataFrame | None = None) -> dict:
    """name -> split -> metrics of the targets the baseline covers (incl. daylight-only solar)."""
    df = W.load_dataset() if df is None else df
    preds = predictions(df)
    out = {}
    for name, by_split in preds.items():
        out[name] = {}
        for s, (y, q) in by_split.items():
            res = metrics(np.nan_to_num(y), np.nan_to_num(q), W.daylight_mask(df, _starts(df, s)))
            keep = TARGETS_OF[name] + (["solar_day"] if "solar" in TARGETS_OF[name] else [])
            out[name][s] = {t: res[t] for t in keep}
    return out


_START_CACHE: dict = {}


def _starts(df, split):
    if split not in _START_CACHE:
        _START_CACHE.update({s: w.start for s, w in W.prepare(df, W.BASE, W.FUTURE)[0].items()})
    return _START_CACHE[split]


if __name__ == "__main__":
    res = run()
    print("VALIDATION\n" + table({k: v["val"] for k, v in res.items()}))
    with open(W.ROOT / "results" / "baselines.json", "w") as f:
        json.dump(res, f, indent=1)
