"""Baselines for the day-ahead task (inputs end at D 11:00, target = D+1 00:00-23:00).

 - naive_lastday : same hour of the last fully observed day (D-1), i.e. target - 48 h
 - naive_dayD    : price only, same hour of day D (target - 24 h; the day-D auction prices are known at 12:00)
 - naive_lastweek: same hour and weekday one week before the target day (target - 168 h)
 - smard_fc_solar: the official SMARD day-ahead solar forecast for the target day (benchmark, solar only)

They give a point forecast only; to score them like the neural nets, the 10%/90% band is the baseline's own
residual quantile **per target hour of day** (the targets are day-aligned), estimated on the training days.
Run:  python src/baselines.py
"""
import json

import numpy as np
import pandas as pd

import windows as W
from evaluate import metrics, table, to_units

QS = (0.1, 0.9)
TC = [W.BASE.index(t) for t in W.TARGETS]        # price, solar are the first input columns
# row of X (rows are D-6 12:00 ... D 11:00) holding target hour h of day D+1 minus `lag` hours
LAST_X = W.LOOKBACK - 1                           # row of D 11:00
OFFSET = W.GAP + 1                                # D+1 00:00 is OFFSET hours after D 11:00


def naive(X: np.ndarray, kind: str) -> np.ndarray:
    """X: (N, 168, C) scaled windows. Returns the (N, 24, 2) scaled point forecast."""
    lag = {"lastday": 48, "lastweek": 168}[kind]
    rows = [LAST_X + OFFSET + h - lag for h in range(24)]
    assert min(rows) >= 0 and max(rows) <= LAST_X
    return X[:, rows, :][:, :, TC]


def with_band(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.stack([point + lo, point, point + hi], axis=-1)  # (N, 24, 2, 3)


def band(resid: np.ndarray):
    """resid: (N, 24, K) training residuals of day-aligned examples -> per target hour 10/90 % quantiles."""
    return np.quantile(resid, QS, axis=0)


def run() -> dict:
    df = W.load_dataset()
    data, scaler = W.prepare(df, W.BASE, W.FUTURE)
    cols = W.BASE + [c for c in W.FUTURE if c not in W.BASE]
    day = data["train"].start.hour == 0        # one example per training day, targets start at 00:00
    out = {}
    for kind in ("lastday", "lastweek"):
        resid = data["train"].Y[day] - naive(data["train"].X[day], kind)
        lo, hi = band(resid)
        for split in ("val", "test"):
            w = data[split]
            out.setdefault(f"naive_{kind}", {})[split] = metrics(
                to_units(w.Y, scaler, cols), to_units(with_band(naive(w.X, kind), lo, hi), scaler, cols))
    # price only: same hour of day D (target - 24 h). The day-D auction prices are published before 12:00 of D,
    # so this is a legitimate, stronger baseline than D-1. (Solar of D 12-23 is not yet known, so no solar version.)
    pr = df["price"]
    d_pt = lambda w: np.stack([pr.loc[t - pd.Timedelta(hours=24):t - pd.Timedelta(hours=1)].to_numpy() for t in w.start])
    jp = cols.index("price")
    tr_pt = d_pt(data["train"])
    resid = (to_units(data["train"].Y, scaler, cols)[..., 0] - tr_pt)[day]
    lo, hi = band(resid[..., None])
    for split in ("val", "test"):
        w = data[split]
        y = to_units(w.Y, scaler, cols)
        pt = d_pt(w)
        q = np.stack([pt + lo[:, 0], pt, pt + hi[:, 0]], axis=-1)                    # (N, 24, 3)
        qq = np.repeat(y[..., None], 3, axis=-1)                                      # solar = truth, not scored
        qq[:, :, 0, :] = q
        out.setdefault("naive_dayD", {})[split] = {"price": metrics(y, qq)["price"]}
    # official SMARD day-ahead solar forecast for the target day (scaled with fc_solar's own scaler in F)
    k, jf, js = W.FUTURE.index("fc_solar"), cols.index("fc_solar"), cols.index("solar")

    def fc_point(w):  # (N, 24, 2) scaled like Y; the price column is a dummy copy of the truth (not scored)
        p = w.Y.copy()
        units = w.F[..., k] * scaler.scale_[jf] + scaler.mean_[jf]
        p[..., 1] = (units - scaler.mean_[js]) / scaler.scale_[js]
        return p

    resid = (data["train"].Y - fc_point(data["train"]))[day][..., 1:]
    lo, hi = band(resid)
    for split in ("val", "test"):
        w = data[split]
        pt = fc_point(w)
        q = with_band(pt, np.concatenate([np.zeros_like(lo), lo], -1), np.concatenate([np.zeros_like(hi), hi], -1))
        res = metrics(to_units(w.Y, scaler, cols), to_units(q, scaler, cols))
        out.setdefault("smard_fc_solar", {})[split] = {"solar": res["solar"]}
    return out


if __name__ == "__main__":
    res = run()
    print("VALIDATION\n" + table({k: v["val"] for k, v in res.items()}))
    with open(W.ROOT / "results" / "baselines.json", "w") as f:
        json.dump(res, f, indent=1)
