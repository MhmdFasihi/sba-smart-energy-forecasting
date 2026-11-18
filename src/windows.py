"""Step 3: time split, scaling, sliding windows for a real DAY-AHEAD forecast.

Protocol (guide section 9, auction timing): at 12:00 of day D the forecast for day D+1 must be ready.
  input  = the last 168 h before 12:00 of day D          (rows [j-12-168, j-12))
  target = D+1 00:00 ... 23:00, i.e. 12-36 h ahead         (rows [j, j+24))   j = index of D+1 00:00
Known-future inputs for the target day (available at 12:00 of D): calendar features of the target hours
and, optionally, the official SMARD day-ahead forecasts (solar, wind onshore, load).
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LOOKBACK, HORIZON = 168, 24
GAP = 12            # hours between the end of the input (D 12:00) and the start of the target (D+1 00:00)
TARGETS = ["price", "solar"]
CALENDAR = ["is_weekend", "is_holiday", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
BASE = ["price", "solar", "wind_onshore", "wind_offshore", "load", "residual_load"] + CALENDAR  # guide's 12
WEATHER = ["radiation", "sunshine", "cloud_cover", "temp"]
INTRAHOUR = ["price_range_q"]                         # max-min of the four 15-min prices in the hour (ablation)
DAYAHEAD_FC = ["fc_solar", "fc_wind_onshore", "fc_load"]   # official SMARD forecasts for the target day
FUTURE = CALENDAR + DAYAHEAD_FC                       # known-future inputs of the target hours
SPLITS = {"train": ("2019-01-01", "2024-12-31"), "val": ("2025-01-01", "2025-12-31"), "test": ("2026-01-01", None)}


def load_dataset(path=None) -> pd.DataFrame:
    return pd.read_csv(path or ROOT / "data" / "dataset_hourly.csv", index_col=0, parse_dates=True)


def split_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = SPLITS[name]
    return df.loc[a:b]


@dataclass
class Windows:
    X: np.ndarray            # (N, 168, C) scaled inputs, ending at D 11:00
    Y: np.ndarray            # (N, 24, 2) scaled targets (D+1 00:00..23:00)
    F: np.ndarray            # (N, 24, K) scaled known-future inputs of the target hours (K may be 0)
    start: pd.DatetimeIndex  # first target hour (D+1 00:00) of each example


def make_windows(values: np.ndarray, index: pd.DatetimeIndex, input_cols: list[int], target_cols: list[int],
                 future_cols: list[int], only_midnight: bool) -> Windows:
    """values: (T, C) scaled. Example j (target = rows [j, j+24)) uses the input rows [j-GAP-168, j-GAP).
    Needs a contiguous hourly index; examples that cross a gap are skipped. only_midnight=True keeps one
    example per forecast day (target starts at 00:00); False slides over every hour (training augmentation)."""
    T = len(values)
    starts, X, Y, F = [], [], [], []
    ok_step = np.diff(index.values).astype("timedelta64[h]").astype(int) == 1
    seg = np.concatenate([[0], np.cumsum(~ok_step)])  # segment id per row
    for j in range(LOOKBACK + GAP, T - HORIZON + 1):
        if only_midnight and index[j].hour != 0:
            continue
        if seg[j - GAP - LOOKBACK] != seg[j + HORIZON - 1]:
            continue
        X.append(values[j - GAP - LOOKBACK:j - GAP][:, input_cols])
        Y.append(values[j:j + HORIZON][:, target_cols])
        F.append(values[j:j + HORIZON][:, future_cols])
        starts.append(index[j])
    return Windows(np.asarray(X, dtype=np.float32).reshape(len(starts), LOOKBACK, len(input_cols)),
                   np.asarray(Y, dtype=np.float32).reshape(len(starts), HORIZON, len(target_cols)),
                   np.asarray(F, dtype=np.float32).reshape(len(starts), HORIZON, len(future_cols)),
                   pd.DatetimeIndex(starts))


def prepare(df: pd.DataFrame, features: list[str] | None = None, future: list[str] | None = None,
            save_scaler=None):
    """Return dict split -> Windows plus the fitted scaler (fitted on training rows only).
    `features`: input columns (first columns must keep price, solar as in BASE). `future`: known-future
    columns of the target day ([] = none). Validation/test inputs may reach back into the previous split
    (real past data, no leakage); their targets lie fully inside the split."""
    features = list(features or BASE)
    future = list(future or [])
    cols = features + [c for c in future if c not in features]
    df = df.copy()
    fc = [c for c in DAYAHEAD_FC if c in df]
    df[fc] = df[fc].interpolate(limit=48, limit_direction="both")  # a few multi-day holes in 2022-23 (train only)
    scaler = StandardScaler().fit(split_frame(df, "train")[cols].to_numpy())
    scaled = pd.DataFrame(scaler.transform(df[cols].to_numpy()), index=df.index, columns=cols)
    tcols = [cols.index(t) for t in TARGETS]
    fcols = [cols.index(c) for c in future]
    xcols = [cols.index(c) for c in features]
    out = {}
    for name in SPLITS:
        a, b = SPLITS[name]
        hist_start = pd.Timestamp(a) - pd.Timedelta(hours=LOOKBACK + GAP)
        part = scaled.loc[hist_start:b]
        w = make_windows(part.to_numpy(), part.index, xcols, tcols, fcols, only_midnight=(name != "train"))
        keep = w.start >= pd.Timestamp(a)  # targets start inside the split
        out[name] = Windows(w.X[keep], w.Y[keep], w.F[keep], w.start[keep])
    if save_scaler:
        joblib.dump({"scaler": scaler, "features": features, "future": future, "cols": cols}, save_scaler)
    return out, scaler


def inverse_targets(Y: np.ndarray, scaler: StandardScaler, features: list[str]) -> np.ndarray:
    """Scaled (..., 2) target values (or quantiles) back to ct/kWh and MWh."""
    out = np.empty_like(Y)
    for k, t in enumerate(TARGETS):
        j = features.index(t)
        out[..., k] = Y[..., k] * scaler.scale_[j] + scaler.mean_[j]
    return out
