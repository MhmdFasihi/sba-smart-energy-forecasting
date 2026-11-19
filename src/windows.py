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
# price_range_q (max-min of the four 15-min prices) is NOT used: SMARD repeats the hourly price four times before the
# 15-min day-ahead market started (Oct 2025), so the column is exactly 0 for the whole training period (v2 used it by mistake).
INTRAHOUR = ["price_range_q"]
DAYAHEAD_FC = ["fc_solar", "fc_wind_onshore", "fc_load"]   # official SMARD forecasts for the target day
# Known-future inputs of the target hours (D+1), by family:
#  - price_dayD: price of the same hour on day D (= target - 24 h). Day D's auction prices are public at 12:00 of D.
#  - fc_load   : day-ahead load forecast, due two hours before the day-ahead exchange closes (SMARD manual 04/2026).
#  - fc_solar / fc_wind_onshore: due only "by 18:00 of the previous day", vintage at 12:00 unknown -> upper bound only.
DAYD = ["price_dayD"]
FAMILIES = {
    "safe": CALENDAR + ["fc_load"] + DAYD,               # noon-safe (primary result)
    "allfc": CALENDAR + DAYAHEAD_FC + DAYD,              # + wind/solar forecasts (upper bound)
}
FUTURE = FAMILIES["allfc"]
# 2025 is split: Jan-Aug for early stopping / model selection, Sep-Dec only for fitting the conformal band widening.
SPLITS = {"train": ("2019-01-01", "2024-12-31"), "val": ("2025-01-01", "2025-08-31"),
          "cal": ("2025-09-01", "2025-12-31"), "test": ("2026-01-01", None)}


def run_name(model: str, weather: bool, family: str, tag: str = "", seed: int = 0) -> str:
    """Single naming scheme for checkpoints, predictions and logs (used by train.py and run_experiments.py)."""
    return f"{model}{'_weather' if weather else ''}_{family}{('_' + tag) if tag else ''}_s{seed}"


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Columns computed from the dataset itself (no new information): price_dayD = price 24 h earlier."""
    df = df.copy()
    df["price_dayD"] = df["price"].shift(24)
    return df


def load_dataset(path=None) -> pd.DataFrame:
    return add_derived(pd.read_csv(path or ROOT / "data" / "dataset_hourly.csv", index_col=0, parse_dates=True))


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
    df = add_derived(df) if "price_dayD" not in df else df.copy()
    fc = [c for c in DAYAHEAD_FC if c in df]
    df[fc] = df[fc].ffill(limit=48)  # fc_load has a few holes (<= 45 h) in 2022-23; forward fill uses past values only
    scaler = StandardScaler().fit(split_frame(df, "train")[cols].to_numpy())
    dead = [c for c, v in zip(cols, scaler.var_) if v < 1e-12]
    if dead:
        raise ValueError(f"constant input column(s) on the training data: {dead}")
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


def daylight_mask(df: pd.DataFrame, starts: pd.DatetimeIndex, threshold: float = 100.0) -> np.ndarray:
    """(N, 24) bool: target hour is 'daylight' if the TRAINING-period mean solar of its (month, hour) exceeds
    `threshold` MWh. Uses training data only, so the mask does not depend on the scored values."""
    tr = split_frame(df, "train")["solar"]
    clim = tr.groupby([tr.index.month, tr.index.hour]).mean()
    hours = starts.values[:, None] + np.arange(HORIZON) * np.timedelta64(1, "h")
    idx = pd.DatetimeIndex(hours.ravel())
    vals = clim.reindex(list(zip(idx.month, idx.hour))).to_numpy()
    return (vals > threshold).reshape(len(starts), HORIZON)


def truth(df: pd.DataFrame, starts: pd.DatetimeIndex) -> np.ndarray:
    """(N, 24, 2) raw target values (price ct/kWh, solar MWh) of the target days; used for scoring instead of the
    float32 scaled-and-back targets, so night solar is exactly 0."""
    return np.stack([df.loc[t:t + pd.Timedelta(hours=HORIZON - 1), TARGETS].to_numpy(dtype=np.float64) for t in starts])
