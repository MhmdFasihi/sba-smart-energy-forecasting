"""Metrics for quantile forecasts. Predictions have shape (N, 24, 2, 3): [price, solar] x [q10, q50, q90].

All scoring goes through `postprocess` (sorted quantiles, solar clipped at 0 MWh), so the loss, the reported metrics
and the app show the same numbers. `daily_pinball` + `dm_test` give a Diebold-Mariano test over forecast days.
"""
import math

import numpy as np

from models import QUANTILES
from windows import TARGETS

SOLAR = TARGETS.index("solar")


def to_units(arr: np.ndarray, scaler, features: list[str]) -> np.ndarray:
    """Scaled (N, 24, 2[, Q]) -> ct/kWh and MWh."""
    out = np.array(arr, dtype=np.float64)
    for k, t in enumerate(TARGETS):
        j = features.index(t)
        out[:, :, k] = out[:, :, k] * scaler.scale_[j] + scaler.mean_[j]
    return out


def postprocess(q: np.ndarray) -> np.ndarray:
    """(N, 24, 2, 3) in units -> sorted quantiles (no crossing), solar clipped at 0 (generation cannot be negative)."""
    q = np.sort(np.asarray(q, dtype=np.float64), axis=-1)
    q[:, :, SOLAR] = np.clip(q[:, :, SOLAR], 0.0, None)
    return q


def clean_truth(y: np.ndarray) -> np.ndarray:
    """De-scaled truth: solar clipped at 0 (removes float32 round-off below 0 at night; raw solar is never negative)."""
    y = np.array(y, dtype=np.float64)
    y[:, :, SOLAR] = np.clip(y[:, :, SOLAR], 0.0, None)
    return y


def _pinball(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """y (...), q (..., 3) -> pinball loss averaged over the quantiles, shape (...)."""
    d = y[..., None] - q
    a = np.asarray(QUANTILES)
    return np.maximum(a * d, (a - 1) * d).mean(axis=-1)


def _scores(yt: np.ndarray, qt: np.ndarray) -> dict:
    err = qt[..., 1] - yt
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "pinball": float(_pinball(yt, qt).mean()),
            "coverage": float(((yt >= qt[..., 0]) & (yt <= qt[..., 2])).mean()),
            "width": float((qt[..., 2] - qt[..., 0]).mean())}


def metrics(y: np.ndarray, q: np.ndarray, daylight: np.ndarray | None = None) -> dict:
    """y: (N, 24, 2) true values, q: (N, 24, 2, 3) predicted quantiles, both in original units.
    daylight: optional (N, 24) bool mask -> adds 'solar_day' (solar scored on daylight hours only)."""
    q, y = postprocess(q), clean_truth(y)
    res = {t: _scores(y[:, :, k], q[:, :, k]) for k, t in enumerate(TARGETS)}
    if daylight is not None:
        res["solar_day"] = _scores(y[:, :, SOLAR][daylight], q[:, :, SOLAR][daylight])
    return res


def daily_pinball(y: np.ndarray, q: np.ndarray, target: str) -> np.ndarray:
    """Per forecast day mean pinball loss of one target, shape (N,)."""
    k = TARGETS.index(target)
    return _pinball(clean_truth(y)[:, :, k], postprocess(q)[:, :, k]).mean(axis=1)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int | None = None) -> dict:
    """Diebold-Mariano test of equal expected loss on paired daily losses (two-sided, normal approximation,
    Newey-West variance with `lag` autocovariances; default floor(N^(1/3))). mean_diff < 0 means A is better."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lag = int(n ** (1 / 3)) if lag is None else lag
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / math.sqrt(var / n) if var > 0 else 0.0
    p = math.erfc(abs(stat) / math.sqrt(2))
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(p), "n": n, "lag": lag}


def table(results: dict[str, dict]) -> str:
    lines = ["model | target | MAE | RMSE | pinball | coverage | width", "--- | --- | --- | --- | --- | --- | ---"]
    for name, r in results.items():
        for t in TARGETS:
            if t not in r:
                continue
            m = r[t]
            lines.append(f"{name} | {t} | {m['MAE']:.3f} | {m['RMSE']:.3f} | {m['pinball']:.3f} | "
                         f"{m['coverage']:.2f} | {m['width']:.2f}")
    return "\n".join(lines)
