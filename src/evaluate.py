"""Metrics for quantile forecasts. Predictions have shape (N, 24, 2, 3): [price, solar] x [q10, q50, q90]."""
import numpy as np

from windows import TARGETS

QUANTILES = (0.1, 0.5, 0.9)


def to_units(arr: np.ndarray, scaler, features: list[str]) -> np.ndarray:
    """Scaled (N, 24, 2[, Q]) -> ct/kWh and MWh."""
    out = np.array(arr, dtype=np.float64)
    for k, t in enumerate(TARGETS):
        j = features.index(t)
        out[:, :, k] = out[:, :, k] * scaler.scale_[j] + scaler.mean_[j]
    return out


def metrics(y: np.ndarray, q: np.ndarray) -> dict:
    """y: (N, 24, 2) true values, q: (N, 24, 2, 3) predicted quantiles, both in original units."""
    q = np.sort(q, axis=-1)  # no crossing quantiles
    res = {}
    for k, t in enumerate(TARGETS):
        yt, qt = y[:, :, k], q[:, :, k]
        err = qt[..., 1] - yt
        pin = np.mean([np.mean(np.maximum(a * (yt - qt[..., i]), (a - 1) * (yt - qt[..., i])))
                       for i, a in enumerate(QUANTILES)])
        res[t] = {
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "pinball": float(pin),
            "coverage": float(((yt >= qt[..., 0]) & (yt <= qt[..., 2])).mean()),
            "width": float((qt[..., 2] - qt[..., 0]).mean()),
        }
    return res


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
