"""Post-hoc recalibration of the low-high band (conformalized quantile regression, Romano et al. 2019).

The bands of all models are too narrow (coverage < 0.80, see REPORT.md). For every final run and every target the
widening `delta` is learned on the VALIDATION year only: the conformity score of a day-hour is
max(q10 - y, y - q90); delta is its ceil((n+1)*0.8)/n quantile, and the band becomes [q10 - delta, q90 + delta].
The test set is then scored with that fixed delta (it is never used to fit delta). The middle forecast is unchanged,
so MAE/RMSE do not change; coverage, band width and pinball loss do.

Caveat: this assumes validation (2025) and test (2026) are exchangeable. If 2026 is more volatile than 2025,
coverage stays below the target. This step was added after the test set had been scored once.

Run (after src/run_experiments.py final):  python src/calibrate.py   -> results/calibration.json
Then re-run src/final_eval.py --force to put the table into REPORT.md.
"""
import json
import math

import numpy as np

import final_eval as F
import windows as W
from evaluate import metrics, to_units

TARGET_COVERAGE = 0.8


def conformal_delta(yv: np.ndarray, qv: np.ndarray) -> float:
    """yv: (N, 24) truth, qv: (N, 24, 3) sorted quantiles (same units) -> widening of the band."""
    score = np.maximum(qv[..., 0] - yv, yv - qv[..., 2]).ravel()
    n = len(score)
    level = min(1.0, math.ceil((n + 1) * TARGET_COVERAGE) / n)
    return float(np.quantile(score, level, method="higher"))


def widen(q: np.ndarray, deltas: list[float]) -> np.ndarray:
    """q: (N, 24, 2, 3) sorted; deltas per target."""
    out = np.sort(q, axis=-1).copy()
    for k, d in enumerate(deltas):
        out[:, :, k, 0] -= d
        out[:, :, k, 2] += d
    return out


def run() -> dict:
    cache, result = {}, {}
    for g, runs in F.final_runs().items():
        per_run = []
        for r in runs:
            cfg, res, d, sc, cols, z = F.load_run(r, cache)
            qv = np.sort(to_units(z["val"], sc, cols), axis=-1)
            qt = np.sort(to_units(z["test"], sc, cols), axis=-1)
            yv, yt = to_units(d["val"].Y, sc, cols), to_units(d["test"].Y, sc, cols)
            deltas = [conformal_delta(yv[:, :, k], qv[:, :, k]) for k in range(2)]
            per_run.append({"delta": deltas, "val_after": metrics(yv, widen(qv, deltas)),
                            "test_before": metrics(yt, qt), "test_after": metrics(yt, widen(qt, deltas))})
        result[g] = {"runs": runs,
                     "delta": {t: float(np.mean([p["delta"][k] for p in per_run])) for k, t in enumerate(W.TARGETS)},
                     "test_before": F.agg([p["test_before"] for p in per_run]),
                     "test_after": F.agg([p["test_after"] for p in per_run]),
                     "val_after_coverage": {t: float(np.mean([p["val_after"][t]["coverage"] for p in per_run]))
                                            for t in W.TARGETS}}
    return result


if __name__ == "__main__":
    out = run()
    (F.R / "calibration.json").write_text(json.dumps(out, indent=1))
    for g, v in out.items():
        a, b = v["test_before"], v["test_after"]
        print(f"{F.label(g):22s} price cov {a['price']['coverage']['mean']:.2f}->{b['price']['coverage']['mean']:.2f}  "
              f"solar cov {a['solar']['coverage']['mean']:.2f}->{b['solar']['coverage']['mean']:.2f}")
