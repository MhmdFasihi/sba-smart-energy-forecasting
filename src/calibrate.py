"""Split-conformal recalibration of the low-high band (conformalized quantile regression, Romano et al. 2019).

The widening is fitted on the CALIBRATION split (Sep-Dec 2025), which is used for nothing else: early stopping and
all model/hyper-parameter choices use Jan-Aug 2025. For every run, target and target hour h:
    score = max(q10 - y, y - q90) over the calibration days,   delta_h = its ceil((n+1)*0.8)/n quantile,
and the band becomes [q10 - delta_h, q90 + delta_h] (solar clipped at 0). Per-hour deltas respect that night solar
hours have almost no error. The middle forecast (MAE/RMSE) does not change.
Exchangeability caveat: Sep-Dec 2025 and 2026 need not behave alike (season, market changes), so the 0.80 target is
only guaranteed on average under exchangeability.

Run (after the final runs):  python src/calibrate.py  -> results/calibration.json
"""
import json
import math

import numpy as np

import final_eval as F
import windows as W
from evaluate import metrics, postprocess, to_units

TARGET_COVERAGE = 0.8


def conformal_deltas(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """y (N, 24, 2), q (N, 24, 2, 3) sorted, units -> deltas (24, 2)."""
    score = np.maximum(q[..., 0] - y, y - q[..., 2])
    n = score.shape[0]
    level = min(1.0, math.ceil((n + 1) * TARGET_COVERAGE) / n)
    return np.quantile(score, level, axis=0, method="higher")


def widen(q: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    out = postprocess(q).copy()
    out[..., 0] -= deltas
    out[..., 2] += deltas
    return postprocess(out)


def run() -> dict:
    cache, df, result = {}, W.load_dataset(), {}
    for g, runs in F.final_groups().items():
        per_run = []
        for r in runs:
            cfg, d, sc, cols, z = F.load_run(r, cache)
            u = lambda s: (W.truth(df, d[s].start), postprocess(to_units(z[s], sc, cols)))
            (yc, qc), (yt, qt) = u("cal"), u("test")
            dl = W.daylight_mask(df, d["test"].start)
            deltas = conformal_deltas(yc, qc)
            per_run.append({"run": r, "deltas": deltas.tolist(),
                            "cal_after": metrics(yc, widen(qc, deltas)),
                            "test_before": metrics(yt, qt, dl), "test_after": metrics(yt, widen(qt, deltas), dl)})
        result[g] = {"runs": runs, "deltas": {p["run"]: p["deltas"] for p in per_run},
                     "test_before": F.agg([p["test_before"] for p in per_run]),
                     "test_after": F.agg([p["test_after"] for p in per_run]),
                     "cal_after_coverage": {t: float(np.mean([p["cal_after"][t]["coverage"] for p in per_run]))
                                            for t in W.TARGETS}}
    return result


if __name__ == "__main__":
    out = run()
    (F.R / "calibration.json").write_text(json.dumps(out, indent=1))
    for g, v in out.items():
        a, b = v["test_before"], v["test_after"]
        print(f"{F.label(g):30s} price cov {a['price']['coverage']['mean']:.2f}->{b['price']['coverage']['mean']:.2f}  "
              f"solar cov {a['solar']['coverage']['mean']:.2f}->{b['solar']['coverage']['mean']:.2f}")
