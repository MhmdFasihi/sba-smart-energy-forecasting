"""Check results/REPORT.md against the results AND recompute the key numbers independently from the raw files.

1. every row of the test table and of the recalibration table equals the JSON values (rounded as printed);
2. INDEPENDENT recomputation (no project metric code): test MAE / pinball / coverage of every model group from
   results/pred_*.npz and data/dataset_hourly.csv (own de-scaling, sorting, solar clipping, daylight mask), the naive
   and SMARD point-forecast MAEs from lagged raw values, the winner from val_*.json, and the DM test between the two
   family winners;
3. every DM row in the report equals the JSON; seed count, winner name and referenced files are correct.
Exit code 1 on any mismatch.  Run:  python src/check_report.py
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from final_eval import TABLE_COLS, label  # noqa: E402  (labels and column layout only)

QS = np.array([0.1, 0.5, 0.9])


def table_rows(report: str, marker: str) -> dict:
    """rows of the markdown table that follows the line starting with `marker`."""
    lines = report.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(marker))
    rows = {}
    for l in lines[start + 2:]:
        if not l.startswith("|"):
            break
        cells = [c.strip() for c in l.strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def num(cell: str) -> float | None:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", cell)
    return float(m.group(1)) if m else None


def independent_scores(df: pd.DataFrame, run: str, daylight_clim: pd.Series) -> tuple[dict, dict]:
    tr = df.loc["2019-01-01":"2024-12-31"]
    z = np.load(R / f"pred_{run}.npz")
    starts = pd.DatetimeIndex(z["test_start"])
    y = np.stack([df.loc[t:t + pd.Timedelta(hours=23), ["price", "solar"]].to_numpy() for t in starts])
    q = np.sort(z["test"].astype(np.float64), axis=-1)
    out, daily = {}, {}
    for k, c in enumerate(("price", "solar")):
        qq = q[:, :, k] * tr[c].std(ddof=0) + tr[c].mean()
        if c == "solar":
            qq = np.clip(qq, 0, None)
        yy = y[:, :, k]
        d = yy[..., None] - qq
        pin = np.maximum(QS * d, (QS - 1) * d).mean(-1)
        out[c] = {"MAE": np.abs(qq[..., 1] - yy).mean(), "pinball": pin.mean(),
                  "coverage": ((yy >= qq[..., 0]) & (yy <= qq[..., 2])).mean()}
        daily[c] = pin.mean(1)
        if c == "solar":
            hrs = starts.values[:, None] + np.arange(24) * np.timedelta64(1, "h")
            ix = pd.DatetimeIndex(hrs.ravel())
            day = (daylight_clim.reindex(list(zip(ix.month, ix.hour))).to_numpy() > 100).reshape(len(starts), 24)
            out["solar_day"] = {"pinball": pin[day].mean()}
    return out, daily


def dm_p(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    n = len(d)
    lag = int(n ** (1 / 3))
    dc = d - d.mean()
    var = sum(2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1)) + dc @ dc / n
    return math.erfc(abs(d.mean() / math.sqrt(var / n)) / math.sqrt(2))


def check() -> list[str]:
    res = json.load(open(R / "test_results.json"))
    report = (R / "REPORT.md").read_text()
    df = pd.read_csv(ROOT / "data" / "dataset_hourly.csv", index_col=0, parse_dates=True)
    errors = []
    test = res["test"]

    # 1) test table == JSON
    rows = table_rows(report, "| Model | Price MAE")
    if set(rows) != {label(g) for g in test}:
        errors.append(f"test table rows {sorted(rows)} != groups in JSON")
    for g, byt in test.items():
        row = rows.get(label(g))
        if row is None:
            continue
        for k, (t, m, nd) in enumerate(TABLE_COLS):
            if t not in byt:
                if row[k] != "–":
                    errors.append(f"{label(g)} col {k}: expected '–', got {row[k]!r}")
                continue
            v, got = byt[t][m]["mean"], num(row[k])
            if got is None or abs(got - round(v, nd)) > 10 ** -nd * 0.51:
                errors.append(f"{label(g)} {t} {m}: report {row[k]!r} vs JSON {v:.{nd + 2}f}")

    # 2a) independent recomputation of every model group (mean over seeds)
    tr = df.loc["2019-01-01":"2024-12-31", "solar"]
    clim = tr.groupby([tr.index.month, tr.index.hour]).mean()
    daily = {}
    for g, runs in res["groups"].items():
        per = [independent_scores(df, r, clim) for r in runs]
        daily[g] = {c: np.mean([p[1][c] for p in per], axis=0) for c in ("price", "solar")}
        for t, ms in (("price", ("MAE", "pinball", "coverage")), ("solar", ("MAE", "pinball", "coverage")),
                      ("solar_day", ("pinball",))):
            for m in ms:
                mine = float(np.mean([p[0][t][m] for p in per]))
                js = test[g][t][m]["mean"]
                if abs(mine - js) > 1e-6 * max(1.0, abs(js)):
                    errors.append(f"independent recomputation differs: {g} {t} {m}: {mine:.6f} vs JSON {js:.6f}")
    # 2b) baseline point-forecast MAE from lagged raw values
    starts = pd.DatetimeIndex(np.load(R / f"pred_{res['winner_run']}.npz")["test_start"])
    y = np.stack([df.loc[t:t + pd.Timedelta(hours=23), ["price", "solar"]].to_numpy() for t in starts])
    lagged = lambda col, lag: np.stack([df.loc[t - pd.Timedelta(hours=lag):t - pd.Timedelta(hours=lag - 23), col].to_numpy()
                                        for t in starts])
    for name, t, col, lag in (("naive_lastday", "price", "price", 48), ("naive_lastday", "solar", "solar", 48),
                              ("naive_lastweek", "price", "price", 168), ("naive_dayD", "price", "price", 24),
                              ("smard_fc_solar", "solar", "fc_solar", 0)):
        k = 0 if t == "price" else 1
        mine = float(np.abs(lagged(col, lag) - y[:, :, k]).mean())
        js = test[name][t]["MAE"]["mean"]
        if abs(mine - js) > 1e-6 * max(1.0, js):
            errors.append(f"independent baseline MAE differs: {name} {t}: {mine:.4f} vs JSON {js:.4f}")
    # 2c) winner rule from the raw validation files
    vl = {g: np.mean([json.load(open(R / f"val_{r}.json"))["best_val_loss"] for r in runs]) for g, runs in res["groups"].items()}
    safe = min((g for g in vl if g.endswith("_safe")), key=vl.get)
    if safe != res["winner"]:
        errors.append(f"winner should be {safe} (lowest mean validation loss in the safe family), JSON says {res['winner']}")
    # 2d) DM test between the family winners, recomputed
    for t in ("price", "solar"):
        key = f"{res['winner']}|{res['winner_allfc']}|{t}"
        pv = dm_p(daily[res["winner"]][t], daily[res["winner_allfc"]][t])
        if abs(pv - res["dm"][key]["p"]) > 1e-6:
            errors.append(f"DM p-value {key}: independent {pv:.6f} vs JSON {res['dm'][key]['p']:.6f}")

    # 3) DM rows, calibration rows, winner, seeds, files
    lines = report.splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("| Comparison")) + 2
    dm_rows = {}
    for l in lines[i0:]:
        if not l.startswith("|"):
            break
        c = [x.strip() for x in l.strip("|").split("|")]
        dm_rows[(c[0], c[1])] = c
    for key, r in res["dm"].items():
        a, b, t = key.split("|")
        c = dm_rows.get((f"{label(a)} vs {label(b)}", t))
        if c is None or c[2] != f"{r['mean_diff']:+.3f}" or c[3] != f"{r['stat']:+.2f}":
            errors.append(f"DM row missing or wrong: {key}")
    cal_path = R / "calibration.json"
    if cal_path.exists():
        cal = json.load(open(cal_path))
        crow = table_rows(report, "| Model | Price coverage raw")
        for g, c in cal.items():
            exp = [f"{c['test_before'][t][m]['mean']:.{nd}f} → {c['test_after'][t][m]['mean']:.{nd}f}"
                   for t, m, nd in (("price", "coverage", 2), ("price", "pinball", 3), ("solar", "coverage", 2), ("solar", "pinball", 0))]
            if crow.get(label(g)) != exp:
                errors.append(f"recalibration row wrong: {label(g)}")
    if f"**{label(res['winner'])}**" not in report:
        errors.append("winner not named in bold in the report")
    n_seeds = {len(v) for v in res["groups"].values()}
    if len(n_seeds) != 1 or f"{n_seeds.pop()} seeds" not in report:
        errors.append("seed count in the report does not match the runs")
    for f in re.findall(r"`([\w./-]+\.(?:png|csv|md|json|ipynb))`", report):
        if not ((R / f).exists() or (ROOT / f).exists()):
            errors.append(f"referenced file missing: {f}")
    return errors


if __name__ == "__main__":
    errs = check()
    print("report OK (table, recalibration and DM rows match; test metrics recomputed independently)" if not errs
          else "\n".join(errs))
    sys.exit(1 if errs else 0)
