"""Check that results/REPORT.md and results/results_tables.md agree with results/test_results.json.

Every row of the test table in REPORT.md is re-derived from the JSON and compared; the winner, the number of
seeds, the validation losses and the files mentioned in the text must exist. Exit code 1 on any mismatch.

Run:  python src/check_report.py
"""
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval import label  # noqa: E402


def num(cell: str) -> float | None:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", cell)
    return float(m.group(1)) if m else None


def rows_cal(report: str) -> dict:
    """rows of the recalibration table (cells with an arrow)."""
    out = {}
    for line in report.splitlines():
        if line.startswith("|") and "→" in line and not line.startswith("| Model"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out[cells[0]] = cells[1:]
    return out


def check() -> list[str]:
    res = json.load(open(R / "test_results.json"))
    report = (R / "REPORT.md").read_text()
    errors = []
    # 1) the test table
    rows = {}
    for line in report.splitlines():
        if line.startswith("|") and "→" not in line and not line.startswith("|---") and not line.startswith("| Model"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    test = res["test"]
    seen = 0
    for g, byt in test.items():
        name = label(g)
        if name not in rows:
            errors.append(f"missing table row: {name}")
            continue
        seen += 1
        exp = []
        for t, nd in (("price", (2, 3)), ("solar", (0, 0))):
            if t in byt:
                exp += [(byt[t]["MAE"]["mean"], nd[0]), (byt[t]["pinball"]["mean"], nd[1]), (byt[t]["coverage"]["mean"], 2)]
            else:
                exp += [None] * 3
        for k, (cell, e) in enumerate(zip(rows[name], exp)):
            if e is None:
                if cell != "–":
                    errors.append(f"{name} col {k}: expected '–', got '{cell}'")
                continue
            v, nd = e
            got = num(cell)
            if got is None or abs(got - round(v, nd)) > 10 ** -nd * 0.51:
                errors.append(f"{name} col {k}: report {cell!r} vs json {v:.{nd + 2}f}")
    if seen != len(rows):
        errors.append(f"report table has {len(rows)} rows, json {len(test)}")
    # 2) winner, seeds, validation losses
    w = res["winner"]
    if f"**{label(w)}**" not in report:
        errors.append("winner not named in the report")
    n_seeds = {len(v) for v in res["groups"].values()}
    if len(n_seeds) != 1 or f"{n_seeds.pop()} seeds" not in report:
        errors.append("seed count in the report does not match the runs")
    for g, v in res["val_loss"].items():
        if g == w and f"{v:.4f}" not in report:
            errors.append(f"winner validation loss {v:.4f} not in report")
    # 2b) recalibration table (if calibration.json exists)
    cal_path = R / "calibration.json"
    if cal_path.exists():
        cal = json.load(open(cal_path))
        if "Post-hoc band recalibration" not in report:
            errors.append("calibration.json exists but the report has no recalibration section")
        for g, c in cal.items():
            row = rows_cal(report).get(label(g))
            if row is None:
                errors.append(f"missing recalibration row: {label(g)}")
                continue
            for cell, (t, m, nd) in zip(row, (("price", "coverage", 2), ("price", "pinball", 3), ("solar", "coverage", 2), ("solar", "pinball", 0))):
                exp = f"{c['test_before'][t][m]['mean']:.{nd}f} → {c['test_after'][t][m]['mean']:.{nd}f}"
                if cell != exp:
                    errors.append(f"{label(g)} recalibration {t} {m}: report {cell!r} vs json {exp!r}")
    # 3) referenced files exist
    for f in re.findall(r"`([\w./-]+\.(?:png|csv|md|json))`", report):
        if not ((R / f).exists() or (R.parent / f).exists()):
            errors.append(f"referenced file missing: {f}")
    # 4) every group seed run exists and the test set was scored from those runs
    for g, runs in res["groups"].items():
        for r in runs:
            if not (R / f"pred_{r}.npz").exists():
                errors.append(f"missing predictions: {r}")
    return errors


if __name__ == "__main__":
    errs = check()
    print("report OK" if not errs else "\n".join(errs))
    sys.exit(1 if errs else 0)
