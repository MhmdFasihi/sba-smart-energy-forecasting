"""v3 experiment driver. Every decision uses the VALIDATION split (Jan-Aug 2025) only; the test set is never read.

Families of known-future inputs (windows.FAMILIES):
  safe  = calendar + load forecast + day-D prices      -> primary result (all inputs exist at 12:00 of day D)
  allfc = safe + official wind/solar forecasts           -> upper bound (wind/solar vintage at 12:00 unknown)
The rule "safe is primary, allfc is the upper bound" is written to the plan before any test scoring.

Stages (resumable: finished runs are skipped; RUNS jobs in parallel):
  ablate : 3 models x 2 families x {without, with weather} x 2 seeds at lr 1e-3 / dropout 0.1
  tune   : per family (weather as decided) and model: lr {1e-3, 3e-4} x dropout {0.1, 0.2}, 2 seeds each
  final  : the tuned setting with 5 seeds (seeds 0-1 come from tuning)
Run:  python src/run_experiments.py ablate|tune|final [--jobs 2]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windows import run_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MODELS = ["lstm", "cnn", "patchtst"]
FAMILIES = ["safe", "allfc"]
PRIMARY = "safe"
SEARCH_SEEDS = [0, 1]
FINAL_SEEDS = [0, 1, 2, 3, 4]
GRID = [(1e-3, 0.1), (1e-3, 0.2), (3e-4, 0.1), (3e-4, 0.2)]
WEATHER_MIN_GAIN = 0.01          # weather is used only if it lowers the mean validation loss by more than 1 %
PLAN = R / "experiment_plan.json"


def hp_tag(lr: float, dropout: float) -> str:
    return f"lr{lr:g}_do{dropout:g}"


def val_loss(name: str) -> float | None:
    p = R / f"val_{name}.json"
    return json.load(open(p))["best_val_loss"] if p.exists() else None


def train_one(job: tuple) -> tuple[str, float]:
    model, weather, family, lr, dropout, seed = job
    name = run_name(model, weather, family, hp_tag(lr, dropout), seed)
    if val_loss(name) is None:
        cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--model", model, "--family", family, "--seed", str(seed),
               "--lr", str(lr), "--dropout", str(dropout), "--tag", hp_tag(lr, dropout)] + (["--weather"] if weather else [])
        with open(R / f"log_{name}.txt", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    v = val_loss(name)
    print(f"{name}: val loss {v:.4f}", flush=True)
    return name, v


def train_many(jobs: list[tuple], n_jobs: int) -> dict[tuple, float]:
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        out = list(ex.map(train_one, jobs))
    return {j: v for j, (_, v) in zip(jobs, out)}


def load_plan() -> dict:
    return json.load(open(PLAN)) if PLAN.exists() else {}


def save_plan(**kw):
    PLAN.write_text(json.dumps(load_plan() | kw, indent=1))


def ablate(n_jobs: int):
    jobs = [(m, w, f, 1e-3, 0.1, s) for f, m, w, s in product(FAMILIES, MODELS, (False, True), SEARCH_SEEDS)]
    res = train_many(jobs, n_jobs)
    rows, use_weather = {}, {}
    for f in FAMILIES:
        mean = {w: float(np.mean([res[(m, w, f, 1e-3, 0.1, s)] for m in MODELS for s in SEARCH_SEEDS])) for w in (False, True)}
        use_weather[f] = bool(mean[True] < mean[False] * (1 - WEATHER_MIN_GAIN))
        rows[f] = {"without_weather": mean[False], "with_weather": mean[True],
                   "per_model": {f"{m}|w{int(w)}": float(np.mean([res[(m, w, f, 1e-3, 0.1, s)] for s in SEARCH_SEEDS]))
                                 for m in MODELS for w in (False, True)}}
    save_plan(version=3, primary_family=PRIMARY,
              rule="primary result = 'safe' family (inputs available at 12:00 of day D); 'allfc' = upper bound. "
                   "Winner = lowest mean validation loss over 5 seeds within the safe family. Fixed before any v3 test scoring.",
              ablation=rows, use_weather=use_weather, weather_min_gain=WEATHER_MIN_GAIN)
    print("use_weather", use_weather)


def tune(n_jobs: int):
    plan = load_plan()
    jobs = [(m, plan["use_weather"][f], f, lr, do, s) for f, m, (lr, do), s in product(FAMILIES, MODELS, GRID, SEARCH_SEEDS)]
    res = train_many(jobs, n_jobs)
    tuned = {}
    for f, m in product(FAMILIES, MODELS):
        w = plan["use_weather"][f]
        grid = {hp_tag(lr, do): float(np.mean([res[(m, w, f, lr, do, s)] for s in SEARCH_SEEDS])) for lr, do in GRID}
        lr, do = min(GRID, key=lambda g: grid[hp_tag(*g)])
        tuned[f"{f}|{m}"] = {"lr": lr, "dropout": do, "weather": w, "val_loss": grid[hp_tag(lr, do)], "grid": grid}
    save_plan(tuned=tuned)


def final(n_jobs: int):
    plan = load_plan()
    jobs, groups = [], {}
    for f, m in product(FAMILIES, MODELS):
        t = plan["tuned"][f"{f}|{m}"]
        group = f"{m}{'_weather' if t['weather'] else ''}_{f}"
        groups[group] = [run_name(m, t["weather"], f, hp_tag(t["lr"], t["dropout"]), s) for s in FINAL_SEEDS]
        jobs += [(m, t["weather"], f, t["lr"], t["dropout"], s) for s in FINAL_SEEDS]
    train_many(jobs, n_jobs)
    save_plan(final_groups=groups)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ablate", "tune", "final"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    {"ablate": ablate, "tune": tune, "final": final}[a.stage](a.jobs)
