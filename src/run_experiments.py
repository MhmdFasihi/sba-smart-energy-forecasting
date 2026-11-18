"""Experiment driver. Every decision here uses VALIDATION loss only (the test set is never read).

Stages (run in this order, each resumable: finished runs are skipped):
  ablate : seed 0, 3 models x {base, +weather, +future, +weather+future, +intrahour(base+future)}
  tune   : small grid (lr x dropout) per model on the winning input family, seed 0
  final  : 3 seeds x 3 models x {without weather, with weather}, with the tuned hyper-parameters and the
           known-future inputs if the ablation says they help
Run:  python src/run_experiments.py ablate|tune|final
"""
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MODELS = ["lstm", "cnn", "patchtst"]
SEEDS = [0, 1, 2]
PLAN = R / "experiment_plan.json"      # decisions taken from validation (written by tune/final)


def name_of(model, weather=False, future=False, intra=False, tag="", seed=0) -> str:
    return (f"{model}{'_weather' if weather else ''}{'_future' if future else ''}{'_intra' if intra else ''}"
            f"{('_' + tag) if tag else ''}_s{seed}")


def val_loss(name: str) -> float | None:
    p = R / f"val_{name}.json"
    return json.load(open(p))["best_val_loss"] if p.exists() else None


def train(model, weather=False, future=False, intra=False, tag="", seed=0, lr=1e-3, dropout=0.1, hidden=128) -> float:
    name = name_of(model, weather, future, intra, tag, seed)
    if val_loss(name) is None:
        cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--model", model, "--seed", str(seed),
               "--lr", str(lr), "--dropout", str(dropout), "--hidden", str(hidden)]
        cmd += (["--weather"] if weather else []) + (["--future"] if future else []) + (["--intrahour"] if intra else [])
        cmd += (["--tag", tag] if tag else [])
        with open(R / f"log_{name}.txt", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    v = val_loss(name)
    print(f"{name}: val loss {v:.4f}", flush=True)
    return v


def load_plan() -> dict:
    return json.load(open(PLAN)) if PLAN.exists() else {}


def save_plan(**kw):
    plan = load_plan() | kw
    PLAN.write_text(json.dumps(plan, indent=1))


def ablate():
    rows = {}
    for m, (w, f, i) in product(MODELS, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1)]):
        rows[f"{m}|w{w}|f{f}|i{i}"] = train(m, bool(w), bool(f), bool(i))
    mean = lambda w, f, i: float(np.mean([rows[f"{m}|w{w}|f{f}|i{i}"] for m in MODELS]))
    use_future = mean(0, 1, 0) < mean(0, 0, 0)
    use_intra = mean(0, 1, 1) < mean(0, int(use_future), 0) - 1e-4
    save_plan(ablation=rows, use_future=bool(use_future), use_intrahour=bool(use_intra))
    print("use_future", use_future, "use_intrahour", use_intra)


def tune():
    plan = load_plan()
    fut, intra = plan["use_future"], plan["use_intrahour"]
    best = {}
    for m in MODELS:
        res = {}
        for lr, do in product([1e-3, 3e-4], [0.1, 0.2]):
            tag = f"lr{lr:g}_do{do:g}"
            res[(lr, do)] = train(m, False, fut, intra, tag=tag, lr=lr, dropout=do)
        lr, do = min(res, key=res.get)
        best[m] = {"lr": lr, "dropout": do, "val_loss": res[(lr, do)], "grid": {f"{a:g}/{b:g}": v for (a, b), v in res.items()}}
    save_plan(tuned=best)


def final():
    plan = load_plan()
    fut, intra = plan["use_future"], plan["use_intrahour"]
    for seed, m, w in product(SEEDS, MODELS, (False, True)):
        hp = plan["tuned"][m]
        train(m, w, fut, intra, tag="final", seed=seed, lr=hp["lr"], dropout=hp["dropout"])


if __name__ == "__main__":
    {"ablate": ablate, "tune": tune, "final": final}[sys.argv[1]]()
