"""Smoke tests of the Streamlit app with Streamlit's headless AppTest. The app reads a results folder
(SBA_RESULTS_DIR); here a throw-away one is built from whatever trained runs exist (no real test_results.json)."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNS = sorted(p.stem.removeprefix("val_") for p in (ROOT / "results").glob("val_*_s0.json"))
RUNS = [r for r in RUNS if "family" in json.load(open(ROOT / "results" / f"val_{r}.json"))]   # v3 runs only
pytestmark = pytest.mark.skipif(not RUNS or not (ROOT / "data" / "dataset_hourly.csv").exists(),
                                reason="needs trained runs and the dataset")


@pytest.fixture()
def results_dir(tmp_path, monkeypatch):
    for r in RUNS:
        for name in (f"val_{r}.json", f"pred_{r}.npz"):
            (tmp_path / name).symlink_to(ROOT / "results" / name)
    metric = {t: {m: {"mean": 1.0, "std": 0.0} for m in ("MAE", "RMSE", "pinball", "coverage", "width")} for t in ("price", "solar")}
    cfgs = {r: json.load(open(ROOT / "results" / f"val_{r}.json")) for r in RUNS}
    groups = {}
    for r, c in cfgs.items():                     # one run per v3 group name (model[_weather]_family)
        groups.setdefault(f"{c['model']}{'_weather' if c.get('weather') else ''}_{c.get('family', 'safe')}", r)
    (tmp_path / "test_results.json").write_text(json.dumps({
        "test": {g: metric for g in groups} | {"naive_lastday": metric},
        "val": {}, "val_loss": {g: 0.05 + i / 1000 for i, g in enumerate(groups)}, "val_loss_std": {g: 0.0 for g in groups},
        "groups": {g: [r] for g, r in groups.items()}, "best_runs": groups, "winner": next(iter(groups)),
        "winner_run": next(iter(groups.values())), "ranking": list(groups)}))
    monkeypatch.setenv("SBA_RESULTS_DIR", str(tmp_path))
    return tmp_path


def run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    return at


def test_demo_mode_renders_without_exception(results_dir):
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4 and len(at.metric) >= 4
    assert any("Day-ahead forecast for" in c.value for c in at.caption)


def test_planning_assistant_finds_hours_and_wraps_overnight(results_dir):
    at = run_app()
    assert not at.exception
    at.button[0].click().run()                    # default window: today 12:00 .. tomorrow 07:00
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Chosen hours" in labels and "Estimated cost" in labels
    assert any("EUR" in i.value for i in at.info)        # template text without API key
    chosen = next(m.value for m in at.metric if m.label == "Chosen hours")
    assert chosen != "none"


def test_real_mode_explains_missing_data_instead_of_crashing(results_dir):
    at = run_app()
    at.segmented_control(key="mode").set_value("Real tomorrow").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.warning or len(at.tabs) == 4                # either a forecast or a clear explanation
