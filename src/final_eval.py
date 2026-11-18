"""Final evaluation. Model choice uses VALIDATION only (lowest mean validation pinball loss over seeds);
the TEST set (2026) is scored exactly once here. The script refuses to run again unless --force is given,
so test_results.json is written once.

All numbers in results/REPORT.md and results/results_tables.md are generated from the metrics computed
here (never typed by hand); src/check_report.py re-checks the written report against test_results.json.

Run:  python src/final_eval.py [--force]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baselines
import windows as W
from evaluate import metrics, to_units

R = W.ROOT / "results"
METRICS = ("MAE", "RMSE", "pinball", "coverage", "width")
LABEL = {"lstm": "LSTM", "cnn": "CNN", "patchtst": "PatchTST", "naive_lastday": "naive last day (D-1)",
         "naive_lastweek": "naive last week", "naive_dayD": "naive day D (price only)", "smard_fc_solar": "SMARD official solar forecast"}


SAFE, ALLFC = "noon-safe", "all fc"      # known-future inputs: calendar + load forecast  |  + official wind/solar forecasts


def family(group: str) -> str:
    return SAFE if "_strict" in group else ALLFC if "_future" in group else ""


def label(group: str) -> str:
    if group in LABEL:
        return LABEL[group]
    model = group.split("_")[0]
    return LABEL[model] + (" + weather" if "_weather" in group else "") + (f" [{family(group)}]" if family(group) else "")


def group_of(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


def final_runs() -> dict[str, list[str]]:
    """group -> run names (seeds) of the final experiment (tag 'final')."""
    groups: dict[str, list[str]] = {}
    for p in sorted(R.glob("val_*_final_s*.json")):
        run = p.stem.removeprefix("val_")
        groups.setdefault(group_of(run), []).append(run)
    return groups


def agg(per_run: list[dict]) -> dict:
    """list of {target: {metric: value}} -> {target: {metric: {mean, std}}} (std over seeds, 0 for one run)."""
    out = {}
    for t in per_run[0]:
        out[t] = {m: {"mean": float(np.mean([r[t][m] for r in per_run])),
                      "std": float(np.std([r[t][m] for r in per_run]))} for m in METRICS}
    return out


def load_run(run: str, cache: dict):
    cfg = json.load(open(R / f"val_{run}.json"))
    key = (tuple(cfg["features"]), tuple(cfg["future"]))
    if key not in cache:
        cache[key] = W.prepare(W.load_dataset(), cfg["features"], cfg["future"])
    d, sc = cache[key]
    z = np.load(R / f"pred_{run}.npz")
    cols = cfg["features"] + [c for c in cfg["future"] if c not in cfg["features"]]
    res = {s: metrics(to_units(d[s].Y, sc, cols), to_units(z[s], sc, cols)) for s in ("val", "test")}
    return cfg, res, d, sc, cols, z


def fmt(m: dict, digits: int, seeds: bool) -> str:
    v = f"{m['mean']:.{digits}f}"
    return v + (f" ± {m['std']:.{digits}f}" if seeds and m["std"] > 0 else "")


def table_md(res: dict, order: list[str]) -> str:
    lines = ["| Model | Price MAE | Price pinball | Price coverage | Solar MAE | Solar pinball | Solar coverage |",
             "|---|---|---|---|---|---|---|"]
    for g in order:
        row = [label(g)]
        for t, dg in (("price", (2, 3, 2)), ("solar", (0, 0, 2))):
            if t not in res[g]:
                row += ["–"] * 3
                continue
            r = res[g][t]
            row += [fmt(r["MAE"], dg[0], True), fmt(r["pinball"], dg[1], True), fmt(r["coverage"], dg[2], False)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def full_table_md(res: dict, order: list[str]) -> str:
    lines = ["model | target | MAE | RMSE | pinball | coverage | width", "--- | --- | --- | --- | --- | --- | ---"]
    for g in order:
        for t in W.TARGETS:
            if t in res[g]:
                m = res[g][t]
                lines.append(f"{label(g)} | {t} | " + " | ".join(
                    f"{m[k]['mean']:.3f}" + (f" ± {m[k]['std']:.3f}" if m[k]["std"] > 0 else "") for k in METRICS))
    return "\n".join(lines)


def main(force: bool):
    out_json = R / "test_results.json"
    if out_json.exists() and not force:
        raise SystemExit("test_results.json exists: the test set was already scored once. Use --force to redo it.")
    groups = final_runs()
    assert groups, "no final runs found (run src/run_experiments.py final)"
    plan = json.load(open(R / "experiment_plan.json"))
    cache, per_group, picked = {}, {}, {}
    for g, runs in groups.items():
        loaded = [load_run(r, cache) for r in runs]
        per_group[g] = {"runs": runs, "val": agg([x[1]["val"] for x in loaded]), "test": agg([x[1]["test"] for x in loaded]),
                        "val_loss": float(np.mean([x[0]["best_val_loss"] for x in loaded])),
                        "val_loss_std": float(np.std([x[0]["best_val_loss"] for x in loaded]))}
    # winner = lowest mean validation pinball loss (sorted-quantile loss used for early stopping)
    # primary result = noon-safe family (rule fixed before seeing its results); all-fc family = upper bound
    ranked = sorted((g for g in per_group if family(g) == SAFE), key=lambda g: per_group[g]["val_loss"])
    ranked_all = sorted((g for g in per_group if family(g) == ALLFC), key=lambda g: per_group[g]["val_loss"])
    winner = ranked[0]
    vl = lambda r: json.load(open(R / f"val_{r}.json"))["best_val_loss"]
    best_runs = {g: min(runs, key=vl) for g, runs in groups.items()}   # best seed per group (by validation)
    best_run = best_runs[winner]

    base = baselines.run()
    val = {g: per_group[g]["val"] for g in per_group} | {b: {t: {m: {"mean": v, "std": 0.0} for m, v in mm.items()}
                                                              for t, mm in base[b]["val"].items()} for b in base}
    test = {g: per_group[g]["test"] for g in per_group} | {b: {t: {m: {"mean": v, "std": 0.0} for m, v in mm.items()}
                                                                for t, mm in base[b]["test"].items()} for b in base}
    result = {"val": val, "test": test, "val_loss": {g: per_group[g]["val_loss"] for g in per_group},
              "val_loss_std": {g: per_group[g]["val_loss_std"] for g in per_group},
              "groups": {g: per_group[g]["runs"] for g in per_group}, "best_runs": best_runs, "winner": winner,
              "winner_run": best_run, "winner_allfc": ranked_all[0], "winner_allfc_run": best_runs[ranked_all[0]],
              "ranking": ranked, "ranking_allfc": ranked_all}
    out_json.write_text(json.dumps(result, indent=1))

    order = list(base) + sorted(per_group, key=lambda g: (g.split("_")[0], family(g) != SAFE, "_weather" in g))
    cfg_w, res_w, d_w, sc_w, cols_w, z_w = load_run(best_run, cache)

    # ---- plots (test) ----
    names = [label(g) for g in order]
    for fname, metric, title, hline in (("test_pinball.png", "pinball", "Test pinball loss", None),
                                        ("test_coverage.png", "coverage", "Test coverage of the low-high band (target 0.80)", 0.8)):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for a, t in zip(ax, W.TARGETS):
            xs = [i for i, g in enumerate(order) if t in test[g]]
            a.bar(xs, [test[order[i]][t][metric]["mean"] for i in xs],
                  yerr=[test[order[i]][t][metric]["std"] for i in xs], color="#1f77b4", capsize=3)
            if hline:
                a.axhline(hline, color="k", ls="--")
            a.set_xticks(range(len(names)), names, rotation=60, ha="right")
            a.set_title(f"{title}: {t}")
        fig.tight_layout()
        fig.savefig(R / fname, dpi=150)
        plt.close(fig)

    # example day: fixed, value-independent rule = the middle test day (not picked from test targets)
    yq = np.sort(to_units(z_w["test"], sc_w, cols_w), axis=-1)
    y = to_units(d_w["test"].Y, sc_w, cols_w)
    i = len(yq) // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (a, t) in enumerate(zip(ax, W.TARGETS)):
        a.fill_between(range(24), yq[i, :, k, 0], yq[i, :, k, 2], alpha=0.3, label="low-high")
        a.plot(yq[i, :, k, 1], label="forecast")
        a.plot(y[i, :, k], "k:", label="actual")
        a.set_title(f"{best_run}: {t} on {d_w['test'].start[i].date()}")
        a.legend()
    fig.tight_layout()
    fig.savefig(R / "example_day.png", dpi=150)
    plt.close(fig)

    # monthly error of the winner vs the naive last-day baseline
    d_b, sc_b = W.prepare(W.load_dataset(), W.BASE, W.FUTURE)
    cols_b = W.BASE + [c for c in W.FUTURE if c not in W.BASE]
    zb = to_units(baselines.naive(d_b["test"].X, "lastday"), sc_b, cols_b)
    months = pd.DatetimeIndex(d_b["test"].start).month
    rows = []
    for m in sorted(set(months)):
        s = months == m
        rows.append((m, np.abs(yq[s, :, 0, 1] - y[s, :, 0]).mean(), np.abs(zb[s, :, 0] - y[s, :, 0]).mean(),
                     np.abs(yq[s, :, 1, 1] - y[s, :, 1]).mean(), np.abs(zb[s, :, 1] - y[s, :, 1]).mean()))
    pd.DataFrame(rows, columns=["month", f"{best_run} price MAE", "naive_lastday price MAE",
                                f"{best_run} solar MAE", "naive_lastday solar MAE"]).round(2).to_csv(
        R / "test_mae_by_month.csv", index=False)

    write_reports(result, plan, order, rows, best_run, d_w["test"].start[i].date())


def write_reports(res: dict, plan: dict, order: list[str], monthly: list, best_run: str, example_date):
    val, test, winner = res["val"], res["test"], res["winner"]
    nets = [g for g in order if g in res["groups"]]
    fams = [(SAFE, [g for g in nets if family(g) == SAFE]), (ALLFC, [g for g in nets if family(g) == ALLFC])]
    sn = fams[0][1]
    n_seeds = len(next(iter(res["groups"].values())))
    zt = np.load(R / f"pred_{best_run}.npz")
    n_test = len(zt["test"])
    last_day = pd.Timestamp(zt["test_start"][-1]).date()
    base_names = [g for g in order if g not in nets]
    best_naive = {t: min((b for b in base_names if t in test[b] and b.startswith("naive")),
                         key=lambda b: test[b][t]["pinball"]["mean"]) for t in W.TARGETS}
    rank_test = {t: sorted(sn, key=lambda g: test[g][t]["pinball"]["mean"]) for t in W.TARGETS}
    ranked = res["ranking"]
    test_rank_winner = {t: rank_test[t].index(winner) + 1 for t in W.TARGETS}

    def p(g, t, m="pinball", split=test):
        return split[g][t][m]["mean"]

    lines = []
    A = lines.append
    A("# Smart Energy Forecasting: results report")
    A("")
    A("_Generated by `src/final_eval.py` from `results/test_results.json`; checked by `src/check_report.py`._")
    A("")
    A("Task: **day-ahead** forecast of the German day-ahead electricity price (ct/kWh) and solar generation (MWh per hour) "
      "for day D+1 (00:00-23:00), made at 12:00 of day D (auction cut-off, 12-36 h ahead), with a low (10 %) / middle (50 %) / "
      f"high (90 %) band, from the last 168 h before 12:00 of day D. Data: SMARD (Bundesnetzagentur | SMARD.de, CC BY 4.0); the "
      "price is built from the native 15-minute day-ahead prices (hourly mean); weather from DWD observations "
      "(Open-Meteo for the last weeks, same time convention). Split: train 2019-2024, validation 2025, "
      f"test 2026-01-01 to {last_day} ({n_test} forecast days). "
      f"Each model is trained with {n_seeds} seeds (mean ± std over seeds). Two input families are reported: **[noon-safe]** = "
      "calendar + the day-ahead load forecast of the target day as known-future inputs (the primary result), and **[all fc]** = "
      "additionally the official day-ahead wind and solar forecasts (an upper bound, see the timing caveat).")
    A("")
    A("## Protocol and model selection (validation only)")
    abl = plan["ablation"]
    fam = lambda w, f, i: float(np.mean([abl[f"{m}|w{w}|f{f}|i{i}"] for m in ("lstm", "cnn", "patchtst")]))
    A(f"- Input ablation (seed 0, mean validation pinball loss over LSTM/CNN/PatchTST): base {fam(0,0,0):.4f}; "
      f"+ weather {fam(1,0,0):.4f}; + known-future inputs {fam(0,1,0):.4f}; + weather + future {fam(1,1,0):.4f}; "
      f"+ within-hour 15-min price range {fam(0,1,1):.4f}. Decision: known-future inputs (official SMARD day-ahead solar / wind / "
      f"load forecasts and calendar of the target day) {'used' if plan['use_future'] else 'not used'}; "
      f"15-min range feature {'used' if plan['use_intrahour'] else 'not used'}.")
    sc_ = plan.get("strict_check")
    if sc_:
        A(f"- Timing check (added later, seed 0, same hyper-parameters): with only the noon-safe known inputs (calendar + load forecast) "
          f"the validation loss is {float(np.mean(list(sc_.values()))):.4f}, i.e. about the same as without any known-future input "
          f"({fam(0,0,0):.4f}) and far above the {fam(0,1,1):.4f} with the wind/solar forecasts. The whole gain of the known-future inputs "
          "therefore comes from the official wind and solar forecasts, whose vintage at 12:00 is not verified (see Limitations). "
          "Rule fixed before the strict models were trained: the noon-safe models are the primary result, the all-forecast models an upper bound.")
    tuned = plan["tuned"]
    A("- Hyper-parameter grid (learning rate × dropout, validation loss, seed 0): " + "; ".join(
        f"{label(m)}: lr {tuned[m]['lr']:g}, dropout {tuned[m]['dropout']:g}" for m in tuned) + ".")
    wa = res["winner_allfc"]
    A(f"- Selection (primary, noon-safe family): lowest mean validation pinball loss over seeds → **{label(winner)}** "
      f"({res['val_loss'][winner]:.4f} ± {res['val_loss_std'][winner]:.4f}); next: " + ", ".join(
          f"{label(g)} {res['val_loss'][g]:.4f}" for g in ranked[1:3]) + f". Best of the all-forecast family (upper bound): "
      f"{label(wa)} ({res['val_loss'][wa]:.4f} ± {res['val_loss_std'][wa]:.4f}). "
      "The test set was scored once afterwards (`final_eval.py` refuses to run twice without `--force`) and was not used for any choice. "
      f"The app default is the best seed of the winner (`{best_run}`).")
    A("")
    A(f"## Test results (2026, scored once, {n_seeds} seeds)")
    A("")
    A(table_md(test, order))
    A("")
    A("(Price in ct/kWh, solar in MWh per hour; target coverage 0.80. Full tables incl. RMSE, band width and validation: "
      "`results_tables.md`; figures: `test_pinball.png`, `test_coverage.png`, `example_day.png` (the middle test day, chosen "
      "independently of the values); monthly errors: `test_mae_by_month.csv`.)")
    A("")
    A("## Findings")
    for fname, fnets in fams:
        for t in W.TARGETS:
            beat = [g for g in fnets if p(g, t) < p(best_naive[t], t)]
            rk = sorted(fnets, key=lambda g: p(g, t))
            A(f"- [{fname}] {t.capitalize()}: {len(beat)} of {len(fnets)} neural variants beat the best naive baseline "
              f"({label(best_naive[t])}, pinball {p(best_naive[t], t):.3f}) on test pinball loss; "
              f"best variant {label(rk[0])} ({p(rk[0], t):.3f}), worst {label(rk[-1])} ({p(rk[-1], t):.3f}).")
    sm = test["smard_fc_solar"]["solar"]
    for fname, fnets in fams:
        best_solar_mae = min(fnets, key=lambda g: test[g]["solar"]["MAE"]["mean"])
        best_solar_pin = min(fnets, key=lambda g: test[g]["solar"]["pinball"]["mean"])
        beats = test[best_solar_pin]["solar"]["pinball"]["mean"] < sm["pinball"]["mean"]
        A(f"- [{fname}] Official SMARD day-ahead solar forecast (benchmark, its band comes from training residuals): MAE {sm['MAE']['mean']:.0f}, "
          f"pinball {sm['pinball']['mean']:.0f}. Best neural solar MAE: {label(best_solar_mae)} {test[best_solar_mae]['solar']['MAE']['mean']:.0f}; "
          f"best neural solar pinball: {label(best_solar_pin)} {test[best_solar_pin]['solar']['pinball']['mean']:.0f} → "
          + ("the best neural model beats the official forecast on pinball." if beats else
             "no neural model beats the official forecast on solar."))
    for fname, fnets in fams:
      for t in W.TARGETS:
        parts = []
        for m in ("lstm", "cnn", "patchtst"):
            a = next((g for g in fnets if g.split("_")[0] == m and "_weather" not in g), None)
            b = next((g for g in fnets if g.split("_")[0] == m and "_weather" in g), None)
            if a and b:
                da, db = test[a][t]["pinball"], test[b][t]["pinball"]
                rel = (db["mean"] - da["mean"]) / da["mean"] * 100
                clear = abs(db["mean"] - da["mean"]) > da["std"] + db["std"]
                nd = 3 if t == "price" else 0
                parts.append(f"{label(m)} {da['mean']:.{nd}f}→{db['mean']:.{nd}f} "
                             f"({rel:+.1f} %, {'larger than' if clear else 'within'} seed noise)")
        if parts:
            A(f"- [{fname}] Weather effect on {t} pinball (without → with weather): " + "; ".join(parts) + ".")
    A(f"- Validation winner {label(winner)} ranks {test_rank_winner['price']} of {len(sn)} noon-safe models on test price pinball and "
      f"{test_rank_winner['solar']} of {len(sn)} on test solar pinball; ranks among top models are close and rest on {n_seeds} seeds.")
    covs = {t: [p(g, t, "coverage") for g in nets] for t in W.TARGETS}
    A(f"- Band calibration (target 0.80): price coverage {min(covs['price']):.2f}-{max(covs['price']):.2f}, solar "
      f"{min(covs['solar']):.2f}-{max(covs['solar']):.2f}.")
    mm = pd.DataFrame(monthly, columns=["m", "wp", "np", "ws", "ns"])
    worse = mm[mm["wp"] > mm["np"]]["m"].tolist()
    A(f"- Winner `{best_run}` vs naive last day (D-1), price MAE by month: worse in {len(worse)} of {len(mm)} test months"
      f"{' (month ' + ', '.join(str(int(m)) for m in worse) + ')' if worse else ''} (see `test_mae_by_month.csv`).")
    vcov = {t: [res["val"][g][t]["coverage"]["mean"] for g in nets] for t in W.TARGETS}
    A(f"- Calibration: coverage was already below the 0.80 target on validation (price {min(vcov['price']):.2f}-{max(vcov['price']):.2f}, "
      f"solar {min(vcov['solar']):.2f}-{max(vcov['solar']):.2f}) and is lower on test (price {min(covs['price']):.2f}-{max(covs['price']):.2f}, "
      f"solar {min(covs['solar']):.2f}-{max(covs['solar']):.2f}): the bands are overconfident, most for price. "
      "The table above uses the raw bands; a post-hoc recalibration on the validation year is shown in the next section.")
    A("")
    cal_path = R / "calibration.json"
    if cal_path.exists():
        cal = json.load(open(cal_path))
        A("## Post-hoc band recalibration (added after the first test scoring)")
        A("Conformalized quantile regression (`src/calibrate.py`): the band is widened by a constant per model and target, "
          "learned on the validation year only (it gives validation coverage 0.80 by construction) and then applied unchanged "
          "to the test set. The middle forecast, MAE and RMSE do not change. The table above stays the primary result.")
        A("")
        A("| Model | Price coverage before → after | Price pinball before → after | Solar coverage before → after | Solar pinball before → after |")
        A("|---|---|---|---|---|")
        for g in order:
            if g in cal:
                a_, b_ = cal[g]["test_before"], cal[g]["test_after"]
                cell = lambda t, m, nd: f"{a_[t][m]['mean']:.{nd}f} → {b_[t][m]['mean']:.{nd}f}"
                A(f"| {label(g)} | {cell('price', 'coverage', 2)} | {cell('price', 'pinball', 3)} | "
                  f"{cell('solar', 'coverage', 2)} | {cell('solar', 'pinball', 0)} |")
        pc = [c["test_after"]["price"]["coverage"]["mean"] for c in cal.values()]
        sc_ = [c["test_after"]["solar"]["coverage"]["mean"] for c in cal.values()]
        A("")
        up = lambda t: sum(c["test_after"][t]["coverage"]["mean"] > c["test_before"][t]["coverage"]["mean"] + 0.005 for c in cal.values())
        A(f"After recalibration the test coverage is price {min(pc):.2f}-{max(pc):.2f}, solar {min(sc_):.2f}-{max(sc_):.2f} "
          f"(target 0.80). Price coverage rises for {up('price')} of {len(cal)} models and solar coverage for {up('solar')} of {len(cal)}; "
          f"pinball loss is lower for {sum(c['test_after'][t]['pinball']['mean'] < c['test_before'][t]['pinball']['mean'] for c in cal.values() for t in ('price', 'solar'))} of {2 * len(cal)} model/target pairs. The price bands remain clearly too narrow on 2026. A likely reason is that 2026 "
          "prices are less like 2025 than a constant widening can absorb (distribution shift); this was not investigated further.")
        A("")
    A("## Deviations from the guide")
    A("- Day-ahead protocol (12:00 cut-off) instead of forecasting from midnight; known-future inputs added (see ablation); the "
      "primary result uses only the inputs verifiably available at noon (calendar + load forecast).")
    A("- PatchTST is a native PyTorch implementation, because the neuralforecast version is univariate (no exogenous inputs).")
    A("- CNN uses kernel 4 (dilations 1-32, receptive field 190 h) so the last step sees the whole 168 h week, 6 layers "
      "as in the guide but **128 filters** (not the guide's 64): all three models share one width setting (128 = LSTM units / "
      "CNN filters / PatchTST d_model). LSTM: 2 layers, 128 units as in the guide.")
    A("- Training: Adam, batch 64, up to 50 epochs, early stopping after 5 epochs, as in the guide; the learning rate and "
      "dropout were chosen on validation from the grid above (so the learning rate is not always 0.001). Trained on a "
      "Mac (Apple MPS), not Google Colab.")
    A("- Libraries: charts in the app use Altair (Streamlit built-in) instead of plotly; neuralforecast is not used (see above).")
    A("- The LLM explanation uses Anthropic or NVIDIA (optional), with a template fallback without API key.")
    A("")
    A("## Limitations")
    A(f"- {n_seeds} seeds and one split; differences of a few percent between models are within noise.")
    A("- Weather was trained on observations, not forecasts, so the solar gain would be smaller in real use (guide section 9). "
      "The last weeks of weather come from Open-Meteo (modelled values), not DWD.")
    A("- Timing of the official forecasts (SMARD user manual 04/2026): the day-ahead load forecast must be delivered two hours before "
      "the day-ahead exchange closes (before 12:00) and is only revised for changes of 10 % or more; the day-ahead wind and solar "
      "forecasts must be delivered 'by 18:00 of the previous day'. SMARD stores one value per hour, not the vintage available at 12:00 "
      "of day D, and this project has no archive of those vintages. A wind/solar forecast delivered after 12:00 would not exist in a "
      "real noon run, so the [all fc] models are an upper bound for the noon protocol. The [noon-safe] models avoid the problem but, "
      "on the test year, do not beat the naive baselines: the skill of this project depends on the wind and solar forecasts.")
    A("- Without weather forecasts as inputs, solar and the wind-driven part of the price cannot be predicted well for tomorrow "
      "(weather in this project is observed, not forecast). Archived weather forecasts for the whole training period would be the "
      "next step to get a noon-safe model with skill; they were not used here.")
    A("- The 'naive day D (price only)' baseline was added after the test set had been scored once (it needs no tuning and "
      "changes no model choice); `final_eval.py` was re-run with `--force` only to add it and to correct report wording. All model predictions are unchanged.")
    A("- Negative prices and 2022 produce large errors. Costs in the planner use wholesale prices only (no taxes or grid fees).")
    (R / "REPORT.md").write_text("\n".join(lines) + "\n")

    md = ["# Results", "", f"Winner by validation pinball loss: **{label(winner)}** (`{winner}`)", "",
          f"## Validation (2025), mean ± std over {n_seeds} seeds", full_table_md(val, order), "",
          f"## Test (2026-01-01 to {last_day}), scored once", full_table_md(test, order), "",
          "## Validation loss ranking (sorted-quantile pinball in scaled space, mean over seeds)"]
    md += [f"{i + 1}. {label(g)}: {res['val_loss'][g]:.4f} ± {res['val_loss_std'][g]:.4f}" for i, g in enumerate(ranked)]
    (R / "results_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
