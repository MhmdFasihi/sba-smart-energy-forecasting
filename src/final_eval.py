"""Final evaluation (v3). Model choice uses VALIDATION only (Jan-Aug 2025): the winner is the group with the lowest
mean validation pinball loss over seeds within the PRIMARY family ('safe', inputs available at 12:00 of day D), as
fixed in results/experiment_plan.json before scoring. The TEST set (2026) is scored here; the script refuses to run
again unless --force is given (used only to regenerate the report after calibrate.py, with identical numbers).

All numbers in results/REPORT.md are generated here from the computed metrics (never typed by hand);
src/check_report.py re-checks the report and recomputes the test metrics independently from the raw files.

Run:  python src/final_eval.py [--force]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baselines
import windows as W
from evaluate import daily_pinball, dm_test, metrics, postprocess, to_units

R = W.ROOT / "results"
METRICS = ("MAE", "RMSE", "pinball", "coverage", "width")
LABEL = {"lstm": "LSTM", "cnn": "CNN", "patchtst": "PatchTST", "naive_lastday": "naive last day (D-1)",
         "naive_lastweek": "naive last week", "naive_dayD": "naive day D (price only)",
         "smard_fc_solar": "SMARD official solar forecast"}
SAFE, ALLFC = "noon-safe", "all fc"
ALPHA = 0.05


def family(group: str) -> str:
    return SAFE if group.endswith("_safe") else ALLFC if group.endswith("_allfc") else ""


def label(group: str) -> str:
    if group in LABEL:
        return LABEL[group]
    return LABEL[group.split("_")[0]] + (" + weather" if "_weather" in group else "") + f" [{family(group)}]"


def final_groups() -> dict[str, list[str]]:
    return json.load(open(R / "experiment_plan.json"))["final_groups"]


def agg(per_run: list[dict]) -> dict:
    """list of {target: {metric: value}} -> {target: {metric: {mean, std}}} (std over seeds)."""
    return {t: {m: {"mean": float(np.mean([r[t][m] for r in per_run])), "std": float(np.std([r[t][m] for r in per_run]))}
                for m in METRICS} for t in per_run[0]}


def load_run(run: str, cache: dict):
    cfg = json.load(open(R / f"val_{run}.json"))
    key = (tuple(cfg["features"]), tuple(cfg["future"]))
    if key not in cache:
        cache[key] = W.prepare(W.load_dataset(), cfg["features"], cfg["future"])
    d, sc = cache[key]
    cols = cfg["features"] + [c for c in cfg["future"] if c not in cfg["features"]]
    return cfg, d, sc, cols, np.load(R / f"pred_{run}.npz")


def fmt(m: dict, digits: int) -> str:
    return f"{m['mean']:.{digits}f}" + (f" ± {m['std']:.{digits}f}" if m["std"] > 0 else "")


TABLE_COLS = (("price", "MAE", 2), ("price", "pinball", 3), ("price", "coverage", 2),
              ("solar", "MAE", 0), ("solar", "pinball", 0), ("solar", "coverage", 2), ("solar_day", "pinball", 0))


def table_md(res: dict, order: list[str]) -> str:
    lines = ["| Model | Price MAE | Price pinball | Price coverage | Solar MAE | Solar pinball | Solar coverage | "
             "Solar pinball (daylight) |", "|---|---|---|---|---|---|---|---|"]
    for g in order:
        row = [label(g)] + [fmt(res[g][t][m], nd) if t in res[g] else "–" for t, m, nd in TABLE_COLS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def full_table_md(res: dict, order: list[str]) -> str:
    lines = ["model | target | MAE | RMSE | pinball | coverage | width", "--- | --- | --- | --- | --- | --- | ---"]
    for g in order:
        for t in W.TARGETS + ["solar_day"]:
            if t in res[g]:
                lines.append(f"{label(g)} | {t} | " + " | ".join(fmt(res[g][t][k], 3) for k in METRICS))
    return "\n".join(lines)


def main(force: bool):
    out_json = R / "test_results.json"
    if out_json.exists() and not force:
        raise SystemExit("test_results.json exists: the test set was already scored. Use --force to regenerate.")
    plan = json.load(open(R / "experiment_plan.json"))
    groups = plan["final_groups"]
    df = W.load_dataset()
    cache, per_group, daily = {}, {}, {}
    for g, runs in groups.items():
        val_runs, test_runs, dl_p, dl_s = [], [], [], []
        for r in runs:
            cfg, d, sc, cols, z = load_run(r, cache)
            u = lambda s: (W.truth(df, d[s].start), to_units(z[s], sc, cols))
            (yv, qv), (yt, qt) = u("val"), u("test")
            val_runs.append(metrics(yv, qv, W.daylight_mask(df, d["val"].start)))
            test_runs.append(metrics(yt, qt, W.daylight_mask(df, d["test"].start)))
            dl_p.append(daily_pinball(yt, qt, "price"))
            dl_s.append(daily_pinball(yt, qt, "solar"))
        vl = [json.load(open(R / f"val_{r}.json"))["best_val_loss"] for r in runs]
        per_group[g] = {"val": agg(val_runs), "test": agg(test_runs), "val_loss": float(np.mean(vl)),
                        "val_loss_std": float(np.std(vl)), "best_run": runs[int(np.argmin(vl))]}
        daily[g] = {"price": np.mean(dl_p, axis=0), "solar": np.mean(dl_s, axis=0)}   # seed-averaged daily loss

    # winner: lowest mean validation loss within the primary family (rule in experiment_plan.json)
    rank = lambda fam: sorted((g for g in groups if family(g) == fam), key=lambda g: per_group[g]["val_loss"])
    ranked, ranked_all = rank(SAFE), rank(ALLFC)
    winner, winner_all = ranked[0], ranked_all[0]

    # baselines (metrics + daily losses); the "best naive" per target is chosen on VALIDATION
    bpred = baselines.predictions(df)
    base_res = baselines.run(df)
    test_starts = W.prepare(df, W.BASE, W.FUTURE)[0]["test"].start
    for b, by in bpred.items():
        yt, qt = by["test"]
        daily[b] = {t: daily_pinball(np.nan_to_num(yt), np.nan_to_num(qt), t) for t in baselines.TARGETS_OF[b]}
    best_naive = {t: min((b for b in base_res if b.startswith("naive") and t in base_res[b]["val"]),
                         key=lambda b: base_res[b]["val"][t]["pinball"]) for t in W.TARGETS}

    # Diebold-Mariano tests on seed-averaged daily pinball loss (negative mean_diff = first model better)
    dm = {}
    for g in groups:
        for t in W.TARGETS:
            dm[f"{g}|{best_naive[t]}|{t}"] = dm_test(daily[g][t], daily[best_naive[t]][t])
    dm[f"{winner}|smard_fc_solar|solar"] = dm_test(daily[winner]["solar"], daily["smard_fc_solar"]["solar"])
    dm[f"{winner_all}|smard_fc_solar|solar"] = dm_test(daily[winner_all]["solar"], daily["smard_fc_solar"]["solar"])
    for t in W.TARGETS:
        dm[f"{winner}|{winner_all}|{t}"] = dm_test(daily[winner][t], daily[winner_all][t])

    pack = lambda res: {t: {m: {"mean": v, "std": 0.0} for m, v in mm.items()} for t, mm in res.items()}
    val = {g: per_group[g]["val"] for g in groups} | {b: pack(base_res[b]["val"]) for b in base_res}
    test = {g: per_group[g]["test"] for g in groups} | {b: pack(base_res[b]["test"]) for b in base_res}
    result = {"version": 3, "val": val, "test": test,
              "val_loss": {g: per_group[g]["val_loss"] for g in groups},
              "val_loss_std": {g: per_group[g]["val_loss_std"] for g in groups},
              "groups": groups, "best_runs": {g: per_group[g]["best_run"] for g in groups},
              "winner": winner, "winner_run": per_group[winner]["best_run"],
              "winner_allfc": winner_all, "winner_allfc_run": per_group[winner_all]["best_run"],
              "ranking": ranked, "ranking_allfc": ranked_all, "best_naive": best_naive, "dm": dm,
              "test_days": [str(test_starts[0].date()), str(test_starts[-1].date()), len(test_starts)]}
    out_json.write_text(json.dumps(result, indent=1))

    order = list(base_res) + sorted(groups, key=lambda g: (family(g) != SAFE, g.split("_")[0], "_weather" in g))
    make_plots(result, order, cache, df)
    monthly = monthly_table(result, cache, bpred, df)
    write_reports(result, plan, order, monthly)


def make_plots(res: dict, order: list[str], cache: dict, df: pd.DataFrame):
    test = res["test"]
    names = [label(g) for g in order]
    for fname, metric, title, hline in (("test_pinball.png", "pinball", "Test pinball loss", None),
                                        ("test_coverage.png", "coverage", "Test coverage of the low-high band (target 0.80)", 0.8)):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for a, t in zip(ax, W.TARGETS):
            xs = [i for i, g in enumerate(order) if t in test[g]]
            a.bar(xs, [test[order[i]][t][metric]["mean"] for i in xs], yerr=[test[order[i]][t][metric]["std"] for i in xs],
                  color=["#1f77b4" if family(order[i]) == SAFE else "#ff7f0e" if family(order[i]) else "#7f7f7f" for i in xs],
                  capsize=3)
            if hline:
                a.axhline(hline, color="k", ls="--")
            a.set_xticks(range(len(names)), names, rotation=60, ha="right")
            a.set_title(f"{title}: {t} (blue = noon-safe, orange = all fc, grey = baselines)", fontsize=9)
        fig.tight_layout()
        fig.savefig(R / fname, dpi=150)
        plt.close(fig)
    # example day: fixed, value-independent rule = the middle test day
    cfg, d, sc, cols, z = load_run(res["winner_run"], cache)
    yq, y = postprocess(to_units(z["test"], sc, cols)), W.truth(df, d["test"].start)
    i = len(yq) // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (a, t) in enumerate(zip(ax, W.TARGETS)):
        a.fill_between(range(24), yq[i, :, k, 0], yq[i, :, k, 2], alpha=0.3, label="low-high")
        a.plot(yq[i, :, k, 1], label="forecast")
        a.plot(y[i, :, k], "k:", label="actual")
        a.set_title(f"{res['winner_run']}: {t} on {d['test'].start[i].date()}")
        a.legend()
    fig.tight_layout()
    fig.savefig(R / "example_day.png", dpi=150)
    plt.close(fig)


def monthly_table(res: dict, cache: dict, bpred: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Price MAE by test month: winner (best seed) vs the best naive price baseline."""
    cfg, d, sc, cols, z = load_run(res["winner_run"], cache)
    yq, y = postprocess(to_units(z["test"], sc, cols)), W.truth(df, d["test"].start)
    nb = res["best_naive"]["price"]
    _, qb = bpred[nb]["test"]
    months = pd.DatetimeIndex(d["test"].start).month
    rows = [(m, np.abs(yq[months == m, :, 0, 1] - y[months == m, :, 0]).mean(),
             np.abs(qb[months == m, :, 0, 1] - y[months == m, :, 0]).mean()) for m in sorted(set(months))]
    out = pd.DataFrame(rows, columns=["month", f"{res['winner_run']} price MAE", f"{nb} price MAE"]).round(3)
    out.to_csv(R / "test_mae_by_month.csv", index=False)
    return out


def write_reports(res: dict, plan: dict, order: list[str], monthly: pd.DataFrame):
    val, test, winner, wa = res["val"], res["test"], res["winner"], res["winner_allfc"]
    groups = res["groups"]
    nets = [g for g in order if g in groups]
    fams = [(SAFE, [g for g in nets if family(g) == SAFE]), (ALLFC, [g for g in nets if family(g) == ALLFC])]
    n_seeds = len(next(iter(groups.values())))
    first, last, n_test = res["test_days"]
    dm, bn = res["dm"], res["best_naive"]
    p = lambda g, t, m="pinball": test[g][t][m]["mean"]
    sig = lambda r: f"p = {r['p']:.3f}" if r["p"] >= 0.001 else "p < 0.001"
    lines = []
    A = lines.append
    A("# Smart Energy Forecasting: results report (v3)")
    A("")
    A("_Generated by `src/final_eval.py` from `results/test_results.json`; checked by `src/check_report.py`, which also "
      "recomputes the test metrics independently from the raw prediction and data files._")
    A("")
    A("Task: **day-ahead** forecast of the German day-ahead electricity price (ct/kWh) and solar generation (MWh per hour) for "
      "day D+1 (00:00-23:00), made at 12:00 of day D (auction cut-off), with a low (10 %) / middle (50 %) / high (90 %) band, "
      "from the last 168 h before 12:00 of day D plus known-future inputs of the target day. Data: SMARD (Bundesnetzagentur | "
      "SMARD.de, CC BY 4.0), hourly price = mean of the 15-minute day-ahead prices; weather history from DWD observations "
      "(Open-Meteo for the last weeks). Split: train 2019-2024, validation Jan-Aug 2025 (early stopping, all choices), "
      f"calibration Sep-Dec 2025 (band recalibration only), test {first} to {last} ({n_test} forecast days). "
      f"Every final model is trained with {n_seeds} seeds (mean ± std over seeds).")
    A("")
    A("## Version history (read first)")
    A("- v1 forecast from midnight; v2 introduced the 12:00 cut-off. Both versions scored the 2026 test year, so **2026 is not an "
      "untouched hold-out any more**: the v3 changes below were motivated by reviews of v2, partly by its test results. "
      "Within v3, every decision is taken on validation, and the selection rule was written to `experiment_plan.json` "
      "before the v3 test scoring.")
    A("- v3 fixes: (1) the 15-minute price-range input used in v2 was constant (0) for the whole training period (SMARD "
      "repeats the hourly price before the 15-minute market started in Oct 2025) and is removed; (2) the noon-safe models now "
      "get the day-D auction prices of the target hours (public at 12:00 of D), which v2 withheld; (3) each input family is "
      "tuned separately with 2 seeds per setting and the finals use 5 seeds; (4) Diebold-Mariano tests; (5) solar "
      "quantiles are clipped at 0 and solar is also scored on daylight hours; (6) the band recalibration uses a separate "
      "calibration split and per-hour widening; (7) CNN with the guide's 64 filters; gradient clipping and an LR scheduler.")
    A("")
    A("## Input families")
    A("- **[noon-safe]** (primary result): calendar of the target day, the official day-ahead **load** forecast (due two hours "
      "before the day-ahead exchange closes, SMARD manual 04/2026) and the **day-D prices** of the same hours (public at 12:00 of D).")
    A("- **[all fc]** (upper bound): additionally the official day-ahead **wind and solar** forecasts, which are only due "
      "'by 18:00 of the previous day'. Their version at 12:00 is not available in the data, so these models may use "
      "information a real noon forecast would not have.")
    A("")
    A("## Protocol and model selection (validation only)")
    ab = plan["ablation"]
    for f in ("safe", "allfc"):
        A(f"- Weather ablation [{SAFE if f == 'safe' else ALLFC}] (lr 1e-3, dropout 0.1, mean over 3 models x 2 seeds): without "
          f"weather {ab[f]['without_weather']:.4f}, with weather {ab[f]['with_weather']:.4f} → weather "
          f"{'used' if plan['use_weather'][f] else 'not used'} (rule: used only if it lowers the loss by more than "
          f"{plan['weather_min_gain'] * 100:.0f} %).")
    A("- Hyper-parameters (lr × dropout grid, 2 seeds each, chosen per family and model): " + "; ".join(
        f"{LABEL[k.split('|')[1]]} [{SAFE if k.startswith('safe') else ALLFC}] lr {v['lr']:g} / dropout {v['dropout']:g}"
        for k, v in plan["tuned"].items()) + ".")
    A(f"- Selection rule ({plan['rule']}) → **{label(winner)}** (validation loss {res['val_loss'][winner]:.4f} ± "
      f"{res['val_loss_std'][winner]:.4f}); next: " + ", ".join(f"{label(g)} {res['val_loss'][g]:.4f}" for g in res["ranking"][1:])
      + f". Best of the upper-bound family: {label(wa)} ({res['val_loss'][wa]:.4f} ± {res['val_loss_std'][wa]:.4f}). "
      f"The app default is the best seed of the winner (`{res['winner_run']}`).")
    A("")
    A(f"## Test results ({first} to {last}, {n_seeds} seeds)")
    A("")
    A(table_md(test, order))
    A("")
    A("(Price in ct/kWh, solar in MWh per hour; target coverage 0.80; solar quantiles clipped at 0. 'Daylight' = target hours "
      "whose training-period mean solar for that month and hour exceeds 100 MWh. Full tables incl. RMSE, band width and "
      "validation: `results_tables.md`; figures: `test_pinball.png`, `test_coverage.png`, `example_day.png` (the middle test "
      "day); monthly errors: `test_mae_by_month.csv`.)")
    A("")
    A("## Significance (Diebold-Mariano, daily pinball loss, seed-averaged, Newey-West variance, two-sided)")
    A("")
    A("| Comparison | Target | Mean daily loss difference | DM statistic | p-value |")
    A("|---|---|---|---|---|")
    for key, r in dm.items():
        a, b, t = key.split("|")
        A(f"| {label(a)} vs {label(b)} | {t} | {r['mean_diff']:+.3f} | {r['stat']:+.2f} | {sig(r)} |")
    A("")
    A("(Negative difference = the first model has the lower loss. 'Best naive' per target is chosen on validation: "
      f"price {label(bn['price'])}, solar {label(bn['solar'])}.)")
    A("")
    A("## Findings")
    for fname, fn in fams:
        for t in W.TARGETS:
            better = [g for g in fn if dm[f"{g}|{bn[t]}|{t}"]["mean_diff"] < 0]
            signif = [g for g in better if dm[f"{g}|{bn[t]}|{t}"]["p"] < ALPHA]
            rk = sorted(fn, key=lambda g: p(g, t))
            nd = 3 if t == "price" else 0
            A(f"- [{fname}] {t.capitalize()}: {len(better)} of {len(fn)} model groups have a lower test pinball loss than "
              f"{label(bn[t])} ({p(bn[t], t):.{nd}f}); {len(signif)} of them significantly (DM, p < {ALPHA}). Best: "
              f"{label(rk[0])} ({p(rk[0], t):.{nd}f}), worst: {label(rk[-1])} ({p(rk[-1], t):.{nd}f}).")
    for g in (winner, wa):
        r = dm[f"{g}|smard_fc_solar|solar"]
        A(f"- Solar vs the official SMARD forecast (pinball {p('smard_fc_solar', 'solar'):.0f}, daylight "
          f"{p('smard_fc_solar', 'solar_day'):.0f}): {label(g)} {p(g, 'solar'):.0f} (daylight {p(g, 'solar_day'):.0f}), "
          f"{'better' if r['mean_diff'] < 0 else 'worse'} ({sig(r)}).")
    for t in W.TARGETS:
        r = dm[f"{winner}|{wa}|{t}"]
        nd = 3 if t == "price" else 0
        A(f"- Value of the wind/solar forecasts ({t}): {label(winner)} {p(winner, t):.{nd}f} vs {label(wa)} {p(wa, t):.{nd}f} "
          f"({sig(r)}).")
    rank_w = sorted(fams[0][1], key=lambda g: p(g, "price")).index(winner) + 1
    A(f"- The validation winner {label(winner)} ranks {rank_w} of {len(fams[0][1])} noon-safe groups on test price pinball.")
    cols = list(monthly.columns)
    worse = monthly[monthly[cols[1]] > monthly[cols[2]]]["month"].tolist()
    A(f"- Winner `{res['winner_run']}` vs {label(bn['price'])}, price MAE by month: worse in {len(worse)} of {len(monthly)} "
      f"test months{' (month ' + ', '.join(str(int(m)) for m in worse) + ')' if worse else ''} (`test_mae_by_month.csv`).")
    covs = {t: [p(g, t, "coverage") for g in nets] for t in W.TARGETS}
    vcov = {t: [val[g][t]["coverage"]["mean"] for g in nets] for t in W.TARGETS}
    A(f"- Raw band coverage (target 0.80): validation price {min(vcov['price']):.2f}-{max(vcov['price']):.2f}, solar "
      f"{min(vcov['solar']):.2f}-{max(vcov['solar']):.2f}; test price {min(covs['price']):.2f}-{max(covs['price']):.2f}, solar "
      f"{min(covs['solar']):.2f}-{max(covs['solar']):.2f}. The recalibrated bands are in the next section.")
    A("")
    cal_path = R / "calibration.json"
    if cal_path.exists():
        cal = json.load(open(cal_path))
        A("## Band recalibration (split conformal, fitted on the calibration split Sep-Dec 2025)")
        A("Per run, target and target hour, the band is widened by the conformal quantile of max(q10 - y, y - q90) on the "
          "calibration days (`src/calibrate.py`); the calibration split is used for nothing else. The middle forecast does "
          "not change. The test table above uses the raw bands.")
        A("")
        A("| Model | Price coverage raw → recalibrated | Price pinball raw → recalibrated | Solar coverage raw → recalibrated | "
          "Solar pinball raw → recalibrated |")
        A("|---|---|---|---|---|")
        for g in nets:
            if g in cal:
                a_, b_ = cal[g]["test_before"], cal[g]["test_after"]
                c = lambda t, m, nd: f"{a_[t][m]['mean']:.{nd}f} → {b_[t][m]['mean']:.{nd}f}"
                A(f"| {label(g)} | {c('price', 'coverage', 2)} | {c('price', 'pinball', 3)} | {c('solar', 'coverage', 2)} | "
                  f"{c('solar', 'pinball', 0)} |")
        pc = [c["test_after"]["price"]["coverage"]["mean"] for c in cal.values()]
        sc_ = [c["test_after"]["solar"]["coverage"]["mean"] for c in cal.values()]
        A("")
        rng = lambda v: f"{min(v):.2f}" if f"{min(v):.2f}" == f"{max(v):.2f}" else f"{min(v):.2f}-{max(v):.2f}"
        A(f"Recalibrated test coverage: price {rng(pc)}, solar {rng(sc_)} (target 0.80; on the calibration split itself price "
          f"{rng([c['cal_after_coverage']['price'] for c in cal.values()])}, solar "
          f"{rng([c['cal_after_coverage']['solar'] for c in cal.values()])}). The correction is per hour and can be negative "
          "(narrower band where the raw band covered more than 80 % of the calibration days) or positive; mean price correction "
          "per model: " + ", ".join(f"{label(g)} {np.mean([np.asarray(d)[:, 0] for d in cal[g]['deltas'].values()]):+.2f} ct/kWh"
                                    for g in nets if g in cal) + ". Where test coverage stays below 0.80, the 2026 errors were "
          "larger than those of Sep-Dec 2025; the conformal guarantee only holds if the two periods behave alike.")
        A("")
    A("## Deviations from the guide")
    A("- Day-ahead protocol with a 12:00 cut-off instead of forecasting from midnight; known-future inputs of the target day.")
    A("- PatchTST is a native PyTorch implementation (the neuralforecast version is univariate, no exogenous inputs).")
    A("- CNN: 6 layers, 64 filters as in the guide, kernel 4 (dilations 1-32, receptive field 190 h ≥ 168 h). LSTM: 2 layers, "
      "128 units as in the guide.")
    A("- Training: Adam, batch 64, up to 50 epochs, early stopping after 5 epochs as in the guide, plus gradient clipping "
      "(norm 1) and halving the learning rate after 2 epochs without improvement; learning rate and dropout tuned on "
      "validation. Trained on Apple MPS, not Google Colab (`colab_training.ipynb` runs the same chain).")
    A("- Validation is Jan-Aug 2025; Sep-Dec 2025 is reserved for band recalibration.")
    A("- Charts in the app use Altair instead of plotly. The LLM explanation uses Anthropic or NVIDIA (optional), with a "
      "template fallback.")
    A("")
    A("## Limitations")
    A("- The 2026 test year was already scored in v1 and v2 (see version history); a fully fresh hold-out would need data "
      "after 2026-11.")
    A(f"- {n_seeds} seeds and one split; the DM tests treat the seed-averaged forecast as one forecaster.")
    A("- Weather enters only as observed history up to D 11:00 (no weather forecast for D+1), so it carries little "
      "information about tomorrow. Archived weather forecasts with a day-ahead lead time were checked: the Open-Meteo "
      "previous-runs archive starts in 2024, too late for the 2019-2024 training period.")
    A("- The [all fc] scores depend on the official wind/solar forecasts, whose vintage at 12:00 is unknown (see Input families).")
    A("- Timestamps are local wall-clock time from SMARD: the repeated October hour is dropped and the missing March hour is "
      "filled, which shifts a few windows by one hour around the clock changes. Holidays are national only.")
    A("- Negative prices and the 2022 crisis produce large errors. Planner costs use wholesale prices only (no taxes or "
      "grid fees).")
    (R / "REPORT.md").write_text("\n".join(lines) + "\n")
    (R / "results_tables.md").write_text("## Test\n\n" + full_table_md(test, order) + "\n\n## Validation (Jan-Aug 2025)\n\n"
                                         + full_table_md(val, order) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate an existing test_results.json")
    main(ap.parse_args().force)
