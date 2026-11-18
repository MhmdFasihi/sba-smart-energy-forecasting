"""Smart Energy Forecasting: day-ahead price and solar forecast for Germany + planning assistant.

Run:  streamlit run app.py
Demo mode: pick any forecast day of the validation (2025) or test (2026) period and compare forecast vs reality.
Real mode: forecast the next day from the newest SMARD files in data/ (see README).
Data: SMARD (Bundesnetzagentur | SMARD.de, CC BY 4.0); weather: DWD / Open-Meteo.
"""
import json
import os
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SBA_RESULTS_DIR", ROOT / "results"))   # overridable for tests
sys.path.insert(0, str(ROOT / "src"))
import planner  # noqa: E402
from evaluate import to_units  # noqa: E402

st.set_page_config(page_title="Smart Energy Forecasting", page_icon=":material/bolt:", layout="wide")
PRICE_COLOR, SOLAR_COLOR = "#1f77b4", "#e69f00"


# ---------------------------------------------------------------- data loading (cached)
@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "dataset_hourly.csv", index_col=0, parse_dates=True)


@st.cache_data
def load_results() -> dict | None:
    p = RESULTS / "test_results.json"
    return json.load(open(p)) if p.exists() else None


@st.cache_data
def load_predictions(run: str) -> dict:
    """forecast day (date of D+1) -> (24, 2, 3) price/solar x low/mid/high in ct/kWh and MWh."""
    cfg = json.load(open(RESULTS / f"val_{run}.json"))
    sc = joblib.load(ROOT / "models" / cfg["scaler"])
    z = np.load(RESULTS / f"pred_{run}.npz")
    out = {}
    for split in ("val", "test"):
        units = np.sort(to_units(z[split], sc["scaler"], sc["cols"]), axis=-1)  # low <= mid <= high
        for i, t in enumerate(pd.DatetimeIndex(z[f"{split}_start"])):
            out[t.normalize()] = units[i]
    return out


@st.cache_resource(show_spinner="Loading the newest SMARD data and the model ...")
def real_forecast(run: str, data_signature: tuple):
    import realtime

    return realtime.forecast_next_day(run)


def data_signature() -> tuple:
    return tuple((p.name, p.stat().st_mtime) for p in sorted((ROOT / "data").glob("*.csv")))


def label(group: str) -> str:
    from final_eval import label as _l

    return _l(group)


# ---------------------------------------------------------------- charts
def band_chart(x: pd.DatetimeIndex, q: np.ndarray, unit: str, color: str, actual=None, known=None,
               chosen=None, height: int = 300) -> alt.Chart:
    d = pd.DataFrame({"time": x, "low": q[:, 0], "forecast": q[:, 1], "high": q[:, 2]})
    base = alt.Chart(d).encode(x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")))
    layers = [base.mark_area(opacity=0.25, color=color).encode(y=alt.Y("low:Q", title=unit), y2="high:Q"),
              base.mark_line(color=color, strokeWidth=2.5).encode(
                  y="forecast:Q", tooltip=[alt.Tooltip("time:T", format="%a %H:%M"), alt.Tooltip("low:Q", format=".1f"),
                                           alt.Tooltip("forecast:Q", format=".1f"), alt.Tooltip("high:Q", format=".1f")])]
    if actual is not None:
        da = pd.DataFrame({"time": x, "actual": np.asarray(actual, dtype=float)})
        layers.append(alt.Chart(da).mark_line(color="black", strokeDash=[3, 3], strokeWidth=2).encode(
            x="time:T", y="actual:Q", tooltip=[alt.Tooltip("actual:Q", format=".1f")]))
    if known is not None:
        layers.append(alt.Chart(known).mark_line(color="gray", strokeWidth=2.5).encode(
            x="time:T", y="known:Q", tooltip=[alt.Tooltip("known:Q", format=".1f", title="known auction price")]))
    if chosen is not None and len(chosen):
        layers.append(alt.Chart(chosen).mark_point(color="crimson", size=120, filled=True).encode(
            x="time:T", y="price:Q", tooltip=[alt.Tooltip("time:T", format="%a %H:%M"), alt.Tooltip("price:Q", format=".1f")]))
    return alt.layer(*layers).properties(height=height)


# ---------------------------------------------------------------- sidebar
results = load_results()
if results is None:
    st.error("No evaluation results found. Run src/run_experiments.py and src/final_eval.py first.", icon=":material/error:")
    st.stop()
df = load_dataset()
groups = results["best_runs"]                       # group -> best seed (by validation)

with st.sidebar:
    st.header("Settings")
    mode = st.segmented_control("Mode", ["Demo", "Real tomorrow"], default="Demo", key="mode",
                                help="Demo: pick a past day and compare forecast and reality. Real tomorrow: forecast the "
                                     "next day from the newest SMARD files in data/.")
    # primary (noon-safe) models first, the all-forecast upper-bound models after them; each sorted by validation loss
    ordered = sorted(groups, key=lambda g: ("_future" in g, results["val_loss"][g]))
    names = {g: label(g) + ("  (validation winner)" if g == results["winner"] else "") for g in ordered}
    group = st.selectbox("Model", ordered, format_func=names.get, help="Sorted by validation pinball loss.")
    run = groups[group]
    if mode == "Demo":
        preds = load_predictions(run)
        days = sorted(preds)
        day = st.date_input("Day to forecast", value=days[-1], min_value=days[0], max_value=days[-1])
        st.caption(f"Demo: forecast days {days[0].date()} to {days[-1].date()} (2025 = validation, 2026 = test).")
    st.caption("Data: Bundesnetzagentur | SMARD.de (CC BY 4.0)")

# ---------------------------------------------------------------- the forecast shown on every tab
actual = None
known_today = None
if mode == "Demo":
    d1 = pd.Timestamp(day)
    if d1 not in preds:
        st.warning("No forecast for this day (needs a full 24 h of data).", icon=":material/warning:")
        st.stop()
    q = preds[d1]
    origin = d1 - pd.Timedelta(hours=12)
    seg = df.loc[d1:d1 + pd.Timedelta(hours=23), ["price", "solar"]]
    actual = seg if len(seg) == 24 else None
    kp = df.loc[origin:origin + pd.Timedelta(hours=11), "price"]
    known_today = kp if len(kp) == 12 else None
else:
    try:
        fc = real_forecast(run, data_signature())
    except Exception as e:  # missing data must be explained, not crash the app
        st.warning(f"Real mode is not possible with the files in data/: {e}", icon=":material/cloud_download:")
        st.stop()
    d1, q, origin, known_today = fc.day, fc.q, fc.origin, fc.known_price_today
price_q, solar_q = q[:, 0, :], q[:, 1, :]
x = pd.date_range(d1, periods=24, freq="h")

st.title("Smart energy forecasting")
st.caption(f"Day-ahead forecast for **{d1:%A, %d %B %Y}**, made at {origin:%d %b %H:%M} (auction cut-off), "
           f"model: {label(group)}, run `{run}`.")

if "_future" in group:
    st.warning("This model also uses the official day-ahead wind and solar forecasts of the target day. Their version available at "
               "12:00 is not verified, so its scores are an upper bound for a real noon forecast.", icon=":material/warning:")
else:
    _t, _naive = results["test"], [b for b in results["test"] if b.startswith("naive")]
    _msg = []
    for _k, _n in (("price", "price"), ("solar", "solar")):
        _best = min((b for b in _naive if _k in _t[b]), key=lambda b: _t[b][_k]["pinball"]["mean"])
        _m, _b = _t[group][_k]["pinball"]["mean"], _t[_best][_k]["pinball"]["mean"]
        _msg.append(f"{_n}: {'better' if _m < _b else 'not better'} than the best naive baseline on test pinball loss "
                    f"({_m:.3g} vs {_b:.3g})")
    st.info("Noon-safe model (only inputs known at 12:00). On the 2026 test year, " + "; ".join(_msg) + ".",
            icon=":material/info:")

tab_over, tab_cmp, tab_plan, tab_about = st.tabs(
    [":material/show_chart: Forecast", ":material/leaderboard: Model comparison",
     ":material/bolt: Planning assistant", ":material/info: About"])

# ---------------------------------------------------------------- tab 1: forecast
with tab_over:
    with st.container(horizontal=True):
        st.metric("Cheapest hour", f"{int(np.argmin(price_q[:, 1])):02d}:00", f"{price_q[:, 1].min():.1f} ct/kWh",
                  delta_color="off", border=True)
        st.metric("Most expensive hour", f"{int(np.argmax(price_q[:, 1])):02d}:00", f"{price_q[:, 1].max():.1f} ct/kWh",
                  delta_color="off", border=True)
        st.metric("Mean price", f"{price_q[:, 1].mean():.1f} ct/kWh", border=True)
        st.metric("Solar peak", f"{solar_q[:, 1].max():,.0f} MWh", f"at {int(np.argmax(solar_q[:, 1])):02d}:00",
                  delta_color="off", border=True)
        if actual is not None:
            a = actual["price"].to_numpy()
            st.metric("Price MAE this day", f"{np.abs(a - price_q[:, 1]).mean():.2f} ct/kWh",
                      f"{((a >= price_q[:, 0]) & (a <= price_q[:, 2])).mean():.0%} inside the band", delta_color="off", border=True)
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("Electricity price")
        st.altair_chart(band_chart(x, price_q, "ct/kWh", PRICE_COLOR, None if actual is None else actual["price"].to_numpy()),
                        width="stretch", alt="Price forecast with low-middle-high band for the next day")
    with c2, st.container(border=True):
        st.subheader("Solar generation")
        st.altair_chart(band_chart(x, solar_q, "MWh per hour", SOLAR_COLOR, None if actual is None else actual["solar"].to_numpy()),
                        width="stretch", alt="Solar generation forecast with low-middle-high band for the next day")
    st.caption("Band = 10 % / 90 % quantiles (about 80 % of actual values should fall inside); the dotted line is the real value "
               "for past days.")
    with st.expander("Hourly values"):
        tbl = pd.DataFrame({"hour": x.strftime("%H:%M"), "price low": price_q[:, 0], "price mid": price_q[:, 1],
                            "price high": price_q[:, 2], "solar low": solar_q[:, 0], "solar mid": solar_q[:, 1],
                            "solar high": solar_q[:, 2]})
        if actual is not None:
            tbl["price actual"], tbl["solar actual"] = actual["price"].to_numpy(), actual["solar"].to_numpy()
        st.dataframe(tbl.round(2), hide_index=True, alt="Hourly forecast values")

# ---------------------------------------------------------------- tab 2: model comparison
with tab_cmp:
    st.subheader("Test period (2026), scored once")
    test = results["test"]
    c1, c2 = st.columns(2)
    target = c1.segmented_control("Target", ["price", "solar"], default="price", key="cmp_target")
    metric = c2.segmented_control("Metric", ["pinball", "MAE", "RMSE", "coverage", "width"], default="pinball", key="cmp_metric")
    target, metric = target or "price", metric or "pinball"
    rows = [{"model": label(g), "value": test[g][target][metric]["mean"], "std": test[g][target][metric]["std"],
             "kind": "baseline" if g.startswith(("naive", "smard")) else "neural"} for g in test if target in test[g]]
    cmp_df = pd.DataFrame(rows)
    bars = alt.Chart(cmp_df).mark_bar().encode(
        y=alt.Y("model:N", sort="-x" if metric != "coverage" else None, title=None),
        x=alt.X("value:Q", title=f"{target} {metric}"), color=alt.Color("kind:N", legend=alt.Legend(title=None)),
        tooltip=["model", alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("std:Q", format=".3f", title="std over seeds")])
    layers = [bars]
    if metric == "coverage":
        layers.append(alt.Chart(pd.DataFrame({"v": [0.8]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="v:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=36 * len(rows) + 40), width="stretch",
                    alt=f"Bar chart of test {target} {metric} for every model")
    st.caption("Coverage target is 0.80 (dashed line). Neural models: mean over seeds; baselines: one deterministic run.")
    st.markdown("**Full table**")
    full = [{"model": label(g), "target": t, **{m: round(test[g][t][m]["mean"], 3) for m in ("MAE", "RMSE", "pinball", "coverage", "width")}}
            for g in test for t in test[g]]
    st.dataframe(pd.DataFrame(full), hide_index=True, alt="Test metrics of all models")
    rep = RESULTS / "REPORT.md"
    if rep.exists():
        with st.expander("Full report"):
            st.markdown(rep.read_text())

# ---------------------------------------------------------------- tab 3: planning assistant
with tab_plan:
    st.subheader("Plan the cheapest hours")
    if known_today is None:
        st.info("Today's auction prices are not available for this day, so only tomorrow's hours can be planned.",
                icon=":material/info:")
    first = 12 if known_today is not None else 24
    hour_fmt = lambda h: f"{'Today' if h < 24 else 'Tomorrow'} {h % 24:02d}:00"
    c1, c2, c3 = st.columns(3)
    task = c1.segmented_control("Task", ["Electric car", "Heat pump"], default="Electric car", key="task") or "Electric car"
    energy = c2.number_input("Energy needed (kWh)", 0.5, 200.0, 20.0, 0.5)
    power = c3.number_input("Device power (kW)", 0.5, 50.0, 11.0 if task == "Electric car" else 3.0, 0.5)
    c4, c5 = st.columns(2)
    start, end = c4.select_slider("Allowed window", options=list(range(first, 49)), value=(first, min(31, 48)),
                                  format_func=hour_fmt, help="Hours the device may run. End is exclusive. Today's "
                                  "hours before 12:00 are already past.")
    mode_p = c5.segmented_control("Mode", ["normal", "safe"], default="normal", key="plan_mode",
                                  help="normal ranks hours by the middle forecast, safe by the high value (less price risk)") or "normal"
    if start >= end:
        st.warning("The window end must be after the start.", icon=":material/warning:")
    elif st.button("Find cheapest hours", type="primary", icon=":material/search:"):
        if known_today is not None:
            k = known_today.to_numpy()
            table = np.vstack([np.stack([k, k, k], axis=1), price_q])      # known prices: no uncertainty
            t_idx = pd.date_range(origin, periods=36, freq="h")
        else:
            table, t_idx = price_q, pd.date_range(d1, periods=24, freq="h")
        plan = planner.make_plan(table, energy, power, start, end, mode_p, first_hour=first)
        chosen = pd.DataFrame({"time": [t_idx[h - first] for h in plan.hours], "price": [table[h - first, 1] for h in plan.hours]})
        known_df = None
        if known_today is not None:
            known_df = pd.DataFrame({"time": t_idx[:12], "known": known_today.to_numpy()})
        with st.container(border=True):
            st.altair_chart(band_chart(t_idx[12:] if known_today is not None else t_idx, price_q, "ct/kWh", PRICE_COLOR,
                                       known=known_df, chosen=chosen, height=320),
                            width="stretch", alt="Price (known today, forecast tomorrow) with the chosen hours marked")
        with st.container(horizontal=True):
            st.metric("Chosen hours", planner.contiguous_blocks(plan.hours), border=True)
            st.metric("Estimated cost", f"{plan.cost_eur:.2f} EUR", f"range {plan.cost_low_eur:.2f} - {plan.cost_high_eur:.2f} EUR",
                      delta_color="off", border=True)
            st.metric("Average price", f"{plan.avg_price_ct:.1f} ct/kWh", border=True)
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
            provider = "anthropic" if st.secrets.get("ANTHROPIC_API_KEY") else "nvidia"
        except Exception:
            key, provider = None, None
        text, source = planner.explain(plan, task.lower(), table, key, provider)
        st.info(text, title="Assistant", icon=":material/smart_toy:")
        st.caption("Explanation by LLM" if source == "llm" else "Simple text summary (no API key set)")
        st.caption("Cost uses the wholesale price only; households pay taxes and grid fees on top, and only dynamic "
                   "tariffs pass the hourly price on.")

# ---------------------------------------------------------------- tab 4: about
with tab_about:
    st.markdown("""
**What this is.** A day-ahead forecast of the German electricity price and solar generation with an uncertainty band,
and a small planning assistant (an optimizer picks the hours, an LLM only explains the plan).

**Protocol.** At 12:00 of day D the forecast for day D+1 is made from the last 7 days plus the known-future inputs of the target day
(calendar and the official SMARD day-ahead forecasts). Training 2019-2024, validation 2025, test 2026 (scored once).

**Limitations**
- Weather inputs are observations, not forecasts, so the solar gain would be smaller in real use.
- Prices are wholesale only (no taxes or grid fees). Negative prices and the 2022 energy crisis cause large errors.
- Prices are quarter-hourly since October 2025; they are averaged to hourly values here.

**Data.** Bundesnetzagentur | SMARD.de (CC BY 4.0); DWD open data; Open-Meteo (CC BY 4.0).
""")
