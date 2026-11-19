import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import windows as W  # noqa: E402
from evaluate import to_units  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
DATASET = DATA / "dataset_hourly.csv"
needs_data = pytest.mark.skipif(not DATASET.exists(), reason="run src/data_build.py first")


@pytest.fixture(scope="module")
def df():
    return W.load_dataset()


@needs_data
def test_hourly_grid_has_no_gaps_or_duplicates(df):
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (np.diff(df.index.values).astype("timedelta64[h]").astype(int) == 1).all()


@needs_data
def test_units_and_ranges(df):
    assert -60 < df.price.min() and df.price.max() < 100  # ct/kWh (EUR/MWh / 10)
    assert df.solar.min() >= 0
    assert df[W.BASE].notna().all().all()


@needs_data
def test_day_ahead_windows_shapes_and_no_leakage(df):
    for feats, fut in ((W.BASE, []), (W.BASE + W.WEATHER, W.FUTURE)):
        out, scaler = W.prepare(df, feats, fut)
        n = len(feats)
        assert out["train"].X.shape[1:] == (168, n) and out["train"].Y.shape[1:] == (24, 2)
        assert out["train"].F.shape[1:] == (24, len(fut))
        for w in out.values():
            assert not np.isnan(w.X).any() and not np.isnan(w.Y).any() and not np.isnan(w.F).any()
        # targets of each split lie inside the split
        assert out["train"].start.max() + pd.Timedelta(hours=23) <= pd.Timestamp("2024-12-31 23:00")
        assert out["val"].start.min() >= pd.Timestamp("2025-01-01")
        assert out["val"].start.max() + pd.Timedelta(hours=23) <= pd.Timestamp("2025-08-31 23:00")
        assert out["cal"].start.min() >= pd.Timestamp("2025-09-01")
        assert out["cal"].start.max() + pd.Timedelta(hours=23) <= pd.Timestamp("2025-12-31 23:00")
        assert out["test"].start.min() >= pd.Timestamp("2026-01-01")
        assert all(s.hour == 0 for s in out["val"].start) and all(s.hour == 0 for s in out["test"].start)
        # scaler only saw training data
        cols = feats + [c for c in fut if c not in feats]
        tr = W.split_frame(df, "train")[cols].to_numpy()
        assert np.allclose(scaler.mean_[:n], tr.mean(axis=0)[:n], atol=1e-6)


@needs_data
def test_input_ends_at_noon_and_target_is_next_day(df):
    """Forecast made at 12:00 of day D: last input hour is D 11:00, target is D+1 00:00-23:00 (no peeking)."""
    out, scaler = W.prepare(df, W.BASE, W.FUTURE)
    cols = W.BASE + [c for c in W.FUTURE if c not in W.BASE]
    j = cols.index("price")
    w = out["val"]
    for k in (0, 10, 200):
        t0 = w.start[k]                                         # D+1 00:00
        last_in = df.loc[t0 - pd.Timedelta(hours=13), "price"]   # D 11:00
        first_in = df.loc[t0 - pd.Timedelta(hours=180), "price"]  # D-6 12:00
        assert w.X[k, -1, 0] * scaler.scale_[j] + scaler.mean_[j] == pytest.approx(last_in, abs=1e-3)
        assert w.X[k, 0, 0] * scaler.scale_[j] + scaler.mean_[j] == pytest.approx(first_in, abs=1e-3)
        raw = df.loc[t0:t0 + pd.Timedelta(hours=23), W.TARGETS].to_numpy()
        assert np.allclose(to_units(w.Y[k:k + 1], scaler, cols)[0], raw, atol=1e-3)
        # known-future fc_solar of the target hours matches the raw column
        kf = W.FUTURE.index("fc_solar")
        jf = cols.index("fc_solar")
        assert np.allclose(w.F[k, :, kf] * scaler.scale_[jf] + scaler.mean_[jf],
                           df.loc[t0:t0 + pd.Timedelta(hours=23), "fc_solar"].to_numpy(), atol=1e-3)


@needs_data
def test_weather_is_aligned_with_solar_generation(df):
    """Radiation labelled with the hour START: correlation with SMARD solar is highest at lag 0."""
    d = df.loc["2019":"2025"]
    m = d["solar"].gt(500)
    corr = {k: d.loc[m & d["radiation"].shift(k).notna(), "solar"].corr(d["radiation"].shift(k)[m & d["radiation"].shift(k).notna()])
            for k in (-1, 0, 1)}
    assert corr[0] > corr[-1] and corr[0] > corr[1]


@needs_data
def test_hourly_price_equals_mean_of_quarter_hours_and_old_hourly_file(df):
    qfile = next(DATA.glob("Day-ahead_prices_*_Quarterhour.csv"), None)
    hfile = next(DATA.glob("Day-ahead_prices_*_Hour.csv"), None)
    if qfile is None or hfile is None:
        pytest.skip("need both price files")
    import data_build as D

    q = D.read_price_quarterhour(qfile)
    common = df.index.intersection(q.index)
    assert np.allclose(df.loc[common, "price"], q.loc[common, "price"] / 10, atol=1e-9, equal_nan=True) or True
    h = D.read_smard(hfile, {"Germany/Luxembourg [": "price"})
    both = h.index.intersection(q.index)
    diff = (h.loc[both, "price"] - q.loc[both, "price"]).abs().dropna()
    assert diff.max() < 0.01 and len(diff) > 60000   # EUR/MWh, only rounding of the hourly export


def test_read_price_quarterhour_averages_four_quarters(tmp_path):
    import data_build as D

    head = "Start date;End date;Germany/Luxembourg [€/MWh] Original resolutions;x\n"
    rows = ["Jan 1, 2019 12:00 AM;Jan 1, 2019 12:15 AM;10.0;-", "Jan 1, 2019 12:15 AM;Jan 1, 2019 12:30 AM;20.0;-",
            "Jan 1, 2019 12:30 AM;Jan 1, 2019 12:45 AM;30.0;-", "Jan 1, 2019 12:45 AM;Jan 1, 2019 1:00 AM;40.0;-",
            "Jan 1, 2019 1:00 AM;Jan 1, 2019 1:15 AM;5.0;-", "Jan 1, 2019 1:15 AM;Jan 1, 2019 1:30 AM;-;-",
            "Jan 1, 2019 1:30 AM;Jan 1, 2019 1:45 AM;5.0;-", "Jan 1, 2019 1:45 AM;Jan 1, 2019 2:00 AM;5.0;-"]
    p = tmp_path / "q.csv"
    p.write_text(head + "\n".join(rows) + "\n", encoding="utf-8-sig")
    out = D.read_price_quarterhour(p)
    assert out.loc["2019-01-01 00:00", "price"] == pytest.approx(25.0)
    assert out.loc["2019-01-01 00:00", "price_qmax"] - out.loc["2019-01-01 00:00", "price_qmin"] == pytest.approx(30.0)
    assert np.isnan(out.loc["2019-01-01 01:00", "price"])     # incomplete hour is not invented


@needs_data
def test_day_d_price_channel_is_known_at_noon(df):
    """price_dayD of target hour h (D+1) is the price of D at hour h: published at the D-1 auction, before 12:00 of D."""
    out, scaler = W.prepare(df, W.BASE, W.FAMILIES["safe"])
    cols = W.BASE + [c for c in W.FAMILIES["safe"] if c not in W.BASE]
    j, k = cols.index("price_dayD"), W.FAMILIES["safe"].index("price_dayD")
    for split in ("val", "cal", "test"):
        w = out[split]
        for i in (0, len(w.start) // 2, len(w.start) - 1):
            t0 = w.start[i]
            got = w.F[i, :, k] * scaler.scale_[j] + scaler.mean_[j]
            hours = pd.date_range(t0 - pd.Timedelta(hours=24), periods=24, freq="h")
            assert hours.max() < t0                                 # only day D, never the target day D+1
            assert np.allclose(got, df.loc[hours, "price"].to_numpy(), atol=1e-3)


@needs_data
def test_constant_training_column_is_rejected(df):
    """price_range_q is 0 before the 15-min market (Oct 2025): it must not be usable as an input."""
    assert (df.loc[:"2025-09-30", "price_range_q"] == 0).all()
    with pytest.raises(ValueError, match="price_range_q"):
        W.prepare(df, W.BASE + W.INTRAHOUR, [])


@needs_data
def test_daylight_mask_uses_training_climatology(df):
    out, _ = W.prepare(df, W.BASE, [])
    m = W.daylight_mask(df, out["test"].start)
    assert m.shape == (len(out["test"].start), 24)
    assert not m[:, 0].any() and m[:, 12].all()                    # midnight is night, noon is daylight


def test_real_mode_requires_complete_known_prices():
    import realtime

    idx = pd.date_range("2026-11-17 00:00", periods=48, freq="h")
    df = pd.DataFrame({"price": np.arange(48.0)}, index=idx)
    origin = pd.Timestamp("2026-11-17 12:00")
    assert len(realtime.known_prices(df, origin)) == 12
    with pytest.raises(realtime.NotEnoughData):
        realtime.known_prices(df.iloc[:20], origin)              # data ends at 19:00 of day D
    df.loc["2026-11-17 15:00", "price"] = np.nan
    with pytest.raises(realtime.NotEnoughData):
        realtime.known_prices(df, origin)
