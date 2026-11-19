import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baselines as B  # noqa: E402
import evaluate as E  # noqa: E402
import models as M  # noqa: E402
import windows as W  # noqa: E402


@pytest.mark.parametrize("name", ["lstm", "cnn", "patchtst"])
@pytest.mark.parametrize("k", [0, 9])
def test_output_shape_with_and_without_future_inputs(name, k):
    m = M.build(name, 16, k).eval()
    x, f = torch.randn(3, 168, 16), (torch.randn(3, 24, k) if k else None)
    assert m(x, f).shape == (3, 24, 2, 3)


def test_cnn_receptive_field_covers_the_whole_week():
    assert M.receptive_field(4) >= 168
    m = M.build("cnn", 4).eval()
    x = torch.randn(1, 168, 4, requires_grad=True)
    h = m.conv(x.transpose(1, 2))[:, :, -1].sum()
    h.backward()
    assert x.grad[0, 0].abs().sum() > 0          # the last step really depends on the first hour of the week
    assert M.CNN_DILATIONS == (1, 2, 4, 8, 16, 32)


def test_pinball_loss_hand_computed_and_sorted():
    y = torch.zeros(1, 24, 2)
    pred = torch.zeros(1, 24, 2, 3)
    pred[..., 0], pred[..., 1], pred[..., 2] = -1.0, 0.0, 2.0
    # q=0.1: y-pred=1 -> 0.1; q=0.5: 0; q=0.9: y-pred=-2 -> (0.9-1)*-2=0.2; mean = 0.3/3
    assert M.pinball_loss(pred, y).item() == pytest.approx(0.1)
    crossed = pred[..., [2, 1, 0]]                 # same values, quantiles in the wrong order
    assert M.pinball_loss(crossed, y).item() == pytest.approx(0.1)


def test_metrics_hand_computed():
    y = np.zeros((1, 24, 2))
    y[..., 1] = 10.0
    q = np.zeros((1, 24, 2, 3))
    q[..., 0, 0], q[..., 0, 1], q[..., 0, 2] = -1.0, 1.0, 2.0     # price: y=0 inside, mid error 1
    q[..., 1, 0], q[..., 1, 1], q[..., 1, 2] = 0.0, 5.0, 8.0      # solar: y=10 above the band
    r = E.metrics(y, q)
    assert r["price"]["MAE"] == pytest.approx(1.0) and r["price"]["coverage"] == 1.0 and r["price"]["width"] == 3.0
    assert r["solar"]["MAE"] == pytest.approx(5.0) and r["solar"]["coverage"] == 0.0
    # solar pinball: q.1: 0.1*10, q.5: 0.5*5, q.9: 0.9*2 -> mean(1, 2.5, 1.8)
    assert r["solar"]["pinball"] == pytest.approx((1.0 + 2.5 + 1.8) / 3)


def test_naive_baselines_pick_the_right_hours():
    """Values = hours since the start, so each baseline must return target hour minus its lag."""
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=24 * 20, freq="h")
    hrs = np.arange(len(idx), dtype=float)
    df = pd.DataFrame({"price": hrs, "solar": hrs, "fc_solar": hrs + 0.5}, index=idx)
    starts = pd.DatetimeIndex([pd.Timestamp("2026-01-10")])
    t0 = 9 * 24
    assert np.allclose(B.point(df, starts, "naive_lastday")[0, :, 0], t0 - 48 + np.arange(24))   # D-1 = target - 48 h
    assert np.allclose(B.point(df, starts, "naive_lastweek")[0, :, 1], t0 - 168 + np.arange(24))
    p = B.point(df, starts, "naive_dayD")
    assert np.allclose(p[0, :, 0], t0 - 24 + np.arange(24)) and np.isnan(p[0, :, 1]).all()    # price only
    assert np.allclose(B.point(df, starts, "smard_fc_solar")[0, :, 1], t0 + 0.5 + np.arange(24))


def test_dm_test_detects_a_clear_difference_and_not_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.3, 300)
    assert E.dm_test(a - 0.2, a + rng.normal(0, 0.05, 300))["p"] < 1e-6      # A clearly better
    assert E.dm_test(a, a + rng.normal(0, 0.3, 300))["p"] > 0.01             # same expected loss
    r = E.dm_test(np.zeros(50), np.ones(50) + rng.normal(0, 0.1, 50))
    assert r["mean_diff"] < 0 and r["stat"] < 0


def test_postprocess_sorts_and_clips_solar():
    q = np.zeros((1, 24, 2, 3))
    q[0, :, 0] = [3.0, 1.0, 2.0]        # price, crossed
    q[0, :, 1] = [-5.0, 2.0, -1.0]      # solar, negative values
    out = E.postprocess(q)
    assert (out[0, :, 0] == [1.0, 2.0, 3.0]).all()
    assert (out[0, :, 1] == [0.0, 0.0, 2.0]).all()


def test_split_conformal_per_hour_reaches_target_coverage():
    import calibrate as C

    rng = np.random.default_rng(1)
    n = 2000
    y = rng.normal(0, 1, (n, 24, 2)) * np.linspace(0.1, 3, 24)[None, :, None]   # error size differs by hour
    q = np.zeros((n, 24, 2, 3))
    q[..., 0], q[..., 2] = -0.3, 0.3                                            # far too narrow
    d = C.conformal_deltas(y[: n // 2], q[: n // 2])
    yt, qt = y[n // 2:], q[n // 2:]
    lo, hi = qt[..., 0] - d, qt[..., 2] + d
    cov_by_hour = ((yt[..., 0] >= lo[..., 0]) & (yt[..., 0] <= hi[..., 0])).mean(axis=0)
    assert np.all(np.abs(cov_by_hour - 0.8) < 0.07)
