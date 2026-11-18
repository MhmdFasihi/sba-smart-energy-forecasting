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
    """X rows hold their own row index; row 167 = D 11:00, target hour h of D+1 = row 180+h."""
    X = np.zeros((1, 168, len(W.BASE)), dtype=np.float32)
    X[0, :, :] = np.arange(168)[:, None]
    last_day = B.naive(X, "lastday")[0, :, 0]
    last_week = B.naive(X, "lastweek")[0, :, 0]
    assert np.allclose(last_day, 132 + np.arange(24))   # D-1 same hour = target - 48 h
    assert np.allclose(last_week, 12 + np.arange(24))   # target - 168 h


def test_band_is_per_target_hour():
    rng = np.random.default_rng(0)
    resid = rng.normal(size=(500, 24, 2)) * np.linspace(1, 5, 24)[None, :, None]
    lo, hi = B.band(resid)
    assert lo.shape == (24, 2) and (hi[-1] - lo[-1] > hi[0] - lo[0]).all()   # wider band for noisier hours
