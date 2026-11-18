import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import planner as P  # noqa: E402


def prices():
    mid = np.full(24, 10.0)
    mid[[2, 3, 4]] = [4.0, 2.0, 6.0]
    q = np.stack([mid - 1, mid, mid + 1], axis=1)
    return q


def test_picks_cheapest_hours_and_cost():
    plan = P.make_plan(prices(), energy_kwh=20, power_kw=11, start=0, end=7)
    assert plan.hours_needed == 2
    assert set(plan.hours) == {2, 3}                      # prices 4 and 2 are cheapest
    # 11 kWh at 2 ct + 9 kWh at 4 ct (partial hour goes to the most expensive chosen hour)
    assert plan.cost_eur == pytest.approx((11 * 2 + 9 * 4) / 100)
    assert plan.cost_low_eur < plan.cost_eur < plan.cost_high_eur


def test_window_wraps_into_next_day():
    assert P.allowed_hours(22, 2) == [22, 23, 24, 25]
    assert P.allowed_hours(18, 7) == list(range(18, 31))
    assert P.allowed_hours(5, 5) == list(range(5, 29))
    plan = P.make_plan(prices(), 5, 11, start=10, end=12)
    assert all(10 <= h < 12 for h in plan.hours)


def test_overnight_window_uses_evening_of_D_and_morning_of_next_day():
    """D 12:00 .. D+1 23:00 table (36 rows). 'from 18 until 7' must never pick D 03:00 (an hour in the past)."""
    mid = np.full(36, 10.0)
    mid[(24 + 3) - 12] = 1.0     # D+1 03:00 is the cheapest hour overall
    q = np.stack([mid - 1, mid, mid + 1], axis=1)
    plan = P.make_plan(q, 5, 11, start=18, end=7, first_hour=12)
    assert plan.hours == [27]                              # D+1 03:00 = absolute hour 27
    assert P.contiguous_blocks(plan.hours) == "03:00-04:00 (tomorrow)"


def test_window_outside_the_price_table_is_cut():
    plan = P.make_plan(prices(), 60, 11, start=20, end=30)     # only hours 20..23 exist in a 24 h table
    assert plan.hours == [20, 21, 22, 23] and not plan.feasible


def test_infeasible_window_reports_note():
    plan = P.make_plan(prices(), 50, 11, start=0, end=2)
    assert not plan.feasible and len(plan.hours) == 2 and plan.notes


def test_negative_prices_chosen_and_cost_negative():
    q = prices()
    q[10] = [-6, -5, -4]
    plan = P.make_plan(q, 11, 11, start=8, end=14)
    assert plan.hours == [10] and plan.cost_eur == pytest.approx(-0.55)


def test_safe_mode_ranks_by_high_value():
    q = prices()
    q[3] = [0, 2, 30]  # cheap on average but risky
    normal = P.make_plan(q, 11, 11, 0, 7, "normal")
    safe = P.make_plan(q, 11, 11, 0, 7, "safe")
    assert normal.hours == [3] and safe.hours != [3]


def test_template_text_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = P.make_plan(prices(), 20, 11, 0, 7)
    text, src = P.explain(plan, "electric car", prices())
    assert src == "template" and "EUR" in text


def test_llm_failure_falls_back_to_template(monkeypatch):
    monkeypatch.setattr(P, "_ask_anthropic", lambda key, facts: (_ for _ in ()).throw(RuntimeError("boom")))
    plan = P.make_plan(prices(), 20, 11, 0, 7)
    text, src = P.explain(plan, "electric car", prices(), api_key="k", provider="anthropic")
    assert src == "template" and "LLM unavailable" in text
