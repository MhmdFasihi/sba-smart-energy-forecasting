"""Planning assistant: a small optimizer picks the hours, an LLM (optional) only explains the result.

The LLM never calculates; it receives the finished plan as facts. Without an API key a template
text is used. Costs are wholesale price only (no taxes / grid fees).

Hours are ABSOLUTE hours counted from 00:00 of day D, the day the forecast is made (12:00 = cut-off):
0..23 is day D, 24..47 is day D+1 (the forecast day). The price table passed to make_plan starts at
`first_hour` (12 in the app: known auction prices of D 12-23, then the D+1 forecast), so a window such as
"18:00 until 07:00" correctly means D 18:00 .. D+1 07:00.
"""
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"  # override with env NVIDIA_MODEL
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5"                       # override with env ANTHROPIC_MODEL
SYSTEM_PROMPT = ("You explain an electricity-use plan in plain language in 3-4 sentences, including how "
                 "certain the forecast is. Use ONLY the numbers given; do not calculate or invent numbers. "
                 "Mention that the cost is wholesale price only.")


@dataclass
class Plan:
    hours: list[int]                 # chosen absolute hours (0-23 = day D, 24-47 = day D+1), ascending
    energy_kwh: dict[int, float]     # kWh charged in each chosen hour
    cost_eur: float                  # expected cost using the middle (p50) forecast
    cost_low_eur: float              # same hours, low (p10) prices
    cost_high_eur: float             # same hours, high (p90) prices
    mode: str
    hours_needed: int
    feasible: bool
    avg_price_ct: float
    first_hour: int = 0              # absolute hour of row 0 of the price table
    notes: list[str] = field(default_factory=list)


def allowed_hours(start: int, end: int) -> list[int]:
    """Absolute hours [start, end). start >= end wraps into the next day: (18, 7) -> 18..30 (D 18:00 .. D+1 07:00);
    start == end means a full 24 h window."""
    if start == end:
        return list(range(start, start + 24))
    return list(range(start, end if start < end else end + 24))


def make_plan(price_q: np.ndarray, energy_kwh: float, power_kw: float, start: int = 0, end: int = 24,
              mode: str = "normal", first_hour: int = 0) -> Plan:
    """price_q: (T, 3) low/mid/high price in ct/kWh for the absolute hours first_hour .. first_hour+T-1.
    normal mode ranks hours by the middle value, safe mode by the high value (avoids price risk)."""
    if energy_kwh <= 0 or power_kw <= 0:
        raise ValueError("energy and power must be positive")
    price_q = np.sort(np.asarray(price_q, dtype=float), axis=-1)
    rank_col = 1 if mode == "normal" else 2
    pool = [h for h in allowed_hours(start, end) if first_hour <= h < first_hour + len(price_q)]
    P = lambda h, c: price_q[h - first_hour, c]
    n_need = math.ceil(energy_kwh / power_kw - 1e-9)
    notes = []
    feasible = n_need <= len(pool)
    n = min(n_need, len(pool))
    if not feasible:
        notes.append(f"The allowed window has only {len(pool)} h but {n_need} h are needed; "
                     f"only {n * power_kw:.1f} of {energy_kwh:.1f} kWh can be delivered.")
    chosen = sorted(sorted(pool, key=lambda h: (P(h, rank_col), h))[:n])
    # full power in every chosen hour, the remainder (partial hour) goes to the most expensive chosen hour
    energy = {h: power_kw for h in chosen}
    if chosen and n_need == n:
        rem = energy_kwh - (n - 1) * power_kw
        if rem < power_kw - 1e-9:
            worst = max(chosen, key=lambda h: P(h, 1))
            energy[worst] = rem
    cost = {c: sum(e * P(h, c) for h, e in energy.items()) / 100 for c in range(3)}
    total_kwh = sum(energy.values())
    avg = (cost[1] * 100 / total_kwh) if total_kwh else float("nan")
    if any(P(h, 1) < 0 for h in chosen):
        notes.append("Some chosen hours have negative prices (you are paid to consume).")
    return Plan(chosen, energy, cost[1], cost[0], cost[2], mode, n_need, feasible, avg, first_hour, notes)


def hour_label(h: int) -> str:
    return f"{h % 24:02d}:00" + (" (tomorrow)" if h >= 24 else "")


def contiguous_blocks(hours: list[int]) -> str:
    if not hours:
        return "none"
    blocks, s, p = [], hours[0], hours[0]
    for h in hours[1:] + [None]:
        if h is not None and h == p + 1:
            p = h
            continue
        blocks.append(f"{s % 24:02d}:00-{(p + 1) % 24:02d}:00" + (" (tomorrow)" if s >= 24 else ""))
        if h is not None:
            s = p = h
    return ", ".join(blocks)


def template_summary(plan: Plan, task: str, price_q: np.ndarray) -> str:
    q = np.sort(np.asarray(price_q, dtype=float), axis=-1)
    width = np.mean([q[h - plan.first_hour, 2] - q[h - plan.first_hour, 0] for h in plan.hours]) if plan.hours else 0.0
    txt = (f"Charge/run the {task} during {contiguous_blocks(plan.hours)} ({len(plan.hours)} h"
           f"{', safe mode' if plan.mode == 'safe' else ''}). Expected cost {plan.cost_eur:.2f} EUR at an "
           f"average {plan.avg_price_ct:.1f} ct/kWh (range {plan.cost_low_eur:.2f}-{plan.cost_high_eur:.2f} EUR). "
           f"The forecast band in these hours is about {width:.1f} ct/kWh wide, so the plan carries "
           f"{'moderate' if width < 6 else 'high'} uncertainty. Wholesale price only, without taxes and grid fees.")
    return " ".join([txt] + plan.notes)


def facts_text(plan: Plan, task: str, price_q: np.ndarray) -> str:
    q = np.sort(np.asarray(price_q, dtype=float), axis=-1)
    return (f"Task: {task}. Mode: {plan.mode}. Chosen hours: {contiguous_blocks(plan.hours)}. "
            f"Hours needed: {plan.hours_needed}. Expected cost: {plan.cost_eur:.2f} EUR "
            f"(plausible range {plan.cost_low_eur:.2f}-{plan.cost_high_eur:.2f} EUR). "
            f"Average price {plan.avg_price_ct:.1f} ct/kWh. Hourly low/mid/high ct/kWh for chosen hours: "
            + "; ".join(f"{hour_label(h)} {q[h - plan.first_hour, 0]:.1f}/{q[h - plan.first_hour, 1]:.1f}/"
                        f"{q[h - plan.first_hour, 2]:.1f}" for h in plan.hours)
            + f". Notes: {' '.join(plan.notes) or 'none'}")


def _ask_anthropic(key: str, facts: str) -> str:
    import anthropic

    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL), max_tokens=400, temperature=0.3,
        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": facts}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _ask_nvidia(key: str, facts: str) -> str:
    import requests

    # NVIDIA API Catalog (build.nvidia.com): OpenAI-compatible chat completions endpoint
    r = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions", timeout=60,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        json={"model": os.environ.get("NVIDIA_MODEL", NVIDIA_DEFAULT_MODEL), "max_tokens": 1024,  # room if the model reasons first
              "temperature": 0.3, "top_p": 0.95,  # low temperature: stick to the given numbers
              "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": facts}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def explain(plan: Plan, task: str, price_q: np.ndarray, api_key: str | None = None,
            provider: str | None = None) -> tuple[str, str]:
    """Return (text, source). Source is 'llm' or 'template'. Provider: 'anthropic' or 'nvidia'; default =
    Anthropic if ANTHROPIC_API_KEY is set, else NVIDIA if NVIDIA_API_KEY is set. api_key (e.g. from
    st.secrets) overrides the environment for the chosen provider."""
    fallback = template_summary(plan, task, price_q)
    if provider is None:
        provider = "anthropic" if (os.environ.get("ANTHROPIC_API_KEY") or (api_key and api_key.startswith("sk-ant"))) else "nvidia"
    key = api_key or os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic" else "NVIDIA_API_KEY")
    if not key:
        return fallback, "template"
    try:
        facts = facts_text(plan, task, price_q)
        text = _ask_anthropic(key, facts) if provider == "anthropic" else _ask_nvidia(key, facts)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # drop reasoning if inlined
        if not text:
            raise ValueError("empty answer")
        return text, "llm"
    except Exception as e:  # network / key problems must never break the app
        return fallback + f" (LLM unavailable: {type(e).__name__})", "template"
