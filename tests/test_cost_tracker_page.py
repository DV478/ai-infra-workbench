"""Smoke tests for the Cost tracker page (M7).

Exercises:
- Streamlit AppTest end-to-end render of the empty-state path.
- Streamlit AppTest end-to-end render after seeding a turn log via log_turn.

The pure-logic helpers (``cost_for_turn``, ``aggregates``, ``aggregates_by_model``,
``log_turn``/``load_turns`` round trip) already have unit tests in
``test_ui_helpers.py`` — we don't duplicate them here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD1 = _REPO_ROOT / "build1_inference_calculator"
if str(_BUILD1) not in sys.path:
    sys.path.insert(0, str(_BUILD1))

import pytest  # noqa: E402
from _ui import cost_counter  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = _BUILD1 / "pages" / "2_Cost_tracker.py"


def test_cost_tracker_empty_state_renders(monkeypatch, tmp_path):
    """No DB → page shows the empty-state info banner, not a crash."""
    monkeypatch.setattr(
        cost_counter,
        "default_db_path",
        lambda: tmp_path / "does_not_exist.duckdb",
    )
    at = AppTest.from_file(str(PAGE), default_timeout=10)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # Expect at least one st.info call (the empty-state banner).
    assert any("No chat turns logged yet" in str(m.value) for m in at.info), (
        "Empty-state info banner should mention that no turns are logged."
    )


def test_cost_tracker_with_turns_renders(monkeypatch, tmp_path):
    """DB populated → hero metrics + table render without error."""
    db = tmp_path / "turns.duckdb"
    monkeypatch.setattr(cost_counter, "default_db_path", lambda: db)

    # Seed two turns across two models so the per-model comparison fires.
    c1 = cost_counter.cost_for_turn(
        "gpt-4o-mini",
        {"prompt_tokens": 400, "completion_tokens": 300,
         "prompt_tokens_details": {"cached_tokens": 100}},
    )
    c2 = cost_counter.cost_for_turn(
        "claude-haiku-5-5",
        {"input_tokens": 500, "output_tokens": 350, "cache_read_input_tokens": 200},
    )
    cost_counter.log_turn(db, cost=c1, latency_ms=1200.0,
                          question="Q1?", answer_md="A1")
    cost_counter.log_turn(db, cost=c2, latency_ms=1800.0,
                          question="Q2?", answer_md="A2")

    at = AppTest.from_file(str(PAGE), default_timeout=15)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    # Hero metrics expected: Turns, Total cost, Avg per turn, Cache hit, Avg latency.
    metric_labels = [m.label for m in at.metric] if hasattr(at, "metric") else []
    assert "Turns logged" in metric_labels
    assert "Total cost" in metric_labels


def test_price_card_includes_both_models():
    """The price card must list every model the picker exposes."""
    from _ui.display_names import MODEL_DISPLAY_NAMES
    assert set(cost_counter.PRICE_PER_MTOK) == set(MODEL_DISPLAY_NAMES), (
        "PRICE_PER_MTOK and MODEL_DISPLAY_NAMES must stay in sync."
    )


def test_pricing_card_last_confirmed_is_set():
    assert cost_counter.PRICE_CARD_LAST_CONFIRMED
    assert len(cost_counter.PRICE_CARD_LAST_CONFIRMED) == 10  # YYYY-MM-DD


@pytest.mark.parametrize("model_id", ["gpt-4o-mini", "claude-haiku-5-5"])
def test_every_model_has_three_rate_fields(model_id):
    prices = cost_counter.PRICE_PER_MTOK[model_id]
    assert "input" in prices
    assert "cached_input" in prices
    assert "output" in prices
    # Cached-input should always be cheaper than full input.
    assert prices["cached_input"] < prices["input"]
    # Output should be more expensive than input (industry pattern).
    assert prices["output"] > prices["input"]
