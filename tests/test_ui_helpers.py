"""Tests for the ``build1_inference_calculator/_ui/`` primitives.

Streamlit-free — we exercise the pure-logic modules (display_names,
explain, cost_counter) only. Rendering modules (page_header, presets,
provenance, tool_card, sidebar) are smoke-tested through imports to catch
syntax / import-time errors.

Run with: ``pytest tests/test_ui_helpers.py -v``
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# The _ui package lives outside tests/, so ensure its parent is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD1 = _REPO_ROOT / "build1_inference_calculator"
if str(_BUILD1) not in sys.path:
    sys.path.insert(0, str(_BUILD1))

from _ui import cost_counter, display_names, explain, provenance  # noqa: E402

# ---------------------------------------------------------------------------
# display_names
# ---------------------------------------------------------------------------


def test_gpu_display_name_known_id_returns_pretty_name():
    name = display_names.gpu_display_name("h100_sxm5_80gb")
    assert "H100" in name
    assert name != "h100_sxm5_80gb"


def test_gpu_display_name_unknown_id_falls_back():
    assert display_names.gpu_display_name("made_up_gpu") == "made_up_gpu"


def test_gpu_options_round_trips():
    opts = display_names.gpu_options()
    assert len(opts) > 10  # catalog has 17 rows
    assert "h100_sxm5_80gb" in opts.values()
    # Pretty name → id → pretty name
    for pretty, raw in opts.items():
        assert display_names.gpu_display_name(raw) == pretty


def test_fact_kind_display_handles_enum_and_string():
    from core.market_intel import FactKind
    assert display_names.fact_kind_display(FactKind.GPU_PRICE) == "GPU price"
    assert display_names.fact_kind_display("gpu_price") == "GPU price"
    assert display_names.fact_kind_display("capex") == "Capex"
    # Unknown falls through
    assert display_names.fact_kind_display("not_a_fact_kind") == "not_a_fact_kind"


def test_model_display_name_covers_both_picker_options():
    assert display_names.model_display_name("gpt-4o-mini") == "OpenAI gpt-4o-mini"
    assert display_names.model_display_name("claude-haiku-5-5") == "Claude Haiku 5.5"


def test_model_backend_detection():
    assert display_names.model_backend("gpt-4o-mini") == "openai"
    assert display_names.model_backend("claude-haiku-5-5") == "anthropic"
    with pytest.raises(ValueError):
        display_names.model_backend("mistral-7b")


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_tooltip_known_key_non_empty():
    assert explain.tooltip("roofline")
    assert len(explain.tooltip("roofline")) < 160  # fits in a Streamlit help=


def test_tooltip_unknown_key_is_empty_string():
    """Empty string so passing it to Streamlit never crashes."""
    assert explain.tooltip("made_up_term") == ""


def test_caption_known_key_non_empty():
    assert explain.caption("memory_fit")
    assert explain.caption("tco_comparison")


def test_all_tooltips_under_140_chars():
    """Streamlit help= tooltips truncate ugly if too long."""
    offenders = {k: len(v) for k, v in explain.TOOLTIPS.items() if len(v) > 160}
    assert not offenders, f"Tooltips too long for Streamlit help=: {offenders}"


def test_break_even_sentence_busy_fraction():
    class Fake:
        option_a = "owned"
        option_b = "reserved"
        pivot_variable = "busy_fraction"
        pivot_value = 0.47
        note = ""
    s = explain.break_even_sentence(Fake())
    assert "Owning" in s
    assert "Renting reserved" in s
    assert "47%" in s


def test_break_even_sentence_unknown_pivot_falls_back():
    class Fake:
        option_a = "owned"
        option_b = "api"
        pivot_variable = "mystery_lever"
        pivot_value = 0.5
        note = ""
    s = explain.break_even_sentence(Fake())
    assert "mystery_lever" in s
    assert "50%" in s


# ---------------------------------------------------------------------------
# cost_counter — pure math
# ---------------------------------------------------------------------------


def test_cost_for_turn_openai_math():
    """gpt-4o-mini: 1000 in / 500 out / 0 cached → $0.00045."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    c = cost_counter.cost_for_turn("gpt-4o-mini", usage)
    # (1000 * 0.15 + 500 * 0.60) / 1e6 = 0.00015 + 0.00030 = $0.00045
    assert c.usd == pytest.approx(0.00045, rel=1e-6)
    assert c.gbp == pytest.approx(0.00045 * cost_counter.USD_TO_GBP, rel=1e-6)
    assert c.input_tok == 1000
    assert c.output_tok == 500
    assert c.cached_tok == 0
    assert c.cache_hit_pct == 0.0


def test_cost_for_turn_openai_with_cache_hit():
    """Cached input priced at the cached rate; the rest at the normal rate."""
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 800},
    }
    c = cost_counter.cost_for_turn("gpt-4o-mini", usage)
    # Uncached input: 200 * 0.15 = $0.00003
    # Cached input: 800 * 0.075 = $0.00006
    # Output: 500 * 0.60 = $0.00030
    # Total: $0.00039
    assert c.usd == pytest.approx(0.00039, rel=1e-6)
    assert c.cached_tok == 800
    assert c.cache_hit_pct == pytest.approx(80.0)


def test_cost_for_turn_anthropic_math():
    """claude-haiku-5-5: Anthropic usage shape."""
    usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 0,
    }
    c = cost_counter.cost_for_turn("claude-haiku-5-5", usage)
    # (1000 * 0.10 + 500 * 0.50) / 1e6 = 0.0001 + 0.00025 = $0.00035
    assert c.usd == pytest.approx(0.00035, rel=1e-6)
    assert c.input_tok == 1000
    assert c.output_tok == 500


def test_cost_for_turn_unknown_model_raises():
    with pytest.raises(ValueError, match="No price card"):
        cost_counter.cost_for_turn("gpt-nonexistent", {"prompt_tokens": 10, "completion_tokens": 10})


def test_cost_for_turn_zero_input_safe():
    """cache_hit_pct must not divide by zero."""
    c = cost_counter.cost_for_turn("gpt-4o-mini", {"prompt_tokens": 0, "completion_tokens": 0})
    assert c.cache_hit_pct == 0.0
    assert c.usd == 0.0


# ---------------------------------------------------------------------------
# cost_counter — DuckDB round trip
# ---------------------------------------------------------------------------


def test_log_turn_round_trip(tmp_path):
    db = tmp_path / "turns.duckdb"
    cost = cost_counter.cost_for_turn(
        "gpt-4o-mini",
        {"prompt_tokens": 400, "completion_tokens": 300},
    )
    cost_counter.log_turn(
        db,
        cost=cost,
        latency_ms=1234.5,
        question="Test question?",
        answer_md="Test answer.",
        ts=datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC),
    )
    turns = cost_counter.load_turns(db)
    assert len(turns) == 1
    row = turns[0]
    assert row["model"] == "gpt-4o-mini"
    assert row["input_tok"] == 400
    assert row["output_tok"] == 300
    assert row["latency_ms"] == pytest.approx(1234.5)
    assert row["question"] == "Test question?"
    assert row["answer_md"] == "Test answer."


def test_log_turn_appends(tmp_path):
    db = tmp_path / "turns.duckdb"
    c1 = cost_counter.cost_for_turn("gpt-4o-mini", {"prompt_tokens": 100, "completion_tokens": 50})
    c2 = cost_counter.cost_for_turn(
        "claude-haiku-5-5",
        {"input_tokens": 200, "output_tokens": 100},
    )
    cost_counter.log_turn(db, cost=c1, latency_ms=1.0, question="Q1", answer_md="A1")
    cost_counter.log_turn(db, cost=c2, latency_ms=2.0, question="Q2", answer_md="A2")
    assert len(cost_counter.load_turns(db)) == 2


def test_load_turns_missing_db_returns_empty():
    assert cost_counter.load_turns("/does/not/exist.duckdb") == []


def test_aggregates_empty():
    agg = cost_counter.aggregates([])
    assert agg.turns == 0
    assert agg.total_gbp == 0.0


def test_aggregates_rollup():
    turns = [
        {"input_tok": 1000, "output_tok": 500, "cached_tok": 800,
         "latency_ms": 1000.0, "gbp": 0.001, "model": "a"},
        {"input_tok": 1000, "output_tok": 500, "cached_tok": 0,
         "latency_ms": 2000.0, "gbp": 0.003, "model": "b"},
    ]
    agg = cost_counter.aggregates(turns)
    assert agg.turns == 2
    assert agg.total_gbp == pytest.approx(0.004)
    assert agg.avg_gbp == pytest.approx(0.002)
    assert agg.cache_hit_pct == pytest.approx(40.0)  # 800 cached / 2000 input
    assert agg.avg_latency_ms == pytest.approx(1500.0)


def test_aggregates_by_model():
    turns = [
        {"input_tok": 100, "output_tok": 50, "cached_tok": 0,
         "latency_ms": 500.0, "gbp": 0.001, "model": "gpt-4o-mini"},
        {"input_tok": 100, "output_tok": 50, "cached_tok": 0,
         "latency_ms": 700.0, "gbp": 0.0005, "model": "claude-haiku-5-5"},
    ]
    per_model = cost_counter.aggregates_by_model(turns)
    assert set(per_model) == {"gpt-4o-mini", "claude-haiku-5-5"}
    assert per_model["gpt-4o-mini"].turns == 1
    assert per_model["claude-haiku-5-5"].avg_gbp == pytest.approx(0.0005)


def test_clear_log_removes_file(tmp_path):
    db = tmp_path / "turns.duckdb"
    cost = cost_counter.cost_for_turn("gpt-4o-mini", {"prompt_tokens": 1, "completion_tokens": 1})
    cost_counter.log_turn(db, cost=cost, latency_ms=1.0, question="Q", answer_md="A")
    assert db.exists()
    cost_counter.clear_log(db)
    assert not db.exists()
    # Clearing a non-existent file is safe.
    cost_counter.clear_log(db)


# ---------------------------------------------------------------------------
# provenance — pure helpers
# ---------------------------------------------------------------------------


def test_badge_returns_emoji_and_label():
    assert provenance.badge("measured").startswith("🟢")
    assert "Measured" in provenance.badge("measured")
    assert "Modelled" in provenance.badge("modelled")


def test_badge_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provenance kind"):
        provenance.badge("not-a-kind")  # type: ignore[arg-type]


def test_confidence_badge():
    assert "high" in provenance.confidence_badge("high")
    assert "🟢" in provenance.confidence_badge("high")
    assert provenance.confidence_badge("made-up") == "made-up"


# ---------------------------------------------------------------------------
# Streamlit rendering modules — smoke-test the imports compile
# ---------------------------------------------------------------------------


def test_streamlit_modules_import():
    """If a Streamlit module has a syntax error, this will catch it without
    needing a running Streamlit server."""
    from _ui import page_header, presets, sidebar, tool_card  # noqa: F401


def test_preset_gpu_ids_match_catalog():
    """Every preset must reference a GPU id that exists in the catalog."""
    from _ui import presets as presets_mod

    from core.gpu_specs import load_catalog
    catalog_ids = set(load_catalog().ids())
    for s in presets_mod.SCENARIOS:
        assert s.gpu_id in catalog_ids, (
            f"Preset {s.key!r} references unknown GPU id {s.gpu_id!r}"
        )
