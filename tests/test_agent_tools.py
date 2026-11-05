"""Tests for the compute advisor's tool registry.

These tests exercise the tool wrappers directly — no Anthropic API calls.
The agent-loop logic is tested separately in test_agent_loop.py using a
stubbed executor.
"""

from __future__ import annotations

import pytest

from core.agents.compute_advisor import (
    MODEL_REGISTRY,
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    run_tool,
)

# ---------- Registry shape --------------------------------------------------


def test_every_schema_has_implementation():
    schema_names = {s["name"] for s in TOOL_SCHEMAS}
    registry_names = set(TOOL_REGISTRY.keys())
    assert schema_names == registry_names, (
        f"schemas-only: {schema_names - registry_names}, "
        f"registry-only: {registry_names - schema_names}"
    )


def test_every_schema_has_description():
    for s in TOOL_SCHEMAS:
        assert s.get("description"), f"{s['name']} has no description"
        assert s["input_schema"]["type"] == "object"


def test_model_registry_matches_schema_enum():
    schema_enum = next(s for s in TOOL_SCHEMAS if s["name"] == "decode_tps")["input_schema"][
        "properties"
    ]["model"]["enum"]
    assert set(schema_enum) == set(MODEL_REGISTRY.keys())


# ---------- Individual tools -----------------------------------------------


def test_decode_tps_roundtrip():
    r = run_tool("decode_tps", {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "batch_size": 1})
    assert 50 < r["tokens_per_second"] < 250
    assert r["bound"] == "memory"


def test_fits_in_memory_70b_fp16_h100_no():
    r = run_tool("fits_in_memory", {
        "gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "dtype": "fp16",
        "batch_size": 1, "seq_len": 2048,
    })
    assert r["fits"] is False
    assert r["headroom_gb"] < 0


def test_fits_in_memory_70b_int4_h100_yes():
    r = run_tool("fits_in_memory", {
        "gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "dtype": "int4",
        "batch_size": 1, "seq_len": 2048,
    })
    assert r["fits"] is True


def test_cost_per_m_tokens_returns_both_measures():
    r = run_tool("cost_per_m_tokens", {
        "gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "hourly_usd": 3.0,
    })
    assert "cost_per_m_tokens_peak" in r
    assert "cost_per_m_tokens_realistic" in r
    assert r["cost_per_m_tokens_realistic"] > r["cost_per_m_tokens_peak"]


def test_compare_options_structure():
    r = run_tool("compare_options", {
        "gpu_id": "h100_sxm5_80gb",
        "model": "llama-3-8b",
        "reserved_hourly": 2.5,
        "on_demand_hourly": 4.5,
        "neocloud_hourly": 2.0,
        "api_output_price_per_m": 2.0,
    })
    assert len(r["options"]) == 5
    assert len(r["stress"]) == 5
    assert len(r["break_evens"]) == 2
    options = {o["option"] for o in r["options"]}
    assert options == {"owned", "reserved", "on_demand", "neocloud", "api"}


def test_api_margin_positive_for_high_rate():
    r = run_tool("api_margin", {
        "gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "hourly_usd": 3.0,
        "api_name": "Test", "api_price_per_m_tokens": 10.0,
    })
    assert r["gross_margin_pct"] > 50


def test_output_tokens_per_hour_returns_day_too():
    r = run_tool("output_tokens_per_hour", {
        "gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "batch_size": 32,
    })
    assert r["tokens_per_day"] == r["tokens_per_hour"] * 24


def test_owned_hourly_32k_in_expected_range():
    r = run_tool("owned_hourly_cost", {"capex_usd": 32000.0})
    assert 2.8 < r["hourly_usd"] < 3.6


def test_break_even_hourly_matches_formula():
    r = run_tool("break_even_hourly", {
        "cost_per_m_tokens_target": 2.0, "output_tokens_per_hr": 1_000_000.0,
    })
    assert r["hourly_usd"] == pytest.approx(2.0, rel=1e-6)


def test_api_only_cost_scales_linearly():
    a = run_tool("api_only_cost", {"api_output_price_per_m": 2.0, "tokens_demanded_3y": 1e9})
    b = run_tool("api_only_cost", {"api_output_price_per_m": 2.0, "tokens_demanded_3y": 2e9})
    assert b["total_usd_3y"] == pytest.approx(2 * a["total_usd_3y"])


# ---------- Error handling -------------------------------------------------


def test_unknown_tool_raises():
    with pytest.raises(ValueError, match="Unknown tool"):
        run_tool("nope", {})


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        run_tool("decode_tps", {"gpu_id": "h100_sxm5_80gb", "model": "gpt-5"})


def test_unknown_gpu_raises():
    with pytest.raises(ValueError, match="No GPU with id"):
        run_tool("decode_tps", {"gpu_id": "fake_gpu", "model": "llama-3-8b"})
