from __future__ import annotations

import pytest

from core.cost_engine.kv_cache import LLAMA3_8B, LLAMA3_70B
from core.cost_engine.token_cost import (
    cost_per_m_tokens,
    implied_api_margin,
    sweep_cost,
)
from core.gpu_specs import load_catalog


def test_cost_per_m_tokens_scales_inversely_with_throughput():
    """Doubling throughput (via bigger batch) should roughly halve $/M tokens
    until the kernel runs out of compute headroom."""
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    cheap = cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=3.0, batch_size=32)
    cheaper = cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=3.0, batch_size=64)
    assert cheaper.cost_per_m_tokens_realistic < cheap.cost_per_m_tokens_realistic


def test_cost_scales_linearly_with_hourly():
    """Doubling hourly rate doubles $/M tokens — no batching effect."""
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    a = cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=2.0, batch_size=16)
    b = cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=4.0, batch_size=16)
    assert b.cost_per_m_tokens_realistic == pytest.approx(2 * a.cost_per_m_tokens_realistic, rel=1e-6)


def test_busy_fraction_pushes_realistic_cost_above_peak():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    tc = cost_per_m_tokens(
        h100, LLAMA3_8B, 8.0, hourly_usd=3.0, batch_size=16, busy_fraction=0.5
    )
    assert tc.cost_per_m_tokens_realistic > tc.cost_per_m_tokens_peak


def test_negative_hourly_rejected():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    with pytest.raises(ValueError):
        cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=-1.0)


def test_busy_fraction_bounds_rejected():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    with pytest.raises(ValueError):
        cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=3.0, busy_fraction=0.0)
    with pytest.raises(ValueError):
        cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=3.0, busy_fraction=1.5)


def test_prefill_share_is_nonnegative():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    tc = cost_per_m_tokens(
        h100, LLAMA3_8B, 8.0, hourly_usd=3.0, batch_size=32, input_length=1024, output_length=128
    )
    assert tc.prefill_share_usd >= 0
    # With an IO ratio of 8, prefill share should be meaningful (not zero)
    assert tc.prefill_share_usd > 0


def test_sweep_cost_returns_correct_count():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    results = sweep_cost(h100, LLAMA3_8B, 8.0, hourly_usd=3.0, batches=(1, 4, 16))
    assert len(results) == 3
    assert [r.batch_size for r in results] == [1, 4, 16]


def test_implied_margin_against_cheap_api():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    tc = cost_per_m_tokens(h100, LLAMA3_8B, 8.0, hourly_usd=3.0, batch_size=32)
    # Cheap API — margin should be low or negative (loss leader)
    margin_cheap = implied_api_margin("Cheap API", 0.10, tc)
    # Expensive API — margin should be positive
    margin_rich = implied_api_margin("Rich API", 10.0, tc)
    assert margin_rich.gross_margin_pct > margin_cheap.gross_margin_pct
    assert margin_rich.gross_margin_pct > 0


def test_llama_70b_fp8_on_h100_is_reasonable():
    """Spot-check headline demo number. Not a tight bound; just sanity."""
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    tc = cost_per_m_tokens(
        h100, LLAMA3_70B, 70.0, hourly_usd=3.0, batch_size=32, dtype="fp8"
    )
    # Must be positive, finite and in a plausible range: $0.1-$5/M tokens
    assert 0.1 < tc.cost_per_m_tokens_realistic < 5.0
    assert tc.bound in ("compute", "memory")
