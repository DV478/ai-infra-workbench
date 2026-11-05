from __future__ import annotations

import pytest

from core.cost_engine.kv_cache import LLAMA3_8B, LLAMA3_70B
from core.cost_engine.tco import (
    HOURS_PER_YEAR,
    OwnedEconomics,
    api_tco,
    break_even_busy_fraction,
    break_even_hourly_rate,
    compare_options,
    demand_stress,
    output_tokens_per_hour,
    owned_hourly_cost,
    owned_tco,
    rented_tco,
)
from core.gpu_specs import load_catalog


def _baseline_econ() -> OwnedEconomics:
    return OwnedEconomics(
        capex_usd=32000.0,
        depreciation_years=3,
        power_draw_watts=700.0,
        power_cost_usd_per_kwh=0.10,
        pue=1.4,
        hosting_usd_per_month=1200.0,
        financing_rate_pct=8.0,
        maintenance_pct_of_capex_per_year=3.0,
    )


# ---------- Owned -----------------------------------------------------------


def test_owned_hourly_cost_components_add_up():
    econ = _baseline_econ()
    hourly = owned_hourly_cost(econ)
    # Depreciation alone = 32000 / (3 * 8760) ≈ $1.217
    # Power = 0.7 * 1.4 * 0.10 = $0.098/hr
    # Hosting = 1200*12/8760 ≈ $1.644/hr
    # Financing = 32000 * 0.08 * 0.5 / 8760 ≈ $0.146/hr
    # Maintenance = 32000 * 0.03 / 8760 ≈ $0.110/hr
    # Sum ≈ $3.22/hr
    assert 2.8 < hourly < 3.6


def test_owned_tco_tokens_match_hours_times_throughput():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    econ = _baseline_econ()
    tph = output_tokens_per_hour(h100, LLAMA3_8B, 8.0, batch_size=32, busy_fraction=0.6)
    result = owned_tco(econ, tph, 0.6)
    expected_tokens = tph * HOURS_PER_YEAR * 3
    assert result.total_output_tokens_3y == pytest.approx(expected_tokens, rel=1e-6)
    assert result.option == "owned"
    assert result.total_usd_3y > 0


# ---------- Rented ----------------------------------------------------------


def test_rented_tco_scales_with_hourly():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    tph = output_tokens_per_hour(h100, LLAMA3_8B, 8.0, batch_size=32, busy_fraction=0.6)
    a = rented_tco("reserved", "Reserved", 2.0, tph, 0.6)
    b = rented_tco("reserved", "Reserved", 4.0, tph, 0.6)
    assert b.total_usd_3y == pytest.approx(2 * a.total_usd_3y, rel=1e-9)
    assert b.usd_per_m_tokens == pytest.approx(2 * a.usd_per_m_tokens, rel=1e-9)


def test_rented_tco_rejects_wrong_option():
    with pytest.raises(ValueError):
        rented_tco("owned", "x", 1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        rented_tco("api", "x", 1.0, 1.0, 0.5)


# ---------- API -------------------------------------------------------------


def test_api_tco_scales_linearly_with_demand():
    a = api_tco("Together", 2.0, tokens_demanded_3y=1_000_000_000)
    b = api_tco("Together", 2.0, tokens_demanded_3y=2_000_000_000)
    assert b.total_usd_3y == pytest.approx(2 * a.total_usd_3y)
    # $/M should be the same
    assert b.usd_per_m_tokens == pytest.approx(a.usd_per_m_tokens)


def test_api_tco_includes_input_cost():
    """IO ratio of 2 means 2x the output tokens were input. Include their cost."""
    a = api_tco("A", 2.0, tokens_demanded_3y=1_000_000, input_price_per_m_tokens=0.0, io_ratio=2.0)
    b = api_tco("B", 2.0, tokens_demanded_3y=1_000_000, input_price_per_m_tokens=1.0, io_ratio=2.0)
    # Extra = 1M * 2 ratio = 2M input tokens * $1/M = $2
    assert b.total_usd_3y == pytest.approx(a.total_usd_3y + 2.0)


# ---------- Break-even ------------------------------------------------------


def test_break_even_busy_fraction_lower_when_renting_is_dear():
    econ = _baseline_econ()
    # Cheap rental → owning not worth it until near-full utilisation (pivot high)
    cheap = break_even_busy_fraction(econ, rented_hourly_usd=1.0)
    # Expensive rental → owning breaks even easier (pivot low)
    dear = break_even_busy_fraction(econ, rented_hourly_usd=10.0)
    assert dear.pivot_value < cheap.pivot_value


def test_break_even_hourly_rate_matches_target():
    """At the returned hourly, rented $/M should equal the target."""
    be = break_even_hourly_rate(cost_per_m_tokens_target=2.0, output_tokens_per_hr=1_000_000)
    # (hourly * 1 hr / 1e6 tokens) * 1e6 = hourly  … should equal $2
    assert be.pivot_value == pytest.approx(2.0, rel=1e-9)


# ---------- Demand stress ---------------------------------------------------


def test_stress_leaves_api_per_m_unchanged():
    api = api_tco("A", 2.0, 1_000_000_000)
    stressed = demand_stress(api, 0.6)
    assert stressed.stressed_usd_per_m == pytest.approx(stressed.base_usd_per_m, rel=1e-6)
    # ... but total spend scales with demand
    assert stressed.stressed_usd_3y == pytest.approx(stressed.base_usd_3y * 0.6)


def test_stress_raises_owned_per_m_cost():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    econ = _baseline_econ()
    tph = output_tokens_per_hour(h100, LLAMA3_8B, 8.0, batch_size=32, busy_fraction=0.6)
    base = owned_tco(econ, tph, 0.6)
    stressed = demand_stress(base, 0.6)
    # Fixed cost, fewer tokens delivered → higher $/M
    assert stressed.stressed_usd_per_m > stressed.base_usd_per_m
    # Owned stranded cost > 0
    assert stressed.stranded_capacity_usd > 0


def test_stress_multiplier_bounds():
    api = api_tco("A", 2.0, 1_000_000)
    with pytest.raises(ValueError):
        demand_stress(api, 0.0)
    with pytest.raises(ValueError):
        demand_stress(api, 3.0)


# ---------- Compose ---------------------------------------------------------


def test_compare_options_returns_five_results():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    scenario = compare_options(
        h100,
        LLAMA3_8B,
        8.0,
        owned_econ=_baseline_econ(),
        reserved_hourly=2.5,
        on_demand_hourly=4.5,
        neocloud_hourly=2.0,
        api_name="Together",
        api_output_price_per_m=2.0,
        api_input_price_per_m=1.0,
    )
    assert len(scenario.results) == 5
    options = {r.option for r in scenario.results}
    assert options == {"owned", "reserved", "on_demand", "neocloud", "api"}
    assert len(scenario.stress) == 5
    assert len(scenario.break_evens) == 2


def test_70b_scenario_api_cheaper_at_low_demand():
    """Classic headline: owning a 70B GPU is expensive at modest demand;
    API should beat it on $/M when the workload is light."""
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    scenario = compare_options(
        h100,
        LLAMA3_70B,
        70.0,
        owned_econ=_baseline_econ(),
        reserved_hourly=3.0,
        on_demand_hourly=6.0,
        neocloud_hourly=2.5,
        api_name="Together 70B",
        api_output_price_per_m=0.90,
        api_input_price_per_m=0.90,
        dtype="fp8",
        busy_fraction=0.2,  # light utilisation
    )
    by_option = {r.option: r for r in scenario.results}
    # API should beat owned at 20% busy_fraction (lots of stranded capex)
    assert by_option["api"].usd_per_m_tokens < by_option["owned"].usd_per_m_tokens
