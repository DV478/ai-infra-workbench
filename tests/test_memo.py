from __future__ import annotations

from datetime import date

from core.cost_engine.kv_cache import LLAMA3_8B
from core.cost_engine.tco import OwnedEconomics, compare_options
from core.gpu_specs import load_catalog
from core.memo import render_memo


def _scenario():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    econ = OwnedEconomics(capex_usd=32000.0)
    return h100, compare_options(
        h100, LLAMA3_8B, 8.0,
        owned_econ=econ,
        reserved_hourly=2.5, on_demand_hourly=4.5, neocloud_hourly=2.0,
        api_name="Together AI", api_output_price_per_m=0.90, api_input_price_per_m=0.90,
    )


def test_memo_renders_without_error():
    gpu, scenario = _scenario()
    md = render_memo(
        scenario,
        gpu_name=gpu.name, gpu_id=gpu.id, dtype="fp16",
        batch_size=32, input_length=512, output_length=256,
        utilisation_frac=0.65, today=date(2026, 11, 5),
    )
    assert isinstance(md, str)
    assert len(md) > 500


def test_memo_includes_headline_sections():
    gpu, scenario = _scenario()
    md = render_memo(
        scenario, gpu_name=gpu.name, gpu_id=gpu.id, dtype="fp16",
        batch_size=32, input_length=512, output_length=256, utilisation_frac=0.65,
    )
    for section in ("Recommendation", "Workload", "Cost comparison", "Break-even", "Demand stress", "Methodology"):
        assert section in md, f"Missing section: {section!r}"


def test_memo_includes_all_five_options():
    gpu, scenario = _scenario()
    md = render_memo(
        scenario, gpu_name=gpu.name, gpu_id=gpu.id, dtype="fp16",
        batch_size=32, input_length=512, output_length=256, utilisation_frac=0.65,
    )
    for label in ("Owned", "Reserved", "On-demand", "Neocloud", "Together AI"):
        assert label in md, f"Memo missing option label: {label!r}"


def test_memo_cheapest_appears_in_recommendation():
    gpu, scenario = _scenario()
    md = render_memo(
        scenario, gpu_name=gpu.name, gpu_id=gpu.id, dtype="fp16",
        batch_size=32, input_length=512, output_length=256, utilisation_frac=0.65,
    )
    cheapest = min(scenario.results, key=lambda r: r.usd_per_m_tokens)
    # The recommendation section should name the cheapest option
    rec_section = md.split("## Workload")[0]
    assert cheapest.label in rec_section
