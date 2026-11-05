from __future__ import annotations

from core.gpu_specs import load_catalog
from core.sources.schema import Confidence, Kind
from core.sources.seed import _spec_to_sources, seed
from core.sources.store import SourceStore


def test_spec_to_sources_covers_all_numeric_fields():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")  # has all optional fields populated
    rows = _spec_to_sources(h100)
    keys = {r.figure_key for r in rows}

    # Required numerics
    assert "h100_sxm5_80gb.fp16_tflops_dense" in keys
    assert "h100_sxm5_80gb.memory_gb" in keys
    assert "h100_sxm5_80gb.memory_bandwidth_gbs" in keys
    # Optional, populated on H100
    assert "h100_sxm5_80gb.fp8_tflops_dense" in keys
    assert "h100_sxm5_80gb.tdp_watts" in keys
    assert "h100_sxm5_80gb.nvlink_bandwidth_gbs" in keys
    # Text fields
    assert "h100_sxm5_80gb.memory_type" in keys
    assert "h100_sxm5_80gb.form_factor" in keys


def test_spec_to_sources_skips_null_optionals():
    cat = load_catalog()
    t4 = cat.by_id("t4")  # has null fp8_tflops_dense and nvlink
    rows = _spec_to_sources(t4)
    keys = {r.figure_key for r in rows}

    # Required present
    assert "t4.fp16_tflops_dense" in keys
    assert "t4.memory_gb" in keys
    # Optional nulls skipped
    assert "t4.fp8_tflops_dense" not in keys
    assert "t4.nvlink_bandwidth_gbs" not in keys


def test_spec_to_sources_confidence_maps_to_kind():
    """High confidence → measured. Medium/low → modelled (figure is our model
    of what the vendor claims, not an independent measurement)."""
    cat = load_catalog()
    t4 = cat.by_id("t4")  # high
    r100 = cat.by_id("r100")  # low
    t4_rows = _spec_to_sources(t4)
    r100_rows = _spec_to_sources(r100)
    assert all(r.kind is Kind.MEASURED for r in t4_rows)
    assert all(r.kind is Kind.MODELLED for r in r100_rows)
    assert all(r.confidence is Confidence.HIGH for r in t4_rows)
    assert all(r.confidence is Confidence.LOW for r in r100_rows)


def test_seed_populates_in_memory_store(tmp_path):
    db = tmp_path / "test_sources.duckdb"
    gpus, srcs = seed(db)
    assert gpus > 15  # we have 17 GPUs in the catalog
    assert srcs > gpus * 5  # at least 5 fields per GPU

    store = SourceStore(db)
    try:
        # Spot-check: H100 bandwidth lookup works end-to-end
        fig = store.figure("h100_sxm5_80gb.memory_bandwidth_gbs")
        assert fig.value == 3350.0
        assert fig.unit == "GB/s"
        assert fig.kind is Kind.MEASURED

        # Rubin R100: should be low confidence
        fig2 = store.figure("r100.memory_bandwidth_gbs")
        assert fig2.confidence is Confidence.LOW
    finally:
        store.close()
