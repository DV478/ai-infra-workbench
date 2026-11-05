from __future__ import annotations

import pytest

from core.gpu_specs import load_catalog


def test_catalog_loads():
    cat = load_catalog()
    ids = cat.ids()

    required = [
        "t4", "l4", "l40s",
        "a100_sxm_80gb",
        "h100_sxm5_80gb", "h100_pcie_80gb", "h200_sxm_141gb",
        "b100", "b200", "gb200", "b300",
        "r100",
        "tpu_v5e", "tpu_v5p", "tpu_trillium",
        "trn1", "trn2",
    ]
    for gpu_id in required:
        assert gpu_id in ids, f"missing {gpu_id} in catalog"


def test_hopper_and_blackwell_are_nvidia():
    cat = load_catalog()
    for gid in ["h100_sxm5_80gb", "h200_sxm_141gb", "b200", "b300", "r100"]:
        assert cat.by_id(gid).vendor == "nvidia"


def test_peak_flops_and_bandwidth_computed():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    assert h100.peak_fp16_flops == pytest.approx(989e12)
    assert h100.peak_memory_bandwidth == pytest.approx(3350e9)


def test_rubin_marked_low_confidence():
    """Rubin numbers are announcement-only; if the YAML drifts to high
    confidence without an independent benchmark citation, this test flags it."""
    r100 = load_catalog().by_id("r100")
    assert r100.confidence == "low"


def test_h100_bandwidth_is_measured_physics():
    """Guardrail: the H100 SXM5 HBM3 bandwidth is a well-known physical spec
    (3350 GB/s). If this changes it either means the datasheet was revised
    (update citation_date) or there's a typo."""
    h100 = load_catalog().by_id("h100_sxm5_80gb")
    assert h100.memory_bandwidth_gbs == 3350
