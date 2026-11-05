from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.mlperf.ingest import (
    cross_check,
    load_bundled_results,
    normalise_system,
    read_results_csv,
)

# ---------- System string normalisation ------------------------------------


@pytest.mark.parametrize(
    "s,expected_id,expected_count",
    [
        ("Dell PowerEdge XE9680, 8x H100 SXM5-80GB", "h100_sxm5_80gb", 8),
        ("Supermicro SYS-821GE-TNHR, 8x H200 SXM", "h200_sxm_141gb", 8),
        ("1x NVIDIA H100 PCIe 80GB", "h100_pcie_80gb", 1),
        ("HGX B200 8-GPU", "b200", None),
        ("Lenovo ThinkSystem, 4x L40S", "l40s", 4),
        ("Intrinsic FOObarred", None, None),
    ],
)
def test_normalise_system(s, expected_id, expected_count):
    gpu_id, count = normalise_system(s)
    assert gpu_id == expected_id
    assert count == expected_count


# ---------- CSV read --------------------------------------------------------


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "v5.0__test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_read_results_csv_normalises_rows(tmp_path):
    rows = [
        {
            "Submitter": "Dell",
            "System": "Dell PowerEdge, 8x H100 SXM5-80GB",
            "Benchmark": "Llama2-70B",
            "Scenario": "Server",
            "Result": 23000.0,
            "Availability": "available",
        },
        {
            "Submitter": "Supermicro",
            "System": "SYS-821GE, 8x H200 SXM",
            "Benchmark": "Llama2-70B",
            "Scenario": "Offline",
            "Result": 30000.0,
            "Availability": "available",
        },
    ]
    csv = _write_csv(tmp_path, rows)
    parsed = read_results_csv(csv, "v5.0")
    assert len(parsed) == 2
    assert parsed[0].gpu_id == "h100_sxm5_80gb"
    assert parsed[0].accelerator_count == 8
    assert parsed[1].gpu_id == "h200_sxm_141gb"
    assert parsed[1].round == "v5.0"


def test_read_results_csv_raises_on_missing_columns(tmp_path):
    csv = _write_csv(tmp_path, [{"foo": 1, "bar": 2}])
    with pytest.raises(ValueError, match="missing required column"):
        read_results_csv(csv, "v5.0")


def test_load_bundled_results_empty_dir(tmp_path):
    # Empty dir → empty list, no crash
    empty = tmp_path / "mlperf_empty"
    empty.mkdir()
    assert load_bundled_results(empty) == []


def test_load_bundled_results_nonexistent_dir():
    # Nonexistent path → empty list
    assert load_bundled_results(Path("/nonexistent/path")) == []


# ---------- Cross-check -----------------------------------------------------


def test_cross_check_computes_error_pct(tmp_path):
    rows = [
        {
            "Submitter": "Dell",
            "System": "8x H100 SXM5-80GB",
            "Benchmark": "Llama2-70B",
            "Scenario": "Server",
            "Result": 20000.0,  # total across 8 GPUs → per-GPU 2500
            "Availability": "available",
        },
    ]
    csv = _write_csv(tmp_path, rows)
    parsed = read_results_csv(csv, "v5.0")

    def predict(gpu_id, model, scenario):
        return 2750.0  # 10% over measured → should get +10% error

    checks = cross_check(parsed, predict)
    assert len(checks) == 1
    c = checks[0]
    assert c.mlperf_tps_per_gpu == pytest.approx(2500.0)
    assert c.predicted_tps_per_gpu == 2750.0
    assert c.error_pct == pytest.approx(10.0)


def test_cross_check_skips_unmappable(tmp_path):
    rows = [{"Submitter": "X", "System": "Mystery", "Benchmark": "foo", "Scenario": "Server", "Result": 100.0}]
    csv = _write_csv(tmp_path, rows)
    parsed = read_results_csv(csv, "v5.0")

    def predict(*_):
        return 200.0

    # gpu_id is None → skipped
    checks = cross_check(parsed, predict)
    assert checks == []


def test_cross_check_skips_when_predict_returns_none(tmp_path):
    rows = [
        {
            "Submitter": "Dell",
            "System": "8x H100 SXM5-80GB",
            "Benchmark": "Mixtral-8x7B",
            "Scenario": "Server",
            "Result": 10000.0,
        }
    ]
    csv = _write_csv(tmp_path, rows)
    parsed = read_results_csv(csv, "v5.0")

    def predict(gpu_id, model, scenario):
        return None  # we don't predict this model yet

    checks = cross_check(parsed, predict)
    assert checks == []
