from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.sources import Confidence, Kind, Source, SourceStore
from core.sources.schema import Build


def _mk(figure_key: str = "h100_sxm5.hbm_bandwidth_gbs", value: float = 3350.0) -> Source:
    return Source(
        figure_key=figure_key,
        value=value,
        unit="GB/s",
        kind=Kind.MEASURED,
        citation="https://www.nvidia.com/.../h100-datasheet.pdf",
        cited_as="NVIDIA H100 datasheet",
        confidence=Confidence.HIGH,
        build=Build.SHARED,
    )


def test_source_round_trip():
    store = SourceStore()
    src = _mk()
    store.add(src)

    got = store.get(src.source_id)
    assert got is not None
    assert got.value == 3350.0
    assert got.unit == "GB/s"
    assert got.kind is Kind.MEASURED


def test_figure_lookup_returns_only_active():
    store = SourceStore()
    first = _mk(value=3000.0)
    store.add(first)

    fig1 = store.figure("h100_sxm5.hbm_bandwidth_gbs")
    assert fig1.value == 3000.0

    newer = _mk(value=3350.0)
    store.supersede(first.source_id, newer)

    fig2 = store.figure("h100_sxm5.hbm_bandwidth_gbs")
    assert fig2.value == 3350.0
    assert fig2.source_id == newer.source_id


def test_figure_missing_raises():
    store = SourceStore()
    with pytest.raises(KeyError):
        store.figure("nonexistent.key")


def test_figure_key_must_be_dotted():
    with pytest.raises(ValidationError):
        Source(
            figure_key="notdotted",
            value=1.0,
            kind=Kind.ASSUMPTION,
            citation="x",
            cited_as="x",
        )


def test_summary_by_kind_counts_only_active():
    store = SourceStore()
    store.add(_mk("a.b", 1.0))
    store.add(
        Source(
            figure_key="c.d",
            value=2.0,
            kind=Kind.MODELLED,
            citation="x",
            cited_as="x",
        )
    )
    counts = store.summary_by_kind()
    assert counts.get("measured") == 1
    assert counts.get("modelled") == 1


def test_text_valued_source():
    store = SourceStore()
    note = Source(
        figure_key="h100_sxm5.form_factor",
        value="SXM5",
        kind=Kind.MEASURED,
        citation="datasheet",
        cited_as="H100 datasheet",
    )
    store.add(note)
    fig = store.figure("h100_sxm5.form_factor")
    assert fig.value == "SXM5"
