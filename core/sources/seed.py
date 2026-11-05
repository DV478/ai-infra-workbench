"""Seed the SourceStore with one row per numeric field in gpu_specs.yaml.

The YAML is the convenient edit surface; the sources table is the audit
surface. Running ``python -m core.sources.seed`` creates a fresh
``data/sources.duckdb`` and writes one row per (gpu_id, field) with the
citation URL already attached. The Streamlit Spec page reads from the
DuckDB table so figures and citations can never drift apart.

Design: the catalog's ``citation`` field applies to the whole GPU row, so
every seeded field inherits it. If a figure is ever revised we insert a
new row with ``supersede`` rather than editing YAML in-place, preserving
history.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.gpu_specs.catalog import GpuSpec, load_catalog
from core.sources.schema import Build, Confidence, Kind, Source
from core.sources.store import SourceStore

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sources.duckdb"

# Numeric fields on GpuSpec to seed. Each tuple is (attr, unit, ignore_if_none).
_NUMERIC_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("fp16_tflops_dense", "TFLOPS", False),
    ("fp8_tflops_dense", "TFLOPS", True),
    ("int8_tops_dense", "TOPS", True),
    ("memory_gb", "GB", False),
    ("memory_bandwidth_gbs", "GB/s", False),
    ("tdp_watts", "W", True),
    ("nvlink_bandwidth_gbs", "GB/s", True),
    ("pcie_gen", "gen", True),
)


def _spec_to_sources(spec: GpuSpec) -> list[Source]:
    """One Source row per non-null numeric field, plus one for memory_type."""
    cited_as = f"{spec.name} datasheet"
    retrieved_at = datetime.combine(spec.citation_date, datetime.min.time())
    confidence = Confidence(spec.confidence)

    rows: list[Source] = []
    for attr, unit, skip_if_none in _NUMERIC_FIELDS:
        value = getattr(spec, attr)
        if value is None:
            if skip_if_none:
                continue
            raise ValueError(f"Required field {spec.id}.{attr} is null")
        rows.append(
            Source(
                figure_key=f"{spec.id}.{attr}",
                value=float(value) if isinstance(value, (int, float)) else value,
                unit=unit,
                kind=Kind.MEASURED if confidence == Confidence.HIGH else Kind.MODELLED,
                citation=spec.citation,
                cited_as=cited_as,
                retrieved_at=retrieved_at,
                confidence=confidence,
                notes=spec.notes,
                build=Build.SHARED,
            )
        )

    text_kind = Kind.MEASURED if confidence == Confidence.HIGH else Kind.MODELLED
    rows.append(
        Source(
            figure_key=f"{spec.id}.memory_type",
            value=spec.memory_type,
            unit=None,
            kind=text_kind,
            citation=spec.citation,
            cited_as=cited_as,
            retrieved_at=retrieved_at,
            confidence=confidence,
            build=Build.SHARED,
        )
    )
    rows.append(
        Source(
            figure_key=f"{spec.id}.form_factor",
            value=spec.form_factor,
            unit=None,
            kind=text_kind,
            citation=spec.citation,
            cited_as=cited_as,
            retrieved_at=retrieved_at,
            confidence=confidence,
            build=Build.SHARED,
        )
    )
    return rows


def seed(db_path: str | Path = DEFAULT_DB_PATH) -> tuple[int, int]:
    """Load the catalog and write all Source rows to the DuckDB file.

    Returns ``(gpus_seeded, sources_added)``. Safe to call repeatedly; it
    creates a new file each time.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and db_path != Path(":memory:"):
        db_path.unlink()

    catalog = load_catalog()
    store = SourceStore(db_path)
    try:
        total = 0
        for spec in catalog.gpus:
            rows = _spec_to_sources(spec)
            store.add_many(rows)
            total += len(rows)
        return len(catalog.gpus), total
    finally:
        store.close()


def main() -> int:
    gpus, srcs = seed()
    print(f"Seeded {gpus} GPUs -> {srcs} Source rows at {DEFAULT_DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
