from __future__ import annotations

from pathlib import Path

import duckdb

from core.sources.schema import Figure, Kind, Source

_DDL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     VARCHAR PRIMARY KEY,
    figure_key    VARCHAR NOT NULL,
    value_num     DOUBLE,
    value_text    VARCHAR,
    unit          VARCHAR,
    kind          VARCHAR NOT NULL,
    citation      VARCHAR NOT NULL,
    cited_as      VARCHAR NOT NULL,
    retrieved_at  TIMESTAMP NOT NULL,
    confidence    VARCHAR NOT NULL,
    notes         VARCHAR,
    build         VARCHAR NOT NULL,
    superseded_by VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_sources_figure ON sources(figure_key);
"""


class SourceStore:
    """DuckDB-backed table of ``Source`` records.

    Use ``path=":memory:"`` in tests.
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._con = duckdb.connect(self.path)
        self._con.execute(_DDL)

    def add(self, source: Source) -> str:
        row = source.as_row()
        self._con.execute(
            """
            INSERT INTO sources
            (source_id, figure_key, value_num, value_text, unit, kind, citation,
             cited_as, retrieved_at, confidence, notes, build, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["source_id"],
                row["figure_key"],
                row["value_num"],
                row["value_text"],
                row["unit"],
                row["kind"],
                row["citation"],
                row["cited_as"],
                row["retrieved_at"],
                row["confidence"],
                row["notes"],
                row["build"],
                row["superseded_by"],
            ],
        )
        return source.source_id

    def add_many(self, sources: list[Source]) -> list[str]:
        return [self.add(s) for s in sources]

    def get(self, source_id: str) -> Source | None:
        res = self._con.execute(
            "SELECT * FROM sources WHERE source_id = ?", [source_id]
        ).fetchone()
        return _row_to_source(res, self._columns()) if res else None

    def active_for(self, figure_key: str) -> Source | None:
        """Most recent non-superseded row for a key."""
        res = self._con.execute(
            """
            SELECT * FROM sources
            WHERE figure_key = ? AND superseded_by IS NULL
            ORDER BY retrieved_at DESC
            LIMIT 1
            """,
            [figure_key],
        ).fetchone()
        return _row_to_source(res, self._columns()) if res else None

    def figure(self, figure_key: str) -> Figure:
        """Return the active ``Figure`` for a key. Raises if missing.

        Every agent tool and UI render goes through this method — it is the
        chokepoint that enforces 'no number without a source'.
        """
        src = self.active_for(figure_key)
        if src is None:
            raise KeyError(f"No active source for figure_key={figure_key!r}")
        return Figure(
            value=src.value,
            unit=src.unit,
            source_id=src.source_id,
            cited_as=src.cited_as,
            kind=src.kind,
            confidence=src.confidence,
        )

    def supersede(self, old_source_id: str, replacement: Source) -> str:
        """Add a replacement row and mark the old one superseded."""
        new_id = self.add(replacement)
        self._con.execute(
            "UPDATE sources SET superseded_by = ? WHERE source_id = ?",
            [new_id, old_source_id],
        )
        return new_id

    def all_keys(self) -> list[str]:
        return [
            r[0]
            for r in self._con.execute(
                "SELECT DISTINCT figure_key FROM sources ORDER BY figure_key"
            ).fetchall()
        ]

    def summary_by_kind(self) -> dict[str, int]:
        rows = self._con.execute(
            "SELECT kind, COUNT(*) FROM sources WHERE superseded_by IS NULL GROUP BY kind"
        ).fetchall()
        return {k: n for k, n in rows}

    def close(self) -> None:
        self._con.close()

    def _columns(self) -> list[str]:
        return [
            "source_id",
            "figure_key",
            "value_num",
            "value_text",
            "unit",
            "kind",
            "citation",
            "cited_as",
            "retrieved_at",
            "confidence",
            "notes",
            "build",
            "superseded_by",
        ]


def _row_to_source(row: tuple, columns: list[str]) -> Source:
    d = dict(zip(columns, row, strict=True))
    value = d["value_num"] if d["value_num"] is not None else d["value_text"]
    return Source(
        source_id=d["source_id"],
        figure_key=d["figure_key"],
        value=value,
        unit=d["unit"],
        kind=Kind(d["kind"]),
        citation=d["citation"],
        cited_as=d["cited_as"],
        retrieved_at=d["retrieved_at"],
        confidence=d["confidence"],
        notes=d["notes"],
        build=d["build"],
        superseded_by=d["superseded_by"],
    )
