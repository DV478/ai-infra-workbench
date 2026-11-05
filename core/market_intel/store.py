from __future__ import annotations

from pathlib import Path

import duckdb

from core.market_intel.schema import FactKind, MarketIntelDoc, MarketIntelFact

_DDL = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id         VARCHAR PRIMARY KEY,
    doc_type       VARCHAR NOT NULL,
    title          VARCHAR NOT NULL,
    body           VARCHAR NOT NULL,
    source_url     VARCHAR,
    published_date VARCHAR,
    company        VARCHAR,
    ticker         VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_docs_company ON docs(company);

CREATE TABLE IF NOT EXISTS facts (
    fact_id      VARCHAR PRIMARY KEY,
    doc_id       VARCHAR NOT NULL,
    fact_kind    VARCHAR NOT NULL,
    subject      VARCHAR NOT NULL,
    value_num    DOUBLE,
    value_unit   VARCHAR,
    value_text   VARCHAR,
    period       VARCHAR,
    company      VARCHAR,
    quote        VARCHAR NOT NULL,
    confidence   VARCHAR NOT NULL,
    notes        VARCHAR,
    extracted_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_kind    ON facts(fact_kind);
CREATE INDEX IF NOT EXISTS idx_facts_company ON facts(company);
CREATE INDEX IF NOT EXISTS idx_facts_doc     ON facts(doc_id);
"""


class MarketIntelStore:
    """DuckDB-backed table of docs and extracted facts."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._con = duckdb.connect(self.path)
        self._con.execute(_DDL)

    def add_doc(self, doc: MarketIntelDoc) -> str:
        self._con.execute(
            """
            INSERT INTO docs (doc_id, doc_type, title, body, source_url,
                              published_date, company, ticker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            [doc.doc_id, doc.doc_type, doc.title, doc.body, doc.source_url,
             doc.published_date, doc.company, doc.ticker],
        )
        return doc.doc_id

    def add_fact(self, fact: MarketIntelFact) -> str:
        row = fact.as_row()
        self._con.execute(
            """
            INSERT INTO facts (fact_id, doc_id, fact_kind, subject, value_num,
                               value_unit, value_text, period, company, quote,
                               confidence, notes, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["fact_id"], row["doc_id"], row["fact_kind"], row["subject"],
                row["value_num"], row["value_unit"], row["value_text"], row["period"],
                row["company"], row["quote"], row["confidence"], row["notes"],
                row["extracted_at"],
            ],
        )
        return fact.fact_id

    def add_facts(self, facts: list[MarketIntelFact]) -> list[str]:
        return [self.add_fact(f) for f in facts]

    def count(self, kind: FactKind | None = None) -> int:
        if kind is None:
            return self._con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        return self._con.execute(
            "SELECT COUNT(*) FROM facts WHERE fact_kind = ?", [kind.value]
        ).fetchone()[0]

    def facts_for_subject(self, subject: str) -> list[dict]:
        rows = self._con.execute(
            "SELECT * FROM facts WHERE subject = ? ORDER BY extracted_at DESC",
            [subject],
        ).fetchall()
        cols = [d[0] for d in self._con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def recent_facts(self, limit: int = 20) -> list[dict]:
        rows = self._con.execute(
            "SELECT * FROM facts ORDER BY extracted_at DESC LIMIT ?", [limit]
        ).fetchall()
        cols = [d[0] for d in self._con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def summary_by_kind(self) -> dict[str, int]:
        rows = self._con.execute(
            "SELECT fact_kind, COUNT(*) FROM facts GROUP BY fact_kind"
        ).fetchall()
        return {k: n for k, n in rows}

    def close(self) -> None:
        self._con.close()
