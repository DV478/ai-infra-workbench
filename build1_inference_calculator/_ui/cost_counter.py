"""Advisor cost accounting + per-turn log persistence.

Two responsibilities:

1. ``cost_for_turn(model, usage)`` — pure math. Given a model id and a usage
   dict from the LLM response, return a breakdown (input/output/cached tokens,
   $, £, cache-hit %).

2. ``log_turn(db_path, ...)`` — append a row to ``data/advisor_turns.duckdb``
   so the Cost tracker page (M7) has something to render. ``load_turns``
   reads back. ``aggregates`` summarises for the hero row.

Streamlit-free — ``render_cost_badge`` is a separate helper in M6's Advisor
page that imports ``streamlit`` and calls ``cost_for_turn`` here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

# USD → GBP; refresh quarterly. Not worth a live FX API for a demo.
# Set near the mid-November 2026 market rate.
USD_TO_GBP: float = 0.78

# Per-model pricing, $ per million tokens (Nov 2026).
# OpenAI figures from openai.com/api/pricing; Anthropic from anthropic.com/pricing.
PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    },
    "claude-haiku-5-5": {
        "input": 0.10,
        "cached_input": 0.01,  # Anthropic prompt cache ~10% of standard
        "output": 0.50,
    },
}

PRICE_CARD_LAST_CONFIRMED: str = "2026-11-08"

# DuckDB schema for turn logs.
TURN_LOG_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS advisor_turns (
    ts TIMESTAMP,
    model TEXT,
    input_tok INTEGER,
    output_tok INTEGER,
    cached_tok INTEGER,
    latency_ms REAL,
    usd REAL,
    gbp REAL,
    question TEXT,
    answer_md TEXT
)
"""


@dataclass(frozen=True)
class TurnCost:
    """Per-turn cost breakdown. Serialisable with ``asdict``."""

    model: str
    input_tok: int
    output_tok: int
    cached_tok: int
    usd: float
    gbp: float
    cache_hit_pct: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_usage(usage: dict[str, Any]) -> tuple[int, int, int]:
    """Normalise usage dicts from OpenAI and Anthropic into one shape.

    Returns ``(input_tok, output_tok, cached_tok)``.

    OpenAI usage: ``prompt_tokens``, ``completion_tokens``, and
    ``prompt_tokens_details.cached_tokens`` (0 if not cached).

    Anthropic usage: ``input_tokens``, ``output_tokens``,
    ``cache_read_input_tokens`` (0 if no cache hit),
    ``cache_creation_input_tokens`` (treated as uncached — you pay for them).
    """
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        input_tok = int(usage.get("prompt_tokens", 0))
        output_tok = int(usage.get("completion_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        cached_tok = int(details.get("cached_tokens", 0))
        return input_tok, output_tok, cached_tok

    input_tok = int(usage.get("input_tokens", 0))
    output_tok = int(usage.get("output_tokens", 0))
    cached_tok = int(usage.get("cache_read_input_tokens", 0))
    return input_tok, output_tok, cached_tok


def cost_for_turn(model: str, usage: dict[str, Any]) -> TurnCost:
    """Compute $ and £ cost for a single turn.

    ``cached_tok`` is counted at the cached-input rate; the remaining
    ``input_tok - cached_tok`` is charged at the standard input rate.
    """
    if model not in PRICE_PER_MTOK:
        raise ValueError(
            f"No price card for model {model!r}. Known: {sorted(PRICE_PER_MTOK)}"
        )
    price = PRICE_PER_MTOK[model]
    input_tok, output_tok, cached_tok = _extract_usage(usage)
    uncached_input = max(input_tok - cached_tok, 0)

    usd = (
        uncached_input * price["input"]
        + cached_tok * price["cached_input"]
        + output_tok * price["output"]
    ) / 1_000_000.0
    gbp = usd * USD_TO_GBP
    cache_hit_pct = (cached_tok / input_tok * 100.0) if input_tok > 0 else 0.0

    return TurnCost(
        model=model,
        input_tok=input_tok,
        output_tok=output_tok,
        cached_tok=cached_tok,
        usd=usd,
        gbp=gbp,
        cache_hit_pct=cache_hit_pct,
    )


def log_turn(
    db_path: str | Path,
    *,
    cost: TurnCost,
    latency_ms: float,
    question: str,
    answer_md: str,
    ts: datetime | None = None,
) -> None:
    """Append one row to the advisor turn log. Creates the DB + table on first use.

    Writes are tiny (<1 KB each). No batching, no WAL trickery — DuckDB is
    perfectly happy with append-one, close-conn, repeat.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    try:
        conn.execute(TURN_LOG_SCHEMA)
        conn.execute(
            "INSERT INTO advisor_turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ts or datetime.now(UTC),
                cost.model,
                cost.input_tok,
                cost.output_tok,
                cost.cached_tok,
                latency_ms,
                cost.usd,
                cost.gbp,
                question,
                answer_md,
            ],
        )
    finally:
        conn.close()


def load_turns(db_path: str | Path) -> list[dict[str, Any]]:
    """Return every logged turn, most recent first. Empty list if DB missing or empty."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = duckdb.connect(str(path), read_only=True)
    try:
        table_exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'advisor_turns'"
        ).fetchone()
        if not table_exists or table_exists[0] == 0:
            return []
        rows = conn.execute(
            "SELECT ts, model, input_tok, output_tok, cached_tok, "
            "latency_ms, usd, gbp, question, answer_md "
            "FROM advisor_turns ORDER BY ts DESC"
        ).fetchall()
    finally:
        conn.close()

    cols = [
        "ts",
        "model",
        "input_tok",
        "output_tok",
        "cached_tok",
        "latency_ms",
        "usd",
        "gbp",
        "question",
        "answer_md",
    ]
    return [dict(zip(cols, r, strict=True)) for r in rows]


def clear_log(db_path: str | Path) -> None:
    """Delete the turn log file. Used by the Cost tracker page's reset button."""
    path = Path(db_path)
    if path.exists():
        path.unlink()


@dataclass(frozen=True)
class TurnAggregates:
    """Summary for the Cost tracker hero row."""

    turns: int
    total_gbp: float
    avg_gbp: float
    cache_hit_pct: float
    avg_latency_ms: float


def aggregates(turns: list[dict[str, Any]]) -> TurnAggregates:
    """Roll up a list of turn rows into hero-row stats."""
    if not turns:
        return TurnAggregates(0, 0.0, 0.0, 0.0, 0.0)

    total_input = sum(r["input_tok"] for r in turns)
    total_cached = sum(r["cached_tok"] for r in turns)
    total_gbp = sum(r["gbp"] for r in turns)
    total_latency = sum(r["latency_ms"] for r in turns)
    n = len(turns)

    return TurnAggregates(
        turns=n,
        total_gbp=total_gbp,
        avg_gbp=total_gbp / n,
        cache_hit_pct=(total_cached / total_input * 100.0) if total_input > 0 else 0.0,
        avg_latency_ms=total_latency / n,
    )


def aggregates_by_model(turns: list[dict[str, Any]]) -> dict[str, TurnAggregates]:
    """Per-model roll-up for the side-by-side comparison chart."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in turns:
        by_model.setdefault(r["model"], []).append(r)
    return {m: aggregates(rows) for m, rows in by_model.items()}


def default_db_path() -> Path:
    """Canonical location for the turn log — shared by Advisor + Cost tracker."""
    # ``build1_inference_calculator/_ui/cost_counter.py`` → repo root is parents[2]
    return Path(__file__).resolve().parents[2] / "data" / "advisor_turns.duckdb"
