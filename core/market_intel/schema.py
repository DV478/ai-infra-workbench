from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from ulid import ULID


class FactKind(str, Enum):
    """Narrow set of fact kinds the extractor is trained to pull.

    Keeping the set tight lets the LLM categorise reliably and lets
    downstream consumers filter without parsing the subject string.
    """

    GPU_PRICE = "gpu_price"               # $/hr list or contract rate
    GPU_SHIPMENT = "gpu_shipment"         # units shipped / delivered
    CAPEX = "capex"                       # announced datacenter capex
    CAPACITY = "capacity"                 # MW, sq-ft, rack counts
    CONTRACT = "contract"                 # multi-year supply deals
    ROADMAP = "roadmap"                   # product / ship-date commitments


def _new_id() -> str:
    return str(ULID())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MarketIntelDoc(BaseModel):
    """Input: one source document the owner wants extracted."""

    doc_id: str = Field(default_factory=_new_id)
    doc_type: str  # 'earnings_transcript' | 'press_release' | 'news' | 'blog' | 'filing'
    title: str
    body: str
    source_url: str | None = None
    published_date: str | None = None  # 'YYYY-MM-DD'
    company: str | None = None
    ticker: str | None = None

    @field_validator("body")
    @classmethod
    def _non_empty_body(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("body cannot be empty")
        return v


class MarketIntelFact(BaseModel):
    """One structured fact extracted from a doc.

    The ``quote`` is the verbatim snippet the LLM claims the fact came
    from — the citation chain that mirrors ``core.sources``. If the quote
    does not appear in the source doc, the fact fails verification.
    """

    fact_id: str = Field(default_factory=_new_id)
    doc_id: str
    fact_kind: FactKind
    subject: str                          # 'H100 SXM5', 'Nscale UK capex', etc.
    value_num: float | None = None
    value_unit: str | None = None         # 'usd_per_hour', 'usd', 'units', 'mw', 'sq_ft'
    value_text: str | None = None         # free-text value if non-numeric
    period: str | None = None             # 'Q3-2025', 'FY2026', '2027-Q1'
    company: str | None = None
    quote: str                             # verbatim snippet from source
    confidence: str = "medium"            # 'high' | 'medium' | 'low'
    notes: str | None = None
    extracted_at: datetime = Field(default_factory=_utcnow)

    @field_validator("quote")
    @classmethod
    def _non_empty_quote(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("quote cannot be empty — every fact needs a citation")
        return v

    def verify_quote_in(self, body: str) -> bool:
        """True if the quote appears verbatim in the source body.

        Casefolded + whitespace-normalised to tolerate minor copy-pasting
        differences. Fails closed: punctuation differences will cause a
        miss, which is the behaviour we want.
        """
        def _norm(s: str) -> str:
            return " ".join(s.split()).casefold()

        return _norm(self.quote) in _norm(body)

    def as_row(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "doc_id": self.doc_id,
            "fact_kind": self.fact_kind.value,
            "subject": self.subject,
            "value_num": self.value_num,
            "value_unit": self.value_unit,
            "value_text": self.value_text,
            "period": self.period,
            "company": self.company,
            "quote": self.quote,
            "confidence": self.confidence,
            "notes": self.notes,
            "extracted_at": self.extracted_at,
        }
