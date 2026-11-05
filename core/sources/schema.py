from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from ulid import ULID


class Kind(str, Enum):
    MEASURED = "measured"
    MODELLED = "modelled"
    SIMULATED = "simulated"
    ASSUMPTION = "assumption"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Build(str, Enum):
    SHARED = "shared"
    BUILD1 = "build1"
    BUILD2 = "build2"
    BUILD3 = "build3"


def _new_id() -> str:
    return str(ULID())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Source(BaseModel):
    """A single provenance record for one figure.

    ``figure_key`` is the stable identifier the rest of the code refers to, e.g.
    ``h100_sxm5.hbm_bandwidth_gbs``. Multiple ``Source`` rows can share a
    ``figure_key`` when a value is revised — only one is active
    (``superseded_by IS NULL``).
    """

    source_id: str = Field(default_factory=_new_id)
    figure_key: str
    value: float | int | str | None
    unit: str | None = None
    kind: Kind
    citation: str
    cited_as: str
    retrieved_at: datetime = Field(default_factory=_utcnow)
    confidence: Confidence = Confidence.MEDIUM
    notes: str | None = None
    build: Build = Build.SHARED
    superseded_by: str | None = None

    @field_validator("figure_key")
    @classmethod
    def _non_empty_key(cls, v: str) -> str:
        if not v or "." not in v:
            raise ValueError("figure_key must be dotted, e.g. 'h100_sxm5.hbm_bandwidth_gbs'")
        return v

    def as_row(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "figure_key": self.figure_key,
            "value_num": self.value if isinstance(self.value, (int, float)) else None,
            "value_text": self.value if isinstance(self.value, str) else None,
            "unit": self.unit,
            "kind": self.kind.value,
            "citation": self.citation,
            "cited_as": self.cited_as,
            "retrieved_at": self.retrieved_at,
            "confidence": self.confidence.value,
            "notes": self.notes,
            "build": self.build.value,
            "superseded_by": self.superseded_by,
        }


class Figure(BaseModel):
    """What agent tools return. Never a raw float."""

    value: float | int | str | None
    unit: str | None = None
    source_id: str
    cited_as: str
    kind: Kind
    confidence: Confidence

    def __str__(self) -> str:
        v = self.value
        u = f" {self.unit}" if self.unit else ""
        return f"{v}{u} [{self.kind.value}; {self.cited_as}]"
