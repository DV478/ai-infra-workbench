"""Measured / modelled / simulated / assumption badges.

Promotes the single most credible pattern in the app — "every number is
tagged" — from a one-liner on the landing page into a visible badge on every
hero metric and headline table across all five pages.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

ProvenanceKind = Literal["measured", "modelled", "simulated", "assumption"]

_BADGES: dict[ProvenanceKind, tuple[str, str]] = {
    "measured":   ("🟢", "Measured"),
    "modelled":   ("🟡", "Modelled"),
    "simulated":  ("🟠", "Simulated"),
    "assumption": ("⚪", "Assumption"),
}


def badge(kind: ProvenanceKind) -> str:
    """Return the badge markdown string: ``'🟢 Measured'``.

    Pure — no streamlit calls, safe to call from any context (including log
    statements and table cell formatters).
    """
    try:
        emoji, label = _BADGES[kind]
    except KeyError as e:
        raise ValueError(
            f"Unknown provenance kind {kind!r}. One of: {list(_BADGES)}"
        ) from e
    return f"{emoji} {label}"


def legend_expander() -> None:
    """Render a collapsible legend explaining what each badge means."""
    with st.expander("Badge legend"):
        st.markdown(
            "- 🟢 **Measured** — from an actual hardware run (Kaggle T4, MLPerf submitter).\n"
            "- 🟡 **Modelled** — computed from physics or formulas (roofline, TCO math).\n"
            "- 🟠 **Simulated** — from a traced workload replay (Alibaba / Azure cluster traces).\n"
            "- ⚪ **Assumption** — a reasonable default the author chose. Hover any assumption for the rationale."
        )


_CONFIDENCE_BADGES: dict[str, str] = {
    "high":   "🟢 high",
    "medium": "🟡 medium",
    "low":    "🔴 low",
}


def confidence_badge(level: str) -> str:
    """Catalog confidence → coloured label. Used in Spec + Market Intel tables."""
    return _CONFIDENCE_BADGES.get(level, level)
