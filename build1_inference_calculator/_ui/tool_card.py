"""Named tool-call cards for the Advisor page.

Replaces the raw ``st.json(tool_call)`` dump with a titled, emoji-tagged
card plus a one-line summary of the result. The raw JSON sits in a nested
expander for the readers who want it.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# Map of tool name → (emoji, friendly title, result-summary function).
_SUMMARISERS: dict[str, tuple[str, str]] = {
    "check_memory_fit":       ("📏", "Memory fit check"),
    "size_cluster":           ("🧮", "Cluster sizing"),
    "compute_token_cost":     ("💷", "Token cost"),
    "compare_tco":            ("💷", "TCO comparison"),
    "compute_break_even":     ("⚖️", "Break-even"),
    "compute_demand_stress":  ("🧯", "Demand-stress test"),
    "api_margin":             ("🧾", "Benchmark comparison (reasoned, not an accusation)"),
    "roofline_throughput":    ("📈", "Roofline throughput"),
    "list_gpus":              ("🗂️", "GPU catalog lookup"),
    "list_models":            ("🗂️", "Model catalog lookup"),
}


def tool_display(tool_name: str) -> tuple[str, str]:
    """Get the (emoji, title) pair for a tool name. Falls back to the raw name."""
    return _SUMMARISERS.get(tool_name, ("🔧", tool_name))


def _one_line_summary(tool_name: str, result: Any) -> str:
    """Best-effort one-liner describing what the tool returned.

    We deliberately swallow errors — if the shape doesn't match, we show
    the raw type so the UI never crashes on a schema drift.
    """
    try:
        if not isinstance(result, dict):
            return f"Returned {type(result).__name__}."

        if tool_name == "check_memory_fit":
            fits = result.get("fits")
            model = result.get("model") or "the model"
            gpu = result.get("gpu") or "the GPU"
            return f"{'Fits' if fits else 'Does not fit'}: {model} on {gpu}."

        if tool_name == "compare_tco":
            opts = result.get("options") or []
            if opts:
                best = min(
                    opts,
                    key=lambda o: o.get("usd_per_m_output_tokens", float("inf")),
                )
                name = best.get("name", "unknown")
                cost = best.get("usd_per_m_output_tokens")
                if cost is not None:
                    return f"Cheapest: {name} at ${cost:.2f}/M output tokens."
            return f"{len(opts)} options compared."

        if tool_name == "compute_break_even":
            note = result.get("note")
            pivot = result.get("pivot_value")
            if note:
                return note
            if pivot is not None:
                return f"Break-even at {pivot:.2f}."

        if tool_name == "compute_demand_stress":
            delta = result.get("delta_vs_plan")
            if isinstance(delta, (int, float)):
                return f"Cost per M tokens shifts {delta:+.1%} under a 40% miss."

        if tool_name == "api_margin":
            margin = result.get("gross_margin_pct")
            if isinstance(margin, (int, float)):
                return f"Implied gross margin: {margin:.0f}%."

        if tool_name in ("compute_token_cost", "roofline_throughput"):
            keys = sorted(k for k in result if not k.startswith("_"))[:3]
            return "Keys: " + ", ".join(keys) if keys else "Returned a result."

        if tool_name in ("list_gpus", "list_models"):
            n = len(result.get("items") or result.get("ids") or [])
            return f"{n} entries returned."

        keys = sorted(k for k in result if not k.startswith("_"))[:3]
        return "Returned " + ", ".join(keys) if keys else "Returned an empty result."
    except Exception as e:  # pragma: no cover — safety net
        return f"(result shape changed: {type(e).__name__})"


def render_tool_call(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_result: Any,
) -> None:
    """Render one tool call as a titled card inside the chat transcript.

    Layout:
        <emoji> **Friendly title**
        One-line summary of the result.
        ▸ Raw input / output JSON (expander)
    """
    emoji, title = tool_display(tool_name)
    summary = _one_line_summary(tool_name, tool_result)

    with st.container(border=True):
        st.markdown(f"{emoji} **{title}**")
        if summary:
            st.caption(summary)
        with st.expander("Raw input / output JSON", expanded=False):
            st.json({"input": tool_input, "result": tool_result})
