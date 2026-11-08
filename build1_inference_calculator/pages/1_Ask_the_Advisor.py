"""Advisor page — natural-language sizing questions.

Strongest page in Build 1. Chat + transparent tool-call panel. This rewrite
adds:
- Two-model picker above the chat (gpt-4o-mini default, Claude Haiku 5.5 switch).
- Per-turn £ cost badge, session total, latency, cache-hit % (Anthropic).
- Starter-question chips so the first click is free.
- Named tool-call cards (one card per tool, raw JSON nested).
- Canned demo transcript on the no-key path — no more developer schema explorer.
- Clear-chat button + Cost tracker link.
- Every completed turn persisted to data/advisor_turns.duckdb via log_turn.

The Advisor never calculates a figure — every number comes from a tool call
against core/cost_engine. See compute_advisor.SYSTEM_PROMPT.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD1 = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _BUILD1):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402
from _ui.cost_counter import (  # noqa: E402
    PRICE_PER_MTOK,
    cost_for_turn,
    default_db_path,
    log_turn,
)
from _ui.display_names import MODEL_DISPLAY_NAMES, model_backend, model_display_name  # noqa: E402
from _ui.page_header import page_header  # noqa: E402
from _ui.tool_card import render_tool_call  # noqa: E402

from core.agents.compute_advisor import ComputeAdvisor  # noqa: E402

DEMO_TRANSCRIPT_PATH = _BUILD1 / "_ui" / "examples" / "advisor_transcript.json"

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Ask the Advisor")

page_header(
    what="A chat agent that sizes AI workloads using real tool calls — never guesses a number.",
    why="Every claim links to a function in the cost engine. The tool-call panel shows exactly what was called.",
    start="Click a starter question below, or type your own.",
    numbers="Each turn shows £ cost, latency, and token breakdown. All of it is logged to the Cost tracker page.",
)

# ---------------------------------------------------------------------------
# Model picker + key detection
# ---------------------------------------------------------------------------
def _secret(name: str) -> str:
    """Look up a secret, falling back to env var. Safe when no secrets.toml exists."""
    val = os.getenv(name) or ""
    if val:
        return val
    try:
        return str(st.secrets[name])
    except Exception:  # noqa: BLE001 — Streamlit raises various types if no secrets.toml
        return ""

openai_key = _secret("OPENAI_API_KEY")
anthropic_key = _secret("ANTHROPIC_API_KEY")

has_any_key = bool(openai_key or anthropic_key)

picker_col, action_col = st.columns([3, 1])
with picker_col:
    model_ids = list(MODEL_DISPLAY_NAMES.keys())  # gpt-4o-mini, claude-haiku-5-5
    labels = [MODEL_DISPLAY_NAMES[m] for m in model_ids]
    # Default to gpt-4o-mini (index 0) — owner-confirmed 2026-11-08.
    model_choice = st.selectbox(
        "Model",
        options=labels,
        index=0,
        key="advisor_model_label",
        help=(
            "Switch backends mid-chat — same tools, same engine, different brain. "
            f"gpt-4o-mini: ${PRICE_PER_MTOK['gpt-4o-mini']['input']}/M in · "
            f"${PRICE_PER_MTOK['gpt-4o-mini']['output']}/M out. "
            f"Claude Haiku 5.5: ${PRICE_PER_MTOK['claude-haiku-5-5']['input']}/M in · "
            f"${PRICE_PER_MTOK['claude-haiku-5-5']['output']}/M out."
        ),
    )
    picked_model = model_ids[labels.index(model_choice)]
with action_col:
    st.write("")  # spacer
    if st.button("🧹 Clear chat", width="stretch"):
        st.session_state.pop("history", None)
        st.session_state.pop("session_gbp", None)
        st.rerun()

# ---------------------------------------------------------------------------
# No-key path: show a canned transcript so visitors see how the agent works
# ---------------------------------------------------------------------------
if not has_any_key:
    st.warning(
        "No `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` set. Showing a **demo transcript** so you can "
        "see how the Advisor works. Set either key in Streamlit secrets to chat live."
    )
    with open(DEMO_TRANSCRIPT_PATH) as f:
        demo = json.load(f)
    st.caption(f"Demo model: **{model_display_name(demo['model'])}**")
    for turn in demo["turns"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("tool_calls"):
                st.markdown("**Tool calls:**")
                for tc in turn["tool_calls"]:
                    render_tool_call(
                        tool_name=tc["name"],
                        tool_input=tc["input"],
                        tool_result=tc["result"],
                    )
            if "cost_gbp" in turn:
                st.caption(
                    f"This turn: £{turn['cost_gbp']:.5f} · "
                    f"{turn['latency_ms']/1000:.1f} s · "
                    f"{turn['usage']['prompt_tokens']} in / "
                    f"{turn['usage']['completion_tokens']} out / "
                    f"{turn['usage']['prompt_tokens_details']['cached_tokens']} cached"
                )
    st.info(
        "➡️ See the **Cost tracker** page in the sidebar for a running breakdown of "
        "input/output/cached tokens, latency and £ once you start chatting."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Live chat — key available
# ---------------------------------------------------------------------------
backend = model_backend(picked_model)
if backend == "openai" and not openai_key:
    st.error(
        "You picked **gpt-4o-mini** but there's no `OPENAI_API_KEY` in secrets. "
        "Add it to Streamlit Cloud secrets or pick Claude Haiku 5.5."
    )
    st.stop()
if backend == "anthropic" and not anthropic_key:
    st.error(
        "You picked **Claude Haiku 5.5** but there's no `ANTHROPIC_API_KEY` in secrets. "
        "Add it to Streamlit Cloud secrets or pick gpt-4o-mini."
    )
    st.stop()

active_key = openai_key if backend == "openai" else anthropic_key

# Rebuild the advisor when the model picker changes.
if (
    "advisor" not in st.session_state
    or st.session_state.get("advisor_model") != picked_model
):
    st.session_state.advisor = ComputeAdvisor(
        api_key=active_key,
        model=picked_model,
        backend=backend,
    )
    st.session_state.advisor_model = picked_model

if "history" not in st.session_state:
    st.session_state.history = []
if "session_gbp" not in st.session_state:
    st.session_state.session_gbp = 0.0

# ---------------------------------------------------------------------------
# Starter-question chips
# ---------------------------------------------------------------------------
if not st.session_state.history:
    st.markdown("**Starter questions — click one:**")
    STARTERS = [
        "Serve Llama-3-70B at 50 rps for 3 years — rent, buy, or API?",
        "Does Llama-3-8B fit on an L40S at batch 32?",
        "At what utilisation does owning H100s beat reserved cloud?",
        "What's the implied gross margin on the Together AI API for Llama-3-70B vs owning H100s?",
    ]
    chip_cols = st.columns(len(STARTERS))
    for col, q in zip(chip_cols, STARTERS, strict=True):
        with col:
            if st.button(q, key=f"starter_{hash(q)}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()

# ---------------------------------------------------------------------------
# History render
# ---------------------------------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("tool_calls"):
            st.markdown("**Tool calls:**")
            for tc in turn["tool_calls"]:
                render_tool_call(
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    tool_result=tc["result"],
                )
        if "cost_gbp" in turn:
            st.caption(
                f"This turn: £{turn['cost_gbp']:.5f} · "
                f"{turn['latency_ms']/1000:.1f} s · "
                f"{turn['input_tok']} in / {turn['output_tok']} out / "
                f"{turn['cached_tok']} cached ({turn.get('cache_hit_pct', 0.0):.0f}% cache hit)"
            )

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_question", None)
question = pending or st.chat_input("Ask a sizing or build-vs-rent question.")

if question:
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model_display_name(picked_model)}…"):
            t0 = time.perf_counter()
            try:
                resp = st.session_state.advisor.ask(question)
            except Exception as e:  # noqa: BLE001
                st.error(f"Advisor failed: {type(e).__name__}: {e}")
                st.stop()
            latency_ms = (time.perf_counter() - t0) * 1000.0

        st.markdown(resp.text)

        tool_dicts = [
            {"name": tc.name, "input": tc.params, "result": tc.result, "error": tc.error}
            for tc in resp.tool_calls
        ]
        if tool_dicts:
            st.markdown("**Tool calls:**")
            for tc in tool_dicts:
                render_tool_call(
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    tool_result=tc["result"],
                )

        # Cost + log
        try:
            cost = cost_for_turn(picked_model, resp.usage or {})
            st.session_state.session_gbp += cost.gbp
            cost_line = (
                f"This turn: **£{cost.gbp:.5f}** · "
                f"{latency_ms/1000:.1f} s · "
                f"{cost.input_tok} in / {cost.output_tok} out / "
                f"{cost.cached_tok} cached ({cost.cache_hit_pct:.0f}% cache hit)  \n"
                f"Session total: **£{st.session_state.session_gbp:.4f}**"
            )
            st.caption(cost_line)

            log_turn(
                default_db_path(),
                cost=cost,
                latency_ms=latency_ms,
                question=question,
                answer_md=resp.text,
            )
        except Exception as e:  # noqa: BLE001
            st.caption(f"(cost accounting skipped: {type(e).__name__}: {e})")
            cost = None

    st.session_state.history.append({
        "role": "assistant",
        "text": resp.text,
        "tool_calls": tool_dicts,
        "cost_gbp": cost.gbp if cost else 0.0,
        "latency_ms": latency_ms,
        "input_tok": cost.input_tok if cost else 0,
        "output_tok": cost.output_tok if cost else 0,
        "cached_tok": cost.cached_tok if cost else 0,
        "cache_hit_pct": cost.cache_hit_pct if cost else 0.0,
    })
