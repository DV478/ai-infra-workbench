"""Cost & token tracker — the observability page.

Reads every Advisor turn out of ``data/advisor_turns.duckdb`` and renders:
- Hero aggregates (turns, cumulative £, avg £, cache-hit %, avg latency)
- Per-model side-by-side comparison when ≥2 models are logged
- Pricing-lever explainer (what each column means in plain English)
- Turn-by-turn table with Q&A expanders
- Current per-model price card
- Reset-log button (gated behind an env-var pin on deployed instances)

Primary audience: the owner (learning how LLM pricing works).
Secondary: a recruiter (credibility signal — 'they instrumented their own LLM spend').
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD1 = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _BUILD1):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from _ui.cost_counter import (  # noqa: E402
    PRICE_CARD_LAST_CONFIRMED,
    PRICE_PER_MTOK,
    USD_TO_GBP,
    aggregates,
    aggregates_by_model,
    clear_log,
    default_db_path,
    load_turns,
)
from _ui.display_names import model_display_name  # noqa: E402
from _ui.explain import caption, tooltip  # noqa: E402
from _ui.page_header import page_header  # noqa: E402

st.title("Cost & token tracker")

page_header(
    what="Every Advisor chat turn, broken down by tokens and £.",
    why="LLM pricing has three levers — input, output, and cached tokens. Watch them move live.",
    start="Chat on the Advisor page; each turn lands here with a full breakdown.",
    numbers="£ per turn, tokens in/out/cached, cache-hit %, and latency in ms.",
)

# ---------------------------------------------------------------------------
# Load turns
# ---------------------------------------------------------------------------
db_path = default_db_path()
turns = load_turns(db_path)

if not turns:
    st.info(
        "No chat turns logged yet — head to the **Ask the Advisor** page and ask "
        "a question. Every turn you run will land here with a token + £ breakdown."
    )
    st.markdown("### What this page will show (preview)")
    st.markdown(
        "- **Hero row** — total turns, cumulative £ cost, avg £ per turn, cache-hit %, avg latency.\n"
        "- **Per-model comparison** — gpt-4o-mini vs Claude Haiku 5.5 side by side, same questions.\n"
        "- **Per-turn table** — one row per message with input/output/cached tokens + £.\n"
        "- **Current price card** — Nov 2026 $/M-token rates, cached-input discount, pound equivalent."
    )

    # Still show the price card so visitors can see pricing without chatting.
    with st.expander("Current per-model price card (Nov 2026)", expanded=True):
        st.caption(caption("pricing_levers"))
        price_rows = []
        for mid, prices in PRICE_PER_MTOK.items():
            price_rows.append({
                "Model": model_display_name(mid),
                "Input ($/M)": prices["input"],
                "Cached input ($/M)": prices["cached_input"],
                "Output ($/M)": prices["output"],
                "Input (£/M)": prices["input"] * USD_TO_GBP,
                "Output (£/M)": prices["output"] * USD_TO_GBP,
            })
        st.dataframe(
            pd.DataFrame(price_rows),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Last confirmed: {PRICE_CARD_LAST_CONFIRMED}. USD→GBP: {USD_TO_GBP}. "
            "Cached tokens are usually repeated prompt prefixes the provider "
            "serves from its own cache."
        )
    st.stop()

# ---------------------------------------------------------------------------
# Hero row aggregates
# ---------------------------------------------------------------------------
agg = aggregates(turns)
hero = st.columns(5)
hero[0].metric(
    "Turns logged",
    f"{agg.turns:,}",
    help="Every completed chat turn across all sessions.",
)
hero[1].metric(
    "Total cost",
    f"£{agg.total_gbp:.4f}",
    help="Cumulative £ spend across every logged turn.",
)
hero[2].metric(
    "Avg per turn",
    f"£{agg.avg_gbp:.5f}",
    help="Mean £ per turn — a baseline for 'what a Q&A costs on the demo'.",
)
hero[3].metric(
    "Cache hit",
    f"{agg.cache_hit_pct:.0f} %",
    help=tooltip("cached_tokens"),
)
hero[4].metric(
    "Avg latency",
    f"{agg.avg_latency_ms:.0f} ms",
    help="Mean wall-clock time per turn, from question-sent to answer-received.",
)
st.caption(caption("cost_tracker_hero"))

# ---------------------------------------------------------------------------
# Pricing-lever explainer — four tiles
# ---------------------------------------------------------------------------
st.markdown("## The three pricing levers")
tiles = st.columns(4)
with tiles[0], st.container(border=True):
    st.markdown("### 📥 Input tokens")
    st.caption(tooltip("input_tokens"))
    total_in = sum(t["input_tok"] for t in turns)
    st.metric("Logged", f"{total_in:,}")
with tiles[1], st.container(border=True):
    st.markdown("### 📤 Output tokens")
    st.caption(tooltip("output_tokens"))
    total_out = sum(t["output_tok"] for t in turns)
    st.metric("Logged", f"{total_out:,}")
with tiles[2], st.container(border=True):
    st.markdown("### 🏷️ Cached tokens")
    st.caption(tooltip("cached_tokens"))
    total_cached = sum(t["cached_tok"] for t in turns)
    st.metric("Logged", f"{total_cached:,}")
with tiles[3], st.container(border=True):
    st.markdown("### ⏱️ Latency")
    st.caption("Wall-clock time per turn — the responsiveness lever.")
    st.metric("Avg", f"{agg.avg_latency_ms/1000:.1f} s")

# ---------------------------------------------------------------------------
# Per-model comparison (only interesting with >1 model)
# ---------------------------------------------------------------------------
by_model = aggregates_by_model(turns)
if len(by_model) >= 2:
    st.markdown("## Same questions, different brains")
    st.caption(
        "Both models answered the same questions during this session. Compare their "
        "£ per turn and latency side by side."
    )
    comp_df = pd.DataFrame([
        {
            "Model": model_display_name(mid),
            "Turns": v.turns,
            "Avg £ per turn": v.avg_gbp,
            "Total £": v.total_gbp,
            "Cache hit %": v.cache_hit_pct,
            "Avg latency (ms)": v.avg_latency_ms,
        }
        for mid, v in by_model.items()
    ])
    c1, c2 = st.columns(2)
    with c1:
        fig_cost = go.Figure(data=[
            go.Bar(
                x=comp_df["Model"],
                y=comp_df["Avg £ per turn"],
                text=[f"£{v:.5f}" for v in comp_df["Avg £ per turn"]],
                textposition="outside",
                marker_color=["#2E7D32" if v == comp_df["Avg £ per turn"].min() else "#78909C"
                              for v in comp_df["Avg £ per turn"]],
            )
        ])
        fig_cost.update_layout(
            title="Avg £ per turn by model",
            yaxis_title="£",
            height=330,
            showlegend=False,
            yaxis_range=[0, comp_df["Avg £ per turn"].max() * 1.25],
            margin={"t": 60, "b": 40, "l": 60, "r": 20},
        )
        st.plotly_chart(fig_cost, width="stretch")
    with c2:
        fig_lat = go.Figure(data=[
            go.Bar(
                x=comp_df["Model"],
                y=comp_df["Avg latency (ms)"],
                text=[f"{v:.0f} ms" for v in comp_df["Avg latency (ms)"]],
                textposition="outside",
                marker_color="#1E88E5",
            )
        ])
        fig_lat.update_layout(
            title="Avg latency by model",
            yaxis_title="ms",
            height=330,
            showlegend=False,
            margin={"t": 60, "b": 40, "l": 60, "r": 20},
        )
        st.plotly_chart(fig_lat, width="stretch")
    st.dataframe(comp_df, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Turn-by-turn table
# ---------------------------------------------------------------------------
st.markdown("## Turn-by-turn log")

turn_rows = [
    {
        "Time": t["ts"],
        "Model": model_display_name(t["model"]),
        "Input tok": t["input_tok"],
        "Output tok": t["output_tok"],
        "Cached tok": t["cached_tok"],
        "Cache hit %": (t["cached_tok"] / t["input_tok"] * 100) if t["input_tok"] else 0.0,
        "Latency (ms)": t["latency_ms"],
        "£ cost": t["gbp"],
        "Question": t["question"][:80] + ("…" if len(t["question"]) > 80 else ""),
    }
    for t in turns
]
turn_df = pd.DataFrame(turn_rows)

st.dataframe(
    turn_df,
    hide_index=True,
    width="stretch",
    column_config={
        "£ cost": st.column_config.NumberColumn(format="£%.5f"),
        "Latency (ms)": st.column_config.NumberColumn(format="%.0f"),
        "Cache hit %": st.column_config.NumberColumn(format="%.0f%%"),
    },
)

with st.expander("Browse individual Q&As"):
    for t in turns[:20]:  # limit to 20 most recent
        with st.container(border=True):
            st.caption(f"{t['ts']} · {model_display_name(t['model'])} · £{t['gbp']:.5f}")
            st.markdown(f"**Q:** {t['question']}")
            st.markdown(f"**A:** {t['answer_md']}")

# ---------------------------------------------------------------------------
# Price card (confirmed source of truth)
# ---------------------------------------------------------------------------
with st.expander("Current per-model price card", expanded=False):
    price_rows = []
    for mid, prices in PRICE_PER_MTOK.items():
        price_rows.append({
            "Model": model_display_name(mid),
            "Input ($/M)": prices["input"],
            "Cached input ($/M)": prices["cached_input"],
            "Output ($/M)": prices["output"],
            "Input (£/M)": prices["input"] * USD_TO_GBP,
            "Output (£/M)": prices["output"] * USD_TO_GBP,
        })
    st.dataframe(pd.DataFrame(price_rows), hide_index=True, width="stretch")
    st.caption(
        f"Last confirmed: {PRICE_CARD_LAST_CONFIRMED}. USD→GBP: {USD_TO_GBP}. "
        "Refresh quarterly."
    )

# ---------------------------------------------------------------------------
# Admin: reset
# ---------------------------------------------------------------------------
with st.expander("Admin"):
    st.caption(
        "Clearing the turn log deletes `data/advisor_turns.duckdb`. "
        "Gated behind the `COST_TRACKER_RESET_PIN` environment variable on deployed instances."
    )
    pin_env = os.getenv("COST_TRACKER_RESET_PIN", "")
    if pin_env:
        pin = st.text_input("Reset PIN", type="password")
        if st.button("Clear turn log", type="secondary"):
            if pin == pin_env:
                clear_log(db_path)
                st.success("Log cleared.")
                st.rerun()
            else:
                st.error("Wrong PIN.")
    else:
        if st.button("Clear turn log (no PIN set — local dev only)", type="secondary"):
            clear_log(db_path)
            st.success("Log cleared.")
            st.rerun()
