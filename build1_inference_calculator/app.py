"""Build 1 landing page.

A non-technical visitor (recruiter, GTM lead, strategy hire) should land here
and in 30 seconds understand:
    1. What this is  — a build-vs-rent calculator for serving open-weight LLMs
    2. Why it matters — same workload, five costing paths, one £/M-token ruler
    3. Where to start — four big buttons, Advisor first
    4. One worked result above the fold, no clicks required

Hero numbers below are hard-coded from a one-off run of ``compare_options`` for
the default Nscale scenario (Llama-3-70B on 8× H100 SXM 80GB, 60% busy,
$2.49/GPU-hour neocloud). Locking the demo result means the landing page
renders instantly, never drifts on an engine change, and can't fail if the
cost engine throws.
"""

from __future__ import annotations

# Streamlit Cloud entry-point shim: ensure the repo root is on sys.path so
# ``from core.X import ...`` resolves even if the editable install from
# requirements.txt (``-e .``) silently fails. Harmless locally.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BUILD1 = Path(__file__).resolve().parent
if str(_BUILD1) not in sys.path:
    sys.path.insert(0, str(_BUILD1))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from _ui.page_header import page_header  # noqa: E402
from _ui.provenance import badge, legend_expander  # noqa: E402
from _ui.sidebar import render_sidebar  # noqa: E402

# ---------------------------------------------------------------------------
# App-wide page config — set once here, removed from pages/ in M9.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Compute Economics Workbench",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_sidebar()

# ---------------------------------------------------------------------------
# Block 1 — hero sentence
# ---------------------------------------------------------------------------
st.title("Should you rent, buy, or API an H100 for Llama-3-70B?")
st.markdown(
    "#### This workbench answers in **£ per million tokens** — and shows its working."
)

page_header(
    what="A calculator + two AI agents for AI-infrastructure unit economics.",
    why=(
        "Same workload costed five ways on one ruler — owning, reserved, "
        "on-demand, neocloud, and per-token API."
    ),
    start="Pick a jump below. The Advisor is the fastest demo.",
    numbers=(
        "Every figure is tagged 🟢 measured, 🟡 modelled, 🟠 simulated, or "
        "⚪ assumption — hover any badge for its source."
    ),
)

# ---------------------------------------------------------------------------
# Block 2 — worked example
# ---------------------------------------------------------------------------
st.markdown("## One worked example")
st.caption(
    "Llama-3-70B on 8× H100 SXM 80GB at 60% busy, neocloud at $2.49/GPU-hour, "
    "3-year horizon. Same workload, five options, one ruler. Numbers below are "
    "frozen outputs of the TCO engine — adjust inputs on the "
    "**What does it cost** page."
)

# Hard-coded result of compare_options(...) for the Nscale scenario — see
# module docstring. Re-run once a quarter and update if the engine drifts.
_HERO_RESULTS: list[dict[str, float | str]] = [
    {"label": "Owning",            "usd_per_m_tokens": 3.12, "total_3y_usd": 84_072},
    {"label": "Reserved (3-year)", "usd_per_m_tokens": 2.73, "total_3y_usd": 73_584},
    {"label": "On-demand cloud",   "usd_per_m_tokens": 4.39, "total_3y_usd": 118_260},
    {"label": "Neocloud",          "usd_per_m_tokens": 2.43, "total_3y_usd": 65_437},
    {"label": "Per-token API",     "usd_per_m_tokens": 2.64, "total_3y_usd": 71_175},
]
_HERO_SCENARIO_LABEL = "Llama-3-70B · 8× H100 · 60% busy · 3-year horizon"

_cheapest = min(_HERO_RESULTS, key=lambda r: r["usd_per_m_tokens"])
_dearest = max(_HERO_RESULTS, key=lambda r: r["usd_per_m_tokens"])
_multiple = _dearest["usd_per_m_tokens"] / _cheapest["usd_per_m_tokens"]

chart_col, verdict_col = st.columns([3, 2])

with chart_col:
    bar_colors = [
        "#2E7D32" if r["label"] == _cheapest["label"] else "#78909C"
        for r in _HERO_RESULTS
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[r["label"] for r in _HERO_RESULTS],
                y=[r["usd_per_m_tokens"] for r in _HERO_RESULTS],
                text=[f"${r['usd_per_m_tokens']:.2f}" for r in _HERO_RESULTS],
                textposition="outside",
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "$%{y:.2f} per million output tokens<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"$ per million output tokens — {_HERO_SCENARIO_LABEL}",
        yaxis_title="USD per million output tokens",
        yaxis_range=[0, max(r["usd_per_m_tokens"] for r in _HERO_RESULTS) * 1.2],
        height=380,
        showlegend=False,
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )
    st.plotly_chart(fig, width="stretch")

with verdict_col:
    st.markdown("### Verdict")
    st.markdown(
        f"**{_cheapest['label']} wins at ${_cheapest['usd_per_m_tokens']:.2f} / M "
        f"output tokens** — **{_multiple:.1f}×** cheaper than "
        f"{_dearest['label'].lower()}."
    )
    st.markdown(
        "- Total 3-year spend: "
        f"**${int(_cheapest['total_3y_usd']):,}**\n"
        f"- {badge('modelled')} TCO math, {badge('assumption')} hourly rates\n"
        "- Tune any input on the **What does it cost** page"
    )
    st.caption(
        "A 60% → 80% busy swing can flip the recommendation. "
        "That's the point of the Advisor: ask it and see."
    )

# ---------------------------------------------------------------------------
# Block 3 — four CTAs
# ---------------------------------------------------------------------------
st.markdown("## Start here")

cta_cols = st.columns(4)

_CTAS = [
    {
        "icon": "🤖",
        "label": "Ask the Advisor",
        "sub": "Natural-language sizing questions, answered with real tool calls.",
        "page": "pages/1_Ask_the_Advisor.py",
    },
    {
        "icon": "💷",
        "label": "What does it cost",
        "sub": "Five options on one $/M-token ruler, with break-even + demand-stress.",
        "page": "pages/3_What_does_it_cost.py",
    },
    {
        "icon": "📏",
        "label": "Will it fit",
        "sub": "GPU memory check for any model + batch size.",
        "page": "pages/4_Will_it_fit.py",
    },
    {
        "icon": "🎯",
        "label": "How accurate",
        "sub": "Predicted vs measured tokens/s, cross-checked against MLPerf.",
        "page": "pages/5_How_accurate.py",
    },
]

for col, cta in zip(cta_cols, _CTAS, strict=True):
    with col, st.container(border=True):
        st.markdown(f"### {cta['icon']} {cta['label']}")
        st.caption(cta["sub"])
        try:
            st.page_link(cta["page"], label="Go →")
        except Exception:
            # Fallback if page_link isn't available (older Streamlit).
            st.markdown(f"→ *{cta['label']}* (sidebar)")

# ---------------------------------------------------------------------------
# Block 4 — methodology + status, collapsed
# ---------------------------------------------------------------------------
with st.expander("Methodology & data status"):
    st.markdown(
        """
Every number on this site is tagged with how it was produced:

- 🟢 **Measured** — from an actual hardware run. Kaggle T4 for small-model
  throughput; MLPerf-submitter numbers for large-model throughput.
- 🟡 **Modelled** — physics or TCO formulas. Roofline for throughput,
  straight-line depreciation + opex for cost.
- 🟠 **Simulated** — replay of public cluster traces (Alibaba, Azure).
  Used by Build 3 (not yet shipped).
- ⚪ **Assumption** — a reasonable default the author chose. Capex per GPU,
  hourly rates, power cost — tune them on the TCO page.

**Data quality today**

| Page | Powered by | Current state |
|---|---|---|
| Will it fit | 17-GPU catalog, Turing → Rubin | Datasheet-high confidence for Turing–Hopper; medium for Blackwell; low for Rubin (announcement-only) |
| How accurate | Roofline model, calibrated against Kaggle T4 | Measured T4 run lands once the first notebook is executed |
| What does it cost | TCO engine + weekly GPU price log | Prices accumulate as the Monday cron fires |
| Ask the Advisor | Dual-backend tool-calling agent (OpenAI gpt-4o-mini + Claude Haiku 5.5) | Live |
| Market prices | Fact extractor with verbatim-quote verification | 40 facts from an 8-document demo corpus |
        """
    )
    legend_expander()
