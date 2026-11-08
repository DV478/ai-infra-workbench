"""TCO page — 'What does it cost per million tokens?'.

The headline page. A non-technical reader lands here, hits one preset, and
sees the recommendation card before touching a slider. Tuning widgets live
in an expander — they exist, they're accessible, they don't dominate.

Flow:
    1. Preset chips (default = Nscale 70B/H100).
    2. Headline recommendation card (cheapest option, winning multiple).
    3. Two charts side by side: $/M tokens bar + demand-stress bar.
    4. Break-even sentences in plain English.
    5. Sensitivity bar — how busy_fraction swings the cost.
    6. 'Tune the inputs' expander (all 20+ widgets).
    7. Memo download next to the headline.
"""

from __future__ import annotations

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
from _ui.explain import break_even_sentence, caption, tooltip  # noqa: E402
from _ui.page_header import page_header  # noqa: E402
from _ui.presets import SCENARIOS, preset_chips, scenario_default  # noqa: E402
from _ui.provenance import badge  # noqa: E402

from core.cost_engine.kv_cache import LLAMA3_8B, LLAMA3_70B, QWEN2_5_3B  # noqa: E402
from core.cost_engine.tco import (  # noqa: E402
    OwnedEconomics,
    compare_options,
    demand_stress,
)
from core.gpu_specs import load_catalog  # noqa: E402
from core.memo import render_memo  # noqa: E402

st.title("What does it cost per million tokens?")

page_header(
    what="Same workload costed five ways — owning, reserved, on-demand, neocloud, per-token API.",
    why="One $/M-token ruler lets you spot the cheapest option *and* the risk of being wrong about demand.",
    start="Pick a preset below. The default is the Nscale/Nebius 70B scenario.",
    numbers="All figures are 3-year blended. Hover any badge for its source.",
)

preset_chips("tco", SCENARIOS)

# ---------------------------------------------------------------------------
# Lookups + preset defaults
# ---------------------------------------------------------------------------
catalog = load_catalog()
MODELS = {
    "Qwen2.5-3B": (QWEN2_5_3B, 3.0),
    "Llama-3-8B": (LLAMA3_8B, 8.0),
    "Llama-3-70B": (LLAMA3_70B, 70.0),
}

gpu_sorted = sorted(catalog.gpus, key=lambda g: (g.vendor, g.release_year, g.name))
GPU_OPTIONS = {g.name: g.id for g in gpu_sorted}
GPU_NAMES = list(GPU_OPTIONS.keys())

_default_gpu_id = scenario_default("tco", "gpu_id", "h100_sxm5_80gb")
try:
    _default_gpu_name = catalog.by_id(_default_gpu_id).name
except KeyError:
    _default_gpu_name = catalog.by_id("h100_sxm5_80gb").name

_default_model_name = scenario_default("tco", "model_name", "Llama-3-70B")
if _default_model_name not in MODELS:
    _default_model_name = "Llama-3-70B"

# ---------------------------------------------------------------------------
# All controls live in one expander. Default values come from the preset.
# ---------------------------------------------------------------------------
with st.expander("Tune the inputs", expanded=False):
    st.caption("Everything feeds the comparison below. Hover any label for an explanation.")

    st.markdown("**Workload**")
    c1, c2, c3 = st.columns(3)
    with c1:
        gpu_name = st.selectbox(
            "GPU",
            options=GPU_NAMES,
            index=GPU_NAMES.index(_default_gpu_name),
            help="Accelerator the workload runs on. Pretty names only.",
        )
        gpu_id = GPU_OPTIONS[gpu_name]
    with c2:
        model_name = st.selectbox(
            "Model",
            options=list(MODELS.keys()),
            index=list(MODELS.keys()).index(_default_model_name),
            help="Open-weight model. Larger = more compute per token.",
        )
    with c3:
        busy = st.slider(
            "Busy fraction",
            0.1, 1.0,
            float(scenario_default("tco", "busy", 0.60)),
            0.05,
            help=tooltip("busy_fraction"),
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        batch = st.number_input("Batch size", 1, 256,
                                int(scenario_default("tco", "batch", 32)),
                                help=tooltip("batch_size"))
    with c2:
        in_len = st.number_input("Avg input length", 64, 32768,
                                 int(scenario_default("tco", "in_len", 512)),
                                 help=tooltip("input_length"))
    with c3:
        out_len = st.number_input("Avg output length", 32, 8192,
                                  int(scenario_default("tco", "out_len", 256)),
                                  help=tooltip("output_length"))
    with c4:
        util = st.slider("Kernel utilisation", 0.1, 1.0,
                         float(scenario_default("tco", "util", 0.65)),
                         0.05, help=tooltip("kernel_utilisation"))

    st.markdown("**Capex + opex (owned option)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        capex = st.number_input("Capex per GPU (USD)", 10000, 80000, 32000, step=1000,
                                help="One GPU + its share of the server, amortised per GPU slot.")
        pue = st.slider("PUE", 1.0, 2.0, 1.3, 0.05, help=tooltip("pue"))
    with c2:
        power_kwh = st.number_input("Power cost ($/kWh)", 0.01, 0.50, 0.09, 0.01,
                                    help=tooltip("electricity_usd_per_kwh"))
        hosting = st.number_input("Hosting ($/month)", 100, 5000, 1200, 50,
                                  help=tooltip("hosting_usd_per_month"))
    with c3:
        financing = st.slider("Financing rate %", 0.0, 20.0, 8.0, 0.5,
                              help=tooltip("financing_pct"))
        maint = st.slider("Maintenance % capex/yr", 0.0, 10.0, 3.0, 0.5,
                          help="Annual maintenance cost as a % of capex.")

    st.markdown("**Hourly rental rates (USD / GPU-hour)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        reserved_h = st.number_input("Reserved 3y", 0.5, 20.0, 2.80, 0.10,
                                     help="Hyperscaler 3-year reservation.")
    with c2:
        on_demand_h = st.number_input("On-demand", 0.5, 30.0, 4.50, 0.10,
                                      help="Pay-as-you-go hyperscaler rate.")
    with c3:
        neocloud_h = st.number_input("Neocloud (Nscale / Nebius / Lambda)",
                                     0.5, 20.0, 2.49, 0.10,
                                     help="Specialist GPU cloud — typically cheapest.")

    st.markdown("**Per-token API comparison**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        api_name = st.text_input("Provider", "Together AI")
    with c2:
        api_out = st.number_input("$/M output tokens", 0.01, 60.0, 0.88, 0.10)
    with c3:
        api_in = st.number_input("$/M input tokens", 0.0, 30.0, 0.88, 0.10)
    with c4:
        io_ratio = st.number_input(
            "Input/output ratio", 0.5, 20.0,
            float(in_len) / max(out_len, 1), 0.1,
            help="How many input tokens per output token on average.",
        )

# ---------------------------------------------------------------------------
# Compute scenario (fast — pure Python, sub-ms)
# ---------------------------------------------------------------------------
gpu = catalog.by_id(gpu_id)
arch, param_b = MODELS[model_name]

owned_econ = OwnedEconomics(
    capex_usd=float(capex),
    depreciation_years=3,
    power_draw_watts=gpu.tdp_watts or 700.0,
    power_cost_usd_per_kwh=float(power_kwh),
    pue=float(pue),
    hosting_usd_per_month=float(hosting),
    financing_rate_pct=float(financing),
    maintenance_pct_of_capex_per_year=float(maint),
)

scenario = compare_options(
    gpu,
    arch,
    param_b,
    owned_econ=owned_econ,
    reserved_hourly=float(reserved_h),
    on_demand_hourly=float(on_demand_h),
    neocloud_hourly=float(neocloud_h),
    api_name=api_name,
    api_output_price_per_m=float(api_out),
    api_input_price_per_m=float(api_in),
    io_ratio=float(io_ratio),
    batch_size=int(batch),
    input_length=int(in_len),
    output_length=int(out_len),
    utilisation_frac=float(util),
    busy_fraction=float(busy),
)

results = scenario.results
cheapest = min(results, key=lambda r: r.usd_per_m_tokens)
dearest = max(results, key=lambda r: r.usd_per_m_tokens)
multiple = dearest.usd_per_m_tokens / cheapest.usd_per_m_tokens if cheapest.usd_per_m_tokens else 0

# ---------------------------------------------------------------------------
# Hero recommendation + memo download
# ---------------------------------------------------------------------------
hero_col, memo_col = st.columns([3, 2])

with hero_col, st.container(border=True):
    st.markdown(f"### Recommendation for {model_name} on {gpu.name}")
    st.markdown(
        f"## {cheapest.label}  \n"
        f"### **${cheapest.usd_per_m_tokens:.2f} / million output tokens**"
    )
    st.markdown(
        f"**{multiple:.1f}×** cheaper than {dearest.label.lower()} at "
        f"${dearest.usd_per_m_tokens:.2f}/M. "
        f"Total 3-year spend: **${cheapest.total_usd_3y:,.0f}**."
    )
    st.caption(
        f"{badge('modelled')} TCO math · "
        f"{badge('assumption')} hourly rates · "
        f"workload ~{scenario.workload_tokens_per_year/1e9:.1f} B output tokens/year"
    )

with memo_col, st.container(border=True):
    st.markdown("### 📄 Download the memo")
    st.caption("A 2-page markdown sizing memo — deal-desk friendly.")
    audience = st.text_input(
        "For (optional)",
        placeholder="Nscale GTM / internal review",
        label_visibility="collapsed",
    )
    memo_md = render_memo(
        scenario,
        gpu_name=gpu.name,
        gpu_id=gpu.id,
        dtype="fp16",
        batch_size=int(batch),
        input_length=int(in_len),
        output_length=int(out_len),
        utilisation_frac=float(util),
        for_audience=audience,
    )
    st.download_button(
        "Download memo (.md)",
        data=memo_md,
        file_name=f"sizing_memo_{model_name.lower().replace(' ', '_')}_{gpu.id}.md",
        mime="text/markdown",
        type="primary",
        width="stretch",
    )
    st.caption("`pandoc memo.md -o memo.pdf` to convert, or Print → Save-as-PDF.")

# ---------------------------------------------------------------------------
# Two side-by-side charts
# ---------------------------------------------------------------------------
st.markdown("## Costs side by side")

df = pd.DataFrame([r.as_dict() for r in results])
cost_col, stress_col = st.columns(2)

with cost_col:
    colors = ["#2E7D32" if r.label == cheapest.label else "#78909C" for r in results]
    fig = go.Figure(data=[
        go.Bar(
            x=df["label"],
            y=df["usd_per_m_tokens"],
            text=[f"${v:.2f}" for v in df["usd_per_m_tokens"]],
            textposition="outside",
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>$%{y:.2f} per M output tokens<extra></extra>",
        )
    ])
    fig.update_layout(
        title="$ per million output tokens (3-year blended)",
        yaxis_title="USD",
        yaxis_range=[0, df["usd_per_m_tokens"].max() * 1.25],
        height=380,
        showlegend=False,
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(caption("tco_comparison"))

with stress_col:
    stress_rows = [demand_stress(r, 0.60) for r in results]
    s_df = pd.DataFrame([s.as_dict() for s in stress_rows])
    # Lower = better. Highlight cheapest by base cost; show stressed cost.
    stress_colors = [
        "#C62828" if s.label in ("Owned (capex $32,000, 3y depreciation)",
                                  "Reserved 3y cloud", "Neocloud (Lambda/Nebius/Nscale)")
        else "#78909C" for s in stress_rows
    ]
    fig = go.Figure(data=[
        go.Bar(
            x=s_df["label"],
            y=s_df["stressed_usd_per_m"],
            text=[f"${v:.2f}" for v in s_df["stressed_usd_per_m"]],
            textposition="outside",
            marker_color=stress_colors,
            hovertemplate="<b>%{x}</b><br>$%{y:.2f} per M under 40% demand miss<extra></extra>",
        )
    ])
    fig.update_layout(
        title="Under a 40% demand miss",
        yaxis_title="USD per M output tokens",
        yaxis_range=[0, s_df["stressed_usd_per_m"].max() * 1.25],
        height=380,
        showlegend=False,
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(caption("demand_stress_chart"))

# ---------------------------------------------------------------------------
# Break-even (English sentences)
# ---------------------------------------------------------------------------
st.markdown("## Break-even")
st.caption(caption("break_even_header"))
for be in scenario.break_evens:
    st.markdown(f"- {break_even_sentence(be)}")
    if be.note:
        st.caption(f"  _{be.note}_")

# ---------------------------------------------------------------------------
# Busy-fraction sensitivity
# ---------------------------------------------------------------------------
st.markdown("## Does the recommendation survive demand changes?")
st.caption(
    "Same inputs, varying the busy-fraction. A 60% → 80% swing often flips the "
    "winner — the Advisor can walk through this for a given workload."
)

busy_points = [0.30, 0.45, 0.60, 0.75, 0.90]
cheapest_series: list[tuple[str, float]] = []
for b in busy_points:
    sweep = compare_options(
        gpu,
        arch,
        param_b,
        owned_econ=owned_econ,
        reserved_hourly=float(reserved_h),
        on_demand_hourly=float(on_demand_h),
        neocloud_hourly=float(neocloud_h),
        api_name=api_name,
        api_output_price_per_m=float(api_out),
        api_input_price_per_m=float(api_in),
        io_ratio=float(io_ratio),
        batch_size=int(batch),
        input_length=int(in_len),
        output_length=int(out_len),
        utilisation_frac=float(util),
        busy_fraction=b,
    )
    winner = min(sweep.results, key=lambda r: r.usd_per_m_tokens)
    cheapest_series.append((winner.label, winner.usd_per_m_tokens))

sens_df = pd.DataFrame({
    "Busy fraction": [f"{int(b*100)}%" for b in busy_points],
    "Winner": [w for w, _ in cheapest_series],
    "$/M output tokens": [c for _, c in cheapest_series],
})
st.dataframe(sens_df, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Headline table (reference) + per-option notes
# ---------------------------------------------------------------------------
with st.expander("Full headline table + per-option notes"):
    show_cols = ["label", "usd_per_m_tokens", "total_usd_3y", "total_output_tokens_3y"]
    pretty = {
        "label": "Option",
        "usd_per_m_tokens": "$/M output tokens",
        "total_usd_3y": "Total 3-year spend ($)",
        "total_output_tokens_3y": "Output tokens over 3 years",
    }
    st.dataframe(df[show_cols].rename(columns=pretty), hide_index=True, width="stretch")
    st.markdown(f"**Workload tokens/year:** {scenario.workload_tokens_per_year:,.0f}")
    st.markdown(f"**Busy fraction:** {scenario.busy_fraction:.2f}")
    for r in results:
        st.markdown(f"- **{r.label}** — {r.notes}")
