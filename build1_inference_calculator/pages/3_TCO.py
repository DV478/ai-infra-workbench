"""TCO page — 3-year own vs rent vs API comparison."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from core.cost_engine.kv_cache import LLAMA3_8B, LLAMA3_70B, QWEN2_5_3B  # noqa: E402
from core.cost_engine.tco import OwnedEconomics, compare_options, demand_stress  # noqa: E402
from core.gpu_specs import load_catalog  # noqa: E402
from core.memo import render_memo  # noqa: E402

st.title("TCO — build vs rent vs API")

st.caption(
    "Three-year total cost of ownership across five options on one denominator: "
    "$ per million output tokens. All compute figures are **modelled**; "
    "hourly prices should come from `data/prices/prices.parquet` once the weekly cron has run."
)

catalog = load_catalog()
MODELS = {
    "Qwen2.5-3B": (QWEN2_5_3B, 3.0),
    "Llama-3-8B": (LLAMA3_8B, 8.0),
    "Llama-3-70B": (LLAMA3_70B, 70.0),
}

# ---------------------------------------------------------------------------
st.header("Workload")

col1, col2, col3 = st.columns(3)
with col1:
    gpu_id = st.selectbox("GPU", options=catalog.ids(), index=catalog.ids().index("h100_sxm5_80gb"))
with col2:
    model_name = st.selectbox("Model", options=list(MODELS.keys()), index=1)
with col3:
    busy = st.slider("Busy fraction (fraction of billed hours decoding)", 0.1, 1.0, 0.60, 0.05)

col1, col2, col3, col4 = st.columns(4)
with col1:
    batch = st.number_input("Batch size", 1, 256, 32)
with col2:
    in_len = st.number_input("Avg input length", 64, 32768, 512)
with col3:
    out_len = st.number_input("Avg output length", 32, 8192, 256)
with col4:
    util = st.slider("Kernel utilisation %", 0.1, 1.0, 0.65, 0.05)

gpu = catalog.by_id(gpu_id)
arch, param_b = MODELS[model_name]

# ---------------------------------------------------------------------------
st.header("Capex + opex (owned)")

col1, col2, col3 = st.columns(3)
with col1:
    capex = st.number_input("Capex per GPU (USD)", 10000, 80000, 32000, step=1000)
    pue = st.slider("PUE", 1.0, 2.0, 1.4, 0.05)
with col2:
    power_kwh = st.number_input("Power cost ($/kWh)", 0.01, 0.50, 0.10, 0.01)
    hosting = st.number_input("Hosting ($/month)", 100, 5000, 1200, 50)
with col3:
    financing = st.slider("Financing rate %", 0.0, 20.0, 8.0, 0.5)
    maint = st.slider("Maintenance % capex/yr", 0.0, 10.0, 3.0, 0.5)

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

# ---------------------------------------------------------------------------
st.header("Hourly rates")

col1, col2, col3 = st.columns(3)
with col1:
    reserved_h = st.number_input("Reserved 3y $/hr", 0.5, 20.0, 2.50, 0.10)
with col2:
    on_demand_h = st.number_input("On-demand $/hr", 0.5, 30.0, 4.50, 0.10)
with col3:
    neocloud_h = st.number_input("Neocloud $/hr (Lambda/Nebius/Nscale)", 0.5, 20.0, 2.00, 0.10)

st.header("API pricing")
col1, col2, col3, col4 = st.columns(4)
with col1:
    api_name = st.text_input("Provider", "Together AI")
with col2:
    api_out = st.number_input("$/M output tokens", 0.01, 60.0, 2.00, 0.10)
with col3:
    api_in = st.number_input("$/M input tokens", 0.0, 30.0, 1.00, 0.10)
with col4:
    io_ratio = st.number_input("Input/output ratio", 0.5, 20.0, float(in_len) / max(out_len, 1), 0.1)

# ---------------------------------------------------------------------------
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

df = pd.DataFrame([r.as_dict() for r in scenario.results])

# ---------------------------------------------------------------------------
st.header("Headline table")
st.dataframe(df, hide_index=True, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(
        data=[go.Bar(x=df["label"], y=df["usd_per_m_tokens"], text=df["usd_per_m_tokens"].round(2), textposition="auto")]
    )
    fig.update_layout(title="$ per million output tokens (3y blended)", yaxis_title="USD", height=400)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = go.Figure(
        data=[go.Bar(x=df["label"], y=df["total_usd_3y"], text=df["total_usd_3y"].round(0), textposition="auto")]
    )
    fig.update_layout(title="Total 3-year spend (USD)", yaxis_title="USD", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
st.header("Break-even")
for be in scenario.break_evens:
    st.markdown(
        f"- **{be.option_a} vs {be.option_b}** — {be.pivot_variable} = **{be.pivot_value:.3f}**. "
        f"{be.note}"
    )

# ---------------------------------------------------------------------------
st.header("Demand stress — 40% miss")
stress_mult = st.slider("Demand multiplier (1.0 = forecast)", 0.2, 1.5, 0.60, 0.05)
stress_rows = [demand_stress(r, stress_mult) for r in scenario.results]
sdf = pd.DataFrame([s.as_dict() for s in stress_rows])
st.dataframe(sdf, hide_index=True, use_container_width=True)

st.caption(
    "Under a demand miss, owned/reserved/neocloud pay full capacity for less output "
    "(stranded capex). On-demand and API scale with demand. This is the asymmetry "
    "that makes utilisation forecasting the single most important input into TCO."
)

# ---------------------------------------------------------------------------
with st.expander("Scenario details"):
    st.markdown(f"**Workload tokens/year:** {scenario.workload_tokens_per_year:,.0f}")
    st.markdown(f"**Busy fraction:** {scenario.busy_fraction:.2f}")
    for r in scenario.results:
        st.markdown(f"- **{r.label}** — {r.notes}")

# ---------------------------------------------------------------------------
st.header("Export — 2-page sizing memo")

col1, col2 = st.columns([2, 1])
with col1:
    audience = st.text_input("For (audience, optional)", placeholder="Nscale GTM / deal desk / internal review")
with col2:
    st.write("")
    st.write("")
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
    )

with st.expander("Preview"):
    st.markdown(memo_md)
st.caption(
    "To PDF: open the downloaded .md in a markdown viewer and print to PDF, "
    "or run `pandoc memo.md -o memo.pdf`."
)
