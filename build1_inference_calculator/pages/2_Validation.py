"""Validation page — predicted vs measured on T4, vs MLPerf reported.

Three panels:
  1. Roofline chart for the selected GPU + model
  2. Predicted-vs-measured T4 table once ``data/processed/t4_validation.parquet`` exists
  3. MLPerf cross-check table once ``data/raw/mlperf/*.csv`` is populated
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cost_engine.kv_cache import LLAMA3_8B, LLAMA3_70B, TINYLLAMA_1_1B
from core.cost_engine.roofline import decode_tps, roofline_tflops
from core.gpu_specs import load_catalog
from core.mlperf import cross_check, load_bundled_results

ROOT = Path(__file__).resolve().parent.parent.parent
T4_SMOKE_PATH = ROOT / "data" / "processed" / "t4_smoke_test.json"
T4_SWEEP_PATH = ROOT / "data" / "processed" / "t4_validation.parquet"

st.title("Validation — predicted vs measured")

st.caption(
    "Roofline predictions are **modelled**. Measured T4 numbers come from "
    "`notebooks/00_t4_smoke_test.ipynb` and `03_vllm_t4_validation.ipynb` "
    "(Kaggle T4 free tier, vLLM). MLPerf figures are the submitter's measured result."
)

catalog = load_catalog()
MODELS = {
    "TinyLlama-1.1B": (TINYLLAMA_1_1B, 1.1),
    "Llama-3-8B": (LLAMA3_8B, 8.0),
    "Llama-3-70B": (LLAMA3_70B, 70.0),
}

# ---------------------------------------------------------------------------
st.header("1. Roofline")

col1, col2, col3 = st.columns(3)
with col1:
    gpu_id = st.selectbox("GPU", options=catalog.ids(), index=catalog.ids().index("h100_sxm5_80gb"))
with col2:
    model_name = st.selectbox("Model", options=list(MODELS.keys()), index=1)
with col3:
    util = st.slider("Kernel utilisation of peak", 0.1, 1.0, 0.65, 0.05)

gpu = catalog.by_id(gpu_id)
arch, param_b = MODELS[model_name]

intensities = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
achieved = [roofline_tflops(gpu, i, util)[0] for i in intensities]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=intensities,
        y=achieved,
        mode="lines+markers",
        name="Achievable TFLOPS",
        line={"color": "steelblue"},
    )
)
fig.add_hline(
    y=gpu.fp16_tflops_dense * util,
    line_dash="dash",
    line_color="crimson",
    annotation_text=f"Compute ceiling ({gpu.fp16_tflops_dense * util:.0f} TFLOPS)",
)
fig.update_layout(
    xaxis_type="log",
    yaxis_type="log",
    xaxis_title="Arithmetic intensity (FLOPs / byte)",
    yaxis_title="TFLOPS",
    height=400,
    title=f"{gpu.name} — roofline (utilisation {int(util*100)}% of peak)",
)
st.plotly_chart(fig, use_container_width=True)

# Batch sweep for decode TPS
batches = [1, 2, 4, 8, 16, 32, 64, 128]
decode_rows = [
    {
        "batch": b,
        "tps_aggregate": decode_tps(gpu, arch, param_b, batch_size=b, kv_seq_len=1024, utilisation_frac=util).tokens_per_second,
        "tps_per_request": decode_tps(gpu, arch, param_b, batch_size=b, kv_seq_len=1024, utilisation_frac=util).tokens_per_second / b,
        "bound": decode_tps(gpu, arch, param_b, batch_size=b, kv_seq_len=1024, utilisation_frac=util).bound,
    }
    for b in batches
]
decode_df = pd.DataFrame(decode_rows)

col1, col2 = st.columns(2)
with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=decode_df["batch"], y=decode_df["tps_aggregate"], mode="lines+markers", name="aggregate"))
    fig2.update_layout(title="Predicted decode tok/s (aggregate across batch)", xaxis_title="Batch", yaxis_title="tok/s", height=350)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=decode_df["batch"], y=decode_df["tps_per_request"], mode="lines+markers", name="per request"))
    fig3.update_layout(title="Predicted decode tok/s (per request)", xaxis_title="Batch", yaxis_title="tok/s", height=350)
    st.plotly_chart(fig3, use_container_width=True)

st.dataframe(decode_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
st.header("2. T4 — predicted vs measured")

if T4_SMOKE_PATH.exists():
    with open(T4_SMOKE_PATH) as f:
        smoke = json.load(f)
    st.subheader("Smoke test (TinyLlama-1.1B, batch 1)")
    st.json(smoke)
else:
    st.warning(
        f"`{T4_SMOKE_PATH.relative_to(ROOT)}` not found. "
        "Run `notebooks/00_t4_smoke_test.ipynb` on Kaggle T4 and commit the output JSON."
    )

if T4_SWEEP_PATH.exists():
    st.subheader("Full sweep")
    df = pd.read_parquet(T4_SWEEP_PATH)
    # Expected columns: model, batch, in_len, measured_decode_tps, predicted_decode_tps
    if "predicted_decode_tps" in df.columns and "measured_decode_tps" in df.columns:
        df["error_pct"] = (df["predicted_decode_tps"] - df["measured_decode_tps"]) / df["measured_decode_tps"] * 100
        st.dataframe(df, hide_index=True, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["measured_decode_tps"],
                y=df["predicted_decode_tps"],
                mode="markers",
                text=df.get("model", ""),
                marker={"size": 10},
            )
        )
        lim = max(df["measured_decode_tps"].max(), df["predicted_decode_tps"].max()) * 1.1
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line={"dash": "dash", "color": "grey"})
        fig.update_layout(title="Predicted vs measured (identity line)", xaxis_title="Measured tok/s", yaxis_title="Predicted tok/s", height=500)
        st.plotly_chart(fig, use_container_width=True)
        mean_abs_err = df["error_pct"].abs().mean()
        st.metric("Mean absolute error vs measured T4", f"{mean_abs_err:.1f} %")
    else:
        st.dataframe(df, hide_index=True)
else:
    st.info(
        f"`{T4_SWEEP_PATH.relative_to(ROOT)}` not found. "
        "Run `notebooks/03_vllm_t4_validation.ipynb` for the full sweep."
    )

# ---------------------------------------------------------------------------
st.header("3. MLPerf cross-check")

mlperf_rows = load_bundled_results()

if not mlperf_rows:
    st.info(
        "No MLPerf CSVs in `data/raw/mlperf/`. "
        "Drop files named `<round>__<anything>.csv` (e.g. `v5.0__datacenter.csv`) "
        "there from https://mlcommons.org/en/inference-datacenter-50/."
    )
else:
    def _predict(gpu_id: str, model_task: str, scenario: str) -> float | None:
        # Simple map from MLPerf model-task string to our ModelArch.
        task_map = {
            "llama2-70b": (LLAMA3_70B, 70.0),
            "llama-2-70b": (LLAMA3_70B, 70.0),
            "llama3-70b": (LLAMA3_70B, 70.0),
        }
        key = model_task.lower().replace(" ", "")
        if key not in task_map:
            return None
        try:
            spec = catalog.by_id(gpu_id)
        except KeyError:
            return None
        arch_, params = task_map[key]
        # MLPerf Server scenario ~ batched decode; Offline ~ large batch decode.
        bs = 64 if scenario.lower() == "offline" else 16
        res = decode_tps(spec, arch_, params, batch_size=bs, kv_seq_len=1024, utilisation_frac=util)
        return res.tokens_per_second

    checks = cross_check(mlperf_rows, _predict)
    if not checks:
        st.warning("No rows matched our prediction map. Extend `_predict` with more models.")
    else:
        df = pd.DataFrame([c.as_dict() for c in checks])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.metric("Mean absolute error vs MLPerf", f"{df['error_pct'].abs().mean():.1f} %")
