"""Spec page — 'Will your model fit on this GPU?'.

Flipped from the old 'catalog first, memory-fit below' shape. Now:

1. Memory-fit answer up top (hero chart + verdict).
2. Full catalog in an expander as reference.

The memory-fit question is the one a non-technical reader can actually answer
('does the thing even run?') without needing a mental model of TFLOPS or
bandwidth. The catalog stays for the SME reader who wants to compare rows.
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
from _ui.explain import caption, tooltip  # noqa: E402
from _ui.page_header import page_header  # noqa: E402
from _ui.presets import SCENARIOS, preset_chips, scenario_default  # noqa: E402
from _ui.provenance import confidence_badge  # noqa: E402

from core.cost_engine.kv_cache import (  # noqa: E402
    LLAMA3_8B,
    LLAMA3_70B,
    QWEN2_5_1_5B,
    QWEN2_5_3B,
    TINYLLAMA_1_1B,
    fits_in_memory,
    kv_cache_bytes,
    weights_bytes,
)
from core.gpu_specs import load_catalog  # noqa: E402

st.title("Will your model fit on this GPU?")

page_header(
    what="Memory fit check for any open-weight model on any supported GPU.",
    why=(
        "GPU memory (HBM) is the bottleneck for 70B-class models. "
        "If weights + KV cache + activations exceed it, serving fails."
    ),
    start="Click a preset below — or pick a model and GPU and tune the batch.",
    numbers=(
        "Bars show memory in GB. The red line is the GPU's HBM ceiling. "
        "Under the line = fits; over = doesn't fit."
    ),
)

# ---------------------------------------------------------------------------
# Preset chips
# ---------------------------------------------------------------------------
preset_chips("spec", SCENARIOS)

catalog = load_catalog()
MODEL_OPTIONS = {
    "TinyLlama-1.1B": (TINYLLAMA_1_1B, 1.1),
    "Qwen2.5-1.5B":   (QWEN2_5_1_5B, 1.5),
    "Qwen2.5-3B":     (QWEN2_5_3B, 3.0),
    "Llama-3-8B":     (LLAMA3_8B, 8.0),
    "Llama-3-70B":    (LLAMA3_70B, 70.0),
}

# Pretty-name GPU selector. We build a {name: id} map so the UI shows the
# human name but the engine gets the id.
gpu_sorted = sorted(catalog.gpus, key=lambda g: (g.vendor, g.release_year, g.name))
GPU_OPTIONS = {g.name: g.id for g in gpu_sorted}
GPU_NAMES = list(GPU_OPTIONS.keys())
_default_gpu_name = catalog.by_id(
    scenario_default("spec", "gpu_id", "h100_sxm5_80gb")
).name
_default_model_name = scenario_default("spec", "model_name", "Llama-3-8B")
if _default_model_name not in MODEL_OPTIONS:
    _default_model_name = "Llama-3-8B"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    gpu_name = st.selectbox(
        "GPU",
        options=GPU_NAMES,
        index=GPU_NAMES.index(_default_gpu_name),
        help="Pick the accelerator you'd run the model on. Pretty names only — "
             "internal IDs stay in the engine.",
    )
    gpu_id = GPU_OPTIONS[gpu_name]
with col2:
    model_name = st.selectbox(
        "Model",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(_default_model_name),
        help="Open-weight model to serve. Larger = more capable, more memory.",
    )
with col3:
    quant = st.selectbox(
        "Precision",
        options=["fp16", "fp8", "int8", "int4"],
        index=0,
        help=(
            f"{tooltip('fp16')} FP8/INT8 roughly halve memory; INT4 quarters it, "
            "with small accuracy cost."
        ),
    )

col4, col5 = st.columns(2)
with col4:
    batch = st.number_input(
        "Batch size",
        min_value=1,
        max_value=512,
        value=int(scenario_default("spec", "batch", 32)),
        step=1,
        help=tooltip("batch_size"),
    )
with col5:
    seqlen = st.number_input(
        "Sequence length (prompt + reply)",
        min_value=128,
        max_value=131072,
        value=int(scenario_default("spec", "in_len", 512) + scenario_default("spec", "out_len", 256) * 4),
        step=128,
        help=(
            f"Total tokens the KV cache must hold. {tooltip('kv_cache')}"
        ),
    )

gpu = catalog.by_id(gpu_id)
arch, param_b = MODEL_OPTIONS[model_name]
w_bytes = weights_bytes(param_b, quant)
kv_bytes = kv_cache_bytes(arch, int(batch), int(seqlen), quant)
activation_gb = 2.0
fits, headroom = fits_in_memory(
    weight_gb=w_bytes / 1e9,
    kv_cache_gb=kv_bytes / 1e9,
    gpu_memory_gb=gpu.memory_gb,
    activation_overhead_gb=activation_gb,
)

# ---------------------------------------------------------------------------
# Hero verdict + stacked-bar memory chart
# ---------------------------------------------------------------------------
verdict_col, chart_col = st.columns([2, 3])

with verdict_col:
    if fits:
        st.success(
            f"**Fits** — {model_name} ({quant}) on {gpu.name}.\n\n"
            f"**{headroom:.1f} GB headroom** of {gpu.memory_gb} GB."
        )
    else:
        st.error(
            f"**Does not fit** — {model_name} ({quant}) on {gpu.name}.\n\n"
            f"Needs **{-headroom:.1f} GB more** than this GPU provides."
        )
    st.caption(caption("memory_fit"))
    if not fits:
        st.markdown(
            "**Fix options:**\n"
            "- Lower the batch size or sequence length (shrinks KV cache)\n"
            "- Switch precision to FP8 or INT4\n"
            "- Shard the model across multiple GPUs (not modelled here — see TCO)"
        )

with chart_col:
    weights_gb = w_bytes / 1e9
    kv_gb = kv_bytes / 1e9
    used_gb = weights_gb + kv_gb + activation_gb

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Weights",
        x=["Memory use"],
        y=[weights_gb],
        marker_color="#1E88E5",
        text=[f"{weights_gb:.1f} GB"],
        textposition="inside",
        hovertemplate="Weights<br>%{y:.1f} GB<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="KV cache",
        x=["Memory use"],
        y=[kv_gb],
        marker_color="#FB8C00",
        text=[f"{kv_gb:.1f} GB"],
        textposition="inside",
        hovertemplate="KV cache<br>%{y:.1f} GB<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Activations",
        x=["Memory use"],
        y=[activation_gb],
        marker_color="#8E24AA",
        text=[f"{activation_gb:.1f} GB"],
        textposition="inside",
        hovertemplate="Activation allowance<br>%{y:.1f} GB<extra></extra>",
    ))
    # HBM ceiling line.
    fig.add_hline(
        y=gpu.memory_gb,
        line_dash="dash",
        line_color="#C62828",
        annotation_text=f"{gpu.name} ceiling: {gpu.memory_gb} GB",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="stack",
        title="Where the memory goes",
        yaxis_title="GB",
        yaxis_range=[0, max(gpu.memory_gb, used_gb) * 1.15],
        height=380,
        showlegend=True,
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Catalog reference (collapsed)
# ---------------------------------------------------------------------------
with st.expander(f"Browse the full GPU catalog ({len(catalog.gpus)} rows)"):
    st.caption(
        f"Turing (T4, 2018) → Rubin R100 (2026). Dense tensor-core figures. "
        "Hover a row's citation column for the source. "
        f"{tooltip('confidence_badge')}"
    )
    df = pd.DataFrame([g.model_dump() for g in catalog.gpus])
    df["Confidence"] = df["confidence"].map(confidence_badge)
    view_cols = [
        "name",
        "vendor",
        "architecture",
        "release_year",
        "fp16_tflops_dense",
        "fp8_tflops_dense",
        "memory_gb",
        "memory_type",
        "memory_bandwidth_gbs",
        "tdp_watts",
        "Confidence",
        "citation",
        "citation_date",
    ]
    rename = {
        "name": "GPU",
        "vendor": "Vendor",
        "architecture": "Architecture",
        "release_year": "Released",
        "fp16_tflops_dense": "FP16 TFLOPS",
        "fp8_tflops_dense": "FP8 TFLOPS",
        "memory_gb": "Memory (GB)",
        "memory_type": "Memory type",
        "memory_bandwidth_gbs": "Bandwidth (GB/s)",
        "tdp_watts": "TDP (W)",
        "citation": "Source",
        "citation_date": "Citation date",
    }
    st.dataframe(
        df[view_cols].sort_values(["vendor", "release_year", "name"]).rename(columns=rename),
        width="stretch",
        hide_index=True,
    )
