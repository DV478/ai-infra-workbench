"""Spec page — GPU catalog + memory-fit calculator."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

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

st.title("GPU spec catalog")

catalog = load_catalog()
df = pd.DataFrame([g.model_dump() for g in catalog.gpus])

st.markdown(
    f"**{len(df)} GPUs** from Turing (T4, 2018) through Rubin R100 (2026, announcement-only). "
    "Dense tensor-core figures. Hover a row's citation column for the source."
)

confidence_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
df["conf"] = df["confidence"].map(confidence_colors) + " " + df["confidence"]

view_cols = [
    "id",
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
    "conf",
]
st.dataframe(
    df[view_cols].sort_values(["vendor", "release_year", "name"]),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Confidence legend"):
    st.markdown(
        "- 🟢 **high** — vendor datasheet PDF cited\n"
        "- 🟡 **medium** — web/press material; revisit as independent benchmarks mature\n"
        "- 🔴 **low** — announcement-only; FP16 TFLOPS may be modelled from relative-uplift claims"
    )

# ---------------------------------------------------------------------------
st.header("Memory fit")

model_options = {
    "TinyLlama-1.1B": (TINYLLAMA_1_1B, 1.1),
    "Qwen2.5-1.5B": (QWEN2_5_1_5B, 1.5),
    "Qwen2.5-3B": (QWEN2_5_3B, 3.0),
    "Llama-3-8B": (LLAMA3_8B, 8.0),
    "Llama-3-70B": (LLAMA3_70B, 70.0),
}

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    gpu_id = st.selectbox("GPU", options=catalog.ids(), index=catalog.ids().index("h100_sxm5_80gb"))
with col2:
    model_name = st.selectbox("Model", options=list(model_options.keys()), index=3)
with col3:
    quant = st.selectbox("Precision", options=["fp16", "fp8", "int8", "int4"], index=0)
with col4:
    batch = st.number_input("Batch size", min_value=1, max_value=512, value=32, step=1)
with col5:
    seqlen = st.number_input("Sequence length", min_value=128, max_value=131072, value=2048, step=128)

gpu = catalog.by_id(gpu_id)
arch, param_b = model_options[model_name]
w_bytes = weights_bytes(param_b, quant)
kv_bytes = kv_cache_bytes(arch, int(batch), int(seqlen), quant)
fits, headroom = fits_in_memory(
    weight_gb=w_bytes / 1e9,
    kv_cache_gb=kv_bytes / 1e9,
    gpu_memory_gb=gpu.memory_gb,
    activation_overhead_gb=2.0,
)

st.markdown(
    f"**{model_name}** ({quant}) on **{gpu.name}** at batch={batch}, seqlen={seqlen}: "
    + ("✅ **fits**" if fits else "❌ **does NOT fit**")
    + f" ({headroom:.1f} GB headroom of {gpu.memory_gb} GB)"
)

cfit1, cfit2, cfit3, cfit4 = st.columns(4)
cfit1.metric("Weights", f"{w_bytes/1e9:.1f} GB")
cfit2.metric("KV cache", f"{kv_bytes/1e9:.1f} GB")
cfit3.metric("Activations (allowance)", "2.0 GB")
cfit4.metric("GPU HBM", f"{gpu.memory_gb} GB")

st.caption(
    "KV cache is the dominant consumer once sequences get long. "
    "70B models at any batch will not fit on a single 80GB GPU in FP16; "
    "quantise to FP8 / INT4, or shard across GPUs. The calculator assumes single-GPU."
)
