"""Validation page — 'How accurate are the numbers behind this workbench?'.

Reframed from the old 'roofline then empty-state panels' shape. Now:

1. Hero accuracy metric (headline error vs measured).
2. Roofline chart with memory-bandwidth ceiling + the operating point.
3. T4 smoke test — real data if present, else a synthetic demo with a banner.
4. MLPerf cross-check — real CSVs if present, else a synthetic demo CSV.

Guiding principle: the page is never empty. A recruiter clicking from a CV
sees numbers and error bars immediately; the demo banner is honest about
which numbers come from real hardware vs the bundled synthetic set.
"""

from __future__ import annotations

import json
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
from _ui.provenance import badge  # noqa: E402

from core.cost_engine.kv_cache import (  # noqa: E402
    LLAMA3_8B,
    LLAMA3_70B,
    TINYLLAMA_1_1B,
    ModelArch,
)
from core.cost_engine.roofline import decode_tps, roofline_tflops  # noqa: E402
from core.gpu_specs import load_catalog  # noqa: E402
from core.mlperf import cross_check, load_bundled_results, read_results_csv  # noqa: E402

ROOT = _REPO_ROOT
T4_SMOKE_PATH = ROOT / "data" / "processed" / "t4_smoke_test.json"
T4_SWEEP_PATH = ROOT / "data" / "processed" / "t4_validation.parquet"
DEMO_SMOKE_PATH = _BUILD1 / "_ui" / "examples" / "t4_smoke_demo.json"
DEMO_MLPERF_PATH = _BUILD1 / "_ui" / "examples" / "mlperf_demo.csv"

# Extra arches for MLPerf cross-check (inline to avoid touching core/).
MIXTRAL_8X7B = ModelArch(
    name="mistralai/Mixtral-8x7B-v0.1",
    num_layers=32,
    num_attention_heads=32,
    num_kv_heads=8,  # GQA
    head_dim=128,
    hidden_size=4096,
    vocab_size=32000,
    max_position_embeddings=32768,
)

GPT_J_6B = ModelArch(
    name="EleutherAI/gpt-j-6b",
    num_layers=28,
    num_attention_heads=16,
    num_kv_heads=16,  # MHA
    head_dim=256,
    hidden_size=4096,
    vocab_size=50400,
    max_position_embeddings=2048,
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("How accurate are the predictions?")

page_header(
    what="Predicted tokens/sec vs what the hardware actually delivers.",
    why="A cost model is only credible if its throughput number is close to reality.",
    start="Scan the hero error metric. The roofline below shows where your workload lands on the physics ceiling.",
    numbers="Lower error % = closer to measured. Error = (predicted - measured) / measured.",
)

catalog = load_catalog()
MODELS = {
    "TinyLlama-1.1B": (TINYLLAMA_1_1B, 1.1),
    "Llama-3-8B":     (LLAMA3_8B, 8.0),
    "Llama-3-70B":    (LLAMA3_70B, 70.0),
}
gpu_sorted = sorted(catalog.gpus, key=lambda g: (g.vendor, g.release_year, g.name))
GPU_OPTIONS = {g.name: g.id for g in gpu_sorted}
GPU_NAMES = list(GPU_OPTIONS.keys())

# ---------------------------------------------------------------------------
# Hero accuracy metric (works from whichever source is available)
# ---------------------------------------------------------------------------
def _load_smoke() -> tuple[dict, bool]:
    """Return ``(smoke_dict, is_real)``. Falls back to the bundled demo."""
    if T4_SMOKE_PATH.exists():
        with open(T4_SMOKE_PATH) as f:
            return json.load(f), True
    with open(DEMO_SMOKE_PATH) as f:
        return json.load(f), False

smoke, smoke_is_real = _load_smoke()
t4_error_pct = float(smoke.get("error_pct", 0.0))

hero_cols = st.columns(3)
with hero_cols[0]:
    st.metric(
        "Mean error vs measured T4",
        f"{t4_error_pct:.1f} %",
        help="Average absolute percentage error of the roofline model against "
             "measured tokens/sec on a Kaggle T4 (TinyLlama-1.1B, batch 1).",
    )
with hero_cols[1]:
    st.metric(
        "Measured",
        f"{smoke.get('measured_decode_tps', 0):.1f} tok/s",
        help="Actual decode tokens/sec measured on the hardware.",
    )
with hero_cols[2]:
    st.metric(
        "Predicted",
        f"{smoke.get('predicted_decode_tps', 0):.1f} tok/s",
        help="What the roofline model predicts for the same workload.",
    )

if not smoke_is_real:
    st.info(
        f"{badge('simulated')} Showing a **synthetic demo fixture** — representative values, "
        "not a real hardware run. Owner action: execute "
        "`notebooks/00_t4_smoke_test.ipynb` on Kaggle T4 and commit "
        f"`{T4_SMOKE_PATH.relative_to(ROOT)}` to replace this with real data."
    )
else:
    st.success(f"{badge('measured')} Numbers above are from a real Kaggle T4 run.")

# ---------------------------------------------------------------------------
# Roofline chart with memory-bandwidth ceiling + operating point
# ---------------------------------------------------------------------------
st.markdown("## Roofline — where your workload hits the ceiling")
st.caption(caption("roofline_chart") + "  " + tooltip("roofline"))

c1, c2, c3 = st.columns(3)
with c1:
    gpu_name = st.selectbox(
        "GPU",
        options=GPU_NAMES,
        index=GPU_NAMES.index(catalog.by_id("h100_sxm5_80gb").name),
        key="val_gpu",
        help="Which accelerator's roofline to plot.",
    )
    gpu_id = GPU_OPTIONS[gpu_name]
with c2:
    model_name = st.selectbox(
        "Model",
        options=list(MODELS.keys()),
        index=1,
        key="val_model",
    )
with c3:
    util = st.slider(
        "Kernel utilisation of peak",
        0.1, 1.0, 0.65, 0.05,
        key="val_util",
        help=tooltip("kernel_utilisation"),
    )

gpu = catalog.by_id(gpu_id)
arch, param_b = MODELS[model_name]

intensities = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
achieved = [roofline_tflops(gpu, i, util)[0] for i in intensities]

# Memory-bandwidth ceiling: achievable TFLOPS = bandwidth (bytes/sec) * intensity / 1e12
bw_bytes_per_sec = gpu.peak_memory_bandwidth * util
memory_ceiling = [bw_bytes_per_sec * i / 1e12 for i in intensities]

# Operating point: approx arithmetic intensity for decode of this model.
decode = decode_tps(gpu, arch, param_b, batch_size=32, kv_seq_len=1024, utilisation_frac=util)
op_intensity = getattr(decode, "arithmetic_intensity", None) or 10.0
op_achieved = roofline_tflops(gpu, op_intensity, util)[0]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=intensities, y=achieved, mode="lines", name="Achievable TFLOPS (min of both ceilings)",
    line={"color": "steelblue", "width": 3},
))
fig.add_trace(go.Scatter(
    x=intensities, y=memory_ceiling, mode="lines", name="Memory-bandwidth ceiling",
    line={"color": "#FB8C00", "dash": "dot"},
))
fig.add_hline(
    y=gpu.fp16_tflops_dense * util,
    line_dash="dash",
    line_color="crimson",
    annotation_text=f"Compute ceiling ({gpu.fp16_tflops_dense * util:.0f} TFLOPS)",
)
# Operating point marker.
fig.add_trace(go.Scatter(
    x=[op_intensity],
    y=[op_achieved],
    mode="markers+text",
    text=[f"  {model_name} decode"],
    textposition="middle right",
    marker={"size": 14, "color": "#2E7D32", "symbol": "star"},
    name="Your workload",
))

fig.update_layout(
    xaxis_type="log",
    yaxis_type="log",
    xaxis_title="Arithmetic intensity (FLOPs per byte of memory moved)",
    yaxis_title="TFLOPS",
    height=420,
    title=f"{gpu.name} roofline at {int(util*100)}% kernel utilisation",
    margin={"t": 60, "b": 40, "l": 60, "r": 20},
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Memory-bound workloads (low intensity, left of the knee) are limited by HBM; "
    "compute-bound (high intensity, right of the knee) by tensor cores. Decode is "
    "usually memory-bound — that's why bandwidth matters more than FLOPS for "
    "serving."
)

# ---------------------------------------------------------------------------
# T4 smoke test details (expander)
# ---------------------------------------------------------------------------
with st.expander("T4 smoke-test details", expanded=False):
    cols = st.columns(4)
    cols[0].metric("Model", smoke.get("model", "—").split("/")[-1])
    cols[1].metric("Batch", smoke.get("batch_size", 1))
    cols[2].metric("Seq len", smoke.get("kv_seq_len", 1024))
    cols[3].metric("Precision", smoke.get("dtype", "fp16"))
    with st.expander("Raw smoke-test JSON"):
        st.json(smoke)

# ---------------------------------------------------------------------------
# MLPerf cross-check
# ---------------------------------------------------------------------------
st.markdown("## MLPerf cross-check")

def _predict(gpu_id: str, model_task: str, scenario: str) -> float | None:
    """Predict decode tok/s for a (GPU, MLPerf task, scenario) triple.

    Returns None gracefully when we don't recognise the model_task — the
    cross-check joiner skips those rows cleanly instead of crashing.
    """
    key = model_task.lower().replace(" ", "").replace("-", "")
    task_map: dict[str, tuple[ModelArch, float]] = {
        "llama270b":    (LLAMA3_70B, 70.0),   # same arch family as L3-70B for roofline
        "llama2_70b":   (LLAMA3_70B, 70.0),
        "llama370b":    (LLAMA3_70B, 70.0),
        "mixtral8x7b":  (MIXTRAL_8X7B, 46.7),  # MoE total params
        "mixtral":      (MIXTRAL_8X7B, 46.7),
        "gptj6b":       (GPT_J_6B, 6.0),
        "gptj":         (GPT_J_6B, 6.0),
    }
    if key not in task_map:
        return None
    try:
        spec = catalog.by_id(gpu_id)
    except KeyError:
        return None
    arch_, params = task_map[key]
    # MLPerf Server ~ online decode (small batch); Offline ~ max throughput (larger batch).
    bs = 64 if scenario.lower() == "offline" else 16
    return decode_tps(spec, arch_, params, batch_size=bs, kv_seq_len=1024,
                     utilisation_frac=util).tokens_per_second

mlperf_rows = load_bundled_results()
mlperf_is_real = bool(mlperf_rows)
if not mlperf_is_real:
    mlperf_rows = read_results_csv(DEMO_MLPERF_PATH, round_tag="demo__v5.0")

if not mlperf_is_real:
    st.info(
        f"{badge('simulated')} Showing a **synthetic MLPerf fixture** — "
        "representative Llama2-70B / Mixtral-8x7B / GPT-J-6B numbers for an 8× H100 SXM system. "
        "Owner action: drop real MLPerf CSVs into `data/raw/mlperf/` to replace."
    )
else:
    st.success(f"{badge('measured')} Numbers below are from real MLPerf submissions.")

checks = cross_check(mlperf_rows, _predict)
if checks:
    cdf = pd.DataFrame([c.as_dict() for c in checks])
    mean_err = cdf["error_pct"].abs().mean()

    hero = st.columns(3)
    hero[0].metric("Mean error vs MLPerf", f"{mean_err:.1f} %")
    hero[1].metric("Rows checked", len(cdf))
    hero[2].metric("Models covered", cdf["model_task"].nunique() if "model_task" in cdf.columns else 1)

    # Rename for display
    pretty = {
        "submitter": "Submitter",
        "system": "System",
        "model_task": "Model",
        "scenario": "Scenario",
        "measured_tps": "Measured (tok/s)",
        "predicted_tps": "Predicted (tok/s)",
        "error_pct": "Error %",
    }
    show_cols = [c for c in pretty if c in cdf.columns]
    st.dataframe(
        cdf[show_cols].rename(columns=pretty),
        hide_index=True,
        width="stretch",
    )
else:
    st.warning("No rows matched the prediction map. Extend `_predict` with more models.")

# ---------------------------------------------------------------------------
# Batch sweep (expander — reference)
# ---------------------------------------------------------------------------
with st.expander("Decode throughput by batch size (predicted)", expanded=False):
    batches = [1, 2, 4, 8, 16, 32, 64, 128]
    decode_rows = []
    for b in batches:
        res = decode_tps(gpu, arch, param_b, batch_size=b, kv_seq_len=1024, utilisation_frac=util)
        decode_rows.append({
            "batch": b,
            "tps_aggregate": res.tokens_per_second,
            "tps_per_request": res.tokens_per_second / b,
            "bound_by": res.bound,
        })
    decode_df = pd.DataFrame(decode_rows)

    c1, c2 = st.columns(2)
    with c1:
        f1 = go.Figure(data=[go.Scatter(x=decode_df["batch"], y=decode_df["tps_aggregate"],
                                        mode="lines+markers", name="aggregate")])
        f1.update_layout(title="Aggregate tok/s across the batch",
                         xaxis_title="Batch", yaxis_title="tok/s", height=330)
        st.plotly_chart(f1, width="stretch")
    with c2:
        f2 = go.Figure(data=[go.Scatter(x=decode_df["batch"], y=decode_df["tps_per_request"],
                                        mode="lines+markers", name="per request",
                                        line={"color": "#FB8C00"})])
        f2.update_layout(title="Per-request tok/s (latency proxy)",
                         xaxis_title="Batch", yaxis_title="tok/s", height=330)
        st.plotly_chart(f2, width="stretch")
    st.caption(
        "Aggregate throughput grows with batch; per-request throughput falls. "
        "That's the latency/throughput trade-off every serving system makes."
    )
    st.dataframe(decode_df, hide_index=True, width="stretch")
