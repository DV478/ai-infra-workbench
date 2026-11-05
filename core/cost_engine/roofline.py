"""Roofline analysis for LLM prefill and decode.

The roofline model gives an upper bound on achievable throughput as the
minimum of two ceilings:

    achieved_flops_per_second = min(
        peak_flops,                              # compute ceiling
        peak_bandwidth * arithmetic_intensity,   # memory ceiling
    )

where ``arithmetic_intensity`` is FLOPs per byte moved from HBM.

For LLM inference:

* **Prefill** is compute-bound for realistic context lengths. The intensity is
  roughly ``batch * seqlen / 2`` (crude, but captures the regime).
* **Decode** is memory-bound. Each new token requires streaming the full
  weight matrix once per layer; arithmetic intensity is roughly ``2 * batch``
  for the matmul plus KV cache reads.

We stay DECIDEDLY APPROXIMATE here: numbers are reported with explicit
``modelled`` kind, validated against measured T4 runs, and cross-checked
against MLPerf. The point is to have an auditable prediction, not an
oracle. See ``notebooks/01_roofline_t4_h100.ipynb`` for the written-up version.

References:
- "Dissecting Batching Effects in GPT Inference" (Han et al., 2023)
- "Efficiently Scaling Transformer Inference" (Pope et al., 2022)
- Horace He, "Making deep learning go brrrr from first principles"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.cost_engine.kv_cache import ModelArch, _dtype_bytes
from core.gpu_specs.catalog import GpuSpec

Bound = Literal["compute", "memory"]


@dataclass(frozen=True)
class RooflineResult:
    tokens_per_second: float
    bound: Bound
    arithmetic_intensity: float  # FLOPs per byte
    utilisation_frac: float  # fraction of peak we assume the kernel achieves


def _weight_bytes_from_params(param_count_billions: float, dtype: str) -> float:
    return param_count_billions * 1e9 * _dtype_bytes(dtype)


def roofline_tflops(
    gpu: GpuSpec,
    arithmetic_intensity: float,
    utilisation_frac: float = 0.65,
) -> tuple[float, Bound]:
    """Return ``(achieved_tflops, bound)`` for a given arithmetic intensity.

    ``utilisation_frac`` discounts peak to reflect real kernels (65% is a
    reasonable default for well-tuned FP16 GEMM on NVIDIA; measured T4
    runs will refine this).
    """
    peak_flops = gpu.peak_fp16_flops * utilisation_frac
    peak_bw = gpu.peak_memory_bandwidth * utilisation_frac
    memory_ceiling = peak_bw * arithmetic_intensity

    if memory_ceiling < peak_flops:
        return memory_ceiling / 1e12, "memory"
    return peak_flops / 1e12, "compute"


def prefill_tps(
    gpu: GpuSpec,
    arch: ModelArch,
    param_count_billions: float,
    batch_size: int,
    input_length: int,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
) -> RooflineResult:
    """Predicted prefill throughput in prompt-tokens/second across the batch.

    Prefill does roughly ``2 * P * B * T`` FLOPs (forward pass through
    ``P`` params for ``B * T`` tokens). Memory traffic is ``weights + KV_write``.
    """
    p = param_count_billions * 1e9
    flops_per_prefill_token = 2 * p
    weight_bytes = _weight_bytes_from_params(param_count_billions, dtype)

    # Bytes moved per full prefill: weights read once + KV written for B*T tokens.
    kv_bytes_per_token = 2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * _dtype_bytes(dtype)
    total_tokens = batch_size * input_length
    bytes_moved = weight_bytes + total_tokens * kv_bytes_per_token

    intensity = (flops_per_prefill_token * total_tokens) / bytes_moved

    achieved_tflops, bound = roofline_tflops(gpu, intensity, utilisation_frac)
    achieved_flops = achieved_tflops * 1e12
    tps = achieved_flops / flops_per_prefill_token  # tokens / sec, across batch
    return RooflineResult(
        tokens_per_second=tps,
        bound=bound,
        arithmetic_intensity=intensity,
        utilisation_frac=utilisation_frac,
    )


def decode_tps(
    gpu: GpuSpec,
    arch: ModelArch,
    param_count_billions: float,
    batch_size: int,
    kv_seq_len: int,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
) -> RooflineResult:
    """Predicted decode throughput in new-tokens/second across the batch.

    Decode moves ``weights + KV_read`` per step, does ``2 * P * B`` FLOPs.
    At batch=1 this is dominated by weight streaming, so TPS ~
    ``peak_bandwidth / weight_bytes``.
    """
    p = param_count_billions * 1e9
    flops_per_step = 2 * p * batch_size  # across the batch
    weight_bytes = _weight_bytes_from_params(param_count_billions, dtype)

    kv_bytes_per_token = 2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * _dtype_bytes(dtype)
    # Each decode step reads the full KV cache for each sequence in the batch.
    kv_bytes_read = batch_size * kv_seq_len * kv_bytes_per_token
    bytes_moved = weight_bytes + kv_bytes_read

    intensity = flops_per_step / bytes_moved
    achieved_tflops, bound = roofline_tflops(gpu, intensity, utilisation_frac)
    achieved_flops = achieved_tflops * 1e12

    tps = achieved_flops / (flops_per_step / batch_size)  # new tokens/sec across batch
    return RooflineResult(
        tokens_per_second=tps,
        bound=bound,
        arithmetic_intensity=intensity,
        utilisation_frac=utilisation_frac,
    )
