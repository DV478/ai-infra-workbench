from __future__ import annotations

from core.cost_engine import (
    decode_tps,
    kv_cache_bytes,
    kv_cache_bytes_per_token,
    prefill_tps,
    roofline_tflops,
)
from core.cost_engine.kv_cache import (
    LLAMA3_8B,
    LLAMA3_70B,
    TINYLLAMA_1_1B,
    fits_in_memory,
    weights_bytes,
)
from core.gpu_specs import load_catalog

# ---------- KV cache math ----------------------------------------------------


def test_kv_cache_bytes_per_token_llama3_70b_fp16():
    # 2 * 80 layers * 8 kv_heads * 128 head_dim * 2 bytes = 327680
    assert kv_cache_bytes_per_token(LLAMA3_70B, "fp16") == 327680


def test_kv_cache_bytes_llama3_8b_batch16_len2048_fp16():
    # 2 * 32 * 8 * 128 * 2 = 131072 per token
    # per seq 131072 * 2048 = ~268 MB
    # batch 16 => ~4.29 GB
    cache = kv_cache_bytes(LLAMA3_8B, batch_size=16, sequence_length=2048, dtype="fp16")
    assert cache == 131072 * 2048 * 16
    assert 4.0e9 < cache < 4.5e9


def test_kv_cache_halves_at_fp8():
    fp16 = kv_cache_bytes(LLAMA3_8B, 1, 1024, "fp16")
    fp8 = kv_cache_bytes(LLAMA3_8B, 1, 1024, "fp8")
    assert fp16 == 2 * fp8


def test_weights_bytes():
    assert weights_bytes(8, "fp16") == 8e9 * 2
    assert weights_bytes(70, "fp8") == 70e9 * 1


def test_fits_in_memory_h100():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    # 8B FP16 weights = 16 GB; 1 GB KV cache; 2 GB overhead => fits easily.
    fits, headroom = fits_in_memory(16.0, 1.0, h100.memory_gb, 2.0)
    assert fits is True
    assert headroom > 60


def test_70b_fp16_does_not_fit_single_h100():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    fits, headroom = fits_in_memory(140.0, 10.0, h100.memory_gb, 2.0)
    assert fits is False
    assert headroom < 0


# ---------- Roofline --------------------------------------------------------


def test_roofline_low_intensity_is_memory_bound():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    # Intensity 10 FLOPs/byte is well below the knee for H100 (~295 FLOPs/B).
    tflops, bound = roofline_tflops(h100, arithmetic_intensity=10.0, utilisation_frac=1.0)
    assert bound == "memory"


def test_roofline_high_intensity_is_compute_bound():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    tflops, bound = roofline_tflops(h100, arithmetic_intensity=10_000.0, utilisation_frac=1.0)
    assert bound == "compute"


def test_decode_batch1_h100_llama3_8b_sanity():
    """At batch=1, 8B FP16 on H100 SXM5 should sit in the 90-200 tok/s band.

    Memory ceiling: 3350 GB/s * 65% / 16 GB ~ 136 tok/s.
    Allow a wide tolerance; the point is the number is in the right order.
    """
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    res = decode_tps(
        h100, LLAMA3_8B, param_count_billions=8.0,
        batch_size=1, kv_seq_len=1024, dtype="fp16", utilisation_frac=0.65,
    )
    assert res.bound == "memory"
    assert 50 < res.tokens_per_second < 250


def test_decode_scales_sublinearly_with_batch():
    """Decode throughput should rise with batch up to the knee, then flatten
    as it becomes compute-bound."""
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    tps_b1 = decode_tps(h100, LLAMA3_8B, 8.0, batch_size=1, kv_seq_len=512).tokens_per_second
    tps_b16 = decode_tps(h100, LLAMA3_8B, 8.0, batch_size=16, kv_seq_len=512).tokens_per_second
    assert tps_b16 > tps_b1  # batching helps
    # per-request TPS falls or stays flat as batch grows (aggregate rises)
    assert tps_b16 / 16 <= tps_b1 * 1.1


def test_prefill_is_compute_bound_for_h100_large_batch():
    cat = load_catalog()
    h100 = cat.by_id("h100_sxm5_80gb")
    res = prefill_tps(h100, LLAMA3_8B, 8.0, batch_size=16, input_length=2048, dtype="fp16")
    assert res.bound == "compute"


def test_tinyllama_decode_t4_sanity():
    """T4 + TinyLlama-1.1B FP16 at batch=1: memory-bound.

    1.1B * 2 B = 2.2 GB weights; T4 bandwidth 320 GB/s * 0.65 = 208 GB/s;
    so ~ 94 tok/s upper bound. Measured on vLLM is typically 70-100 tok/s.
    """
    cat = load_catalog()
    t4 = cat.by_id("t4")
    res = decode_tps(t4, TINYLLAMA_1_1B, 1.1, batch_size=1, kv_seq_len=512)
    assert res.bound == "memory"
    assert 60 < res.tokens_per_second < 150
