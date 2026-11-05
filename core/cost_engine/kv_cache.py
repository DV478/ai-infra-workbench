"""KV cache size calculations.

For a transformer with ``L`` layers, ``H`` attention heads, head dimension ``d``,
the KV cache for one token in one sequence is::

    2 * L * H_kv * d * dtype_bytes

where ``H_kv`` is the number of KV heads (``H_kv == H`` for MHA, less for GQA/MQA).

Total cache for a batch of ``B`` sequences of length ``T`` is::

    B * T * (2 * L * H_kv * d * dtype_bytes)

A transformer model's KV cache is often the dominant memory consumer at serving
time once sequences get long — this module gives the exact number so the UI can
show 'KV cache fit' independent of weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArch:
    """The arch fields a KV cache calc needs. Pull these from the model's
    ``config.json`` on HuggingFace, not from marketing material."""

    name: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int  # == num_attention_heads for MHA, less for GQA, 1 for MQA
    head_dim: int
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int

    @property
    def param_count_billions(self) -> float | None:
        """Not a derived number — leave None unless caller sets it."""
        return None


# Reference archs. Verify against the HF config before using in production.
LLAMA3_8B = ModelArch(
    name="meta-llama/Meta-Llama-3-8B",
    num_layers=32,
    num_attention_heads=32,
    num_kv_heads=8,  # GQA
    head_dim=128,
    hidden_size=4096,
    vocab_size=128256,
    max_position_embeddings=8192,
)

LLAMA3_70B = ModelArch(
    name="meta-llama/Meta-Llama-3-70B",
    num_layers=80,
    num_attention_heads=64,
    num_kv_heads=8,  # GQA
    head_dim=128,
    hidden_size=8192,
    vocab_size=128256,
    max_position_embeddings=8192,
)

QWEN2_5_1_5B = ModelArch(
    name="Qwen/Qwen2.5-1.5B",
    num_layers=28,
    num_attention_heads=12,
    num_kv_heads=2,  # GQA
    head_dim=128,
    hidden_size=1536,
    vocab_size=151936,
    max_position_embeddings=32768,
)

QWEN2_5_3B = ModelArch(
    name="Qwen/Qwen2.5-3B",
    num_layers=36,
    num_attention_heads=16,
    num_kv_heads=2,  # GQA
    head_dim=128,
    hidden_size=2048,
    vocab_size=151936,
    max_position_embeddings=32768,
)

TINYLLAMA_1_1B = ModelArch(
    name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_layers=22,
    num_attention_heads=32,
    num_kv_heads=4,  # GQA
    head_dim=64,
    hidden_size=2048,
    vocab_size=32000,
    max_position_embeddings=2048,
)


def _dtype_bytes(dtype: str) -> float:
    s = dtype.lower()
    if s in ("fp32", "float32"):
        return 4.0
    if s in ("fp16", "float16", "bf16", "bfloat16"):
        return 2.0
    if s in ("fp8", "int8"):
        return 1.0
    if s in ("int4", "fp4", "awq", "gptq"):
        return 0.5
    raise ValueError(f"Unknown dtype {dtype!r}")


def kv_cache_bytes_per_token(arch: ModelArch, dtype: str = "fp16") -> float:
    """Bytes of KV cache produced per output token for one sequence."""
    return 2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * _dtype_bytes(dtype)


def kv_cache_bytes(
    arch: ModelArch,
    batch_size: int,
    sequence_length: int,
    dtype: str = "fp16",
) -> float:
    """Total KV cache bytes for a batch of sequences at a given length."""
    return batch_size * sequence_length * kv_cache_bytes_per_token(arch, dtype)


def weights_bytes(param_count_billions: float, dtype: str = "fp16") -> float:
    """Approximate weight bytes. Caller owns ``param_count_billions`` since
    marketing names (``Llama-3-70B``) and precise counts (70.6 B) differ."""
    return param_count_billions * 1e9 * _dtype_bytes(dtype)


def fits_in_memory(
    weight_gb: float,
    kv_cache_gb: float,
    gpu_memory_gb: float,
    activation_overhead_gb: float = 2.0,
) -> tuple[bool, float]:
    """Returns ``(fits, headroom_gb)``. ``activation_overhead_gb`` is a rough
    allowance for activations + runtime; refine per engine when measured."""
    used = weight_gb + kv_cache_gb + activation_overhead_gb
    return used <= gpu_memory_gb, gpu_memory_gb - used
