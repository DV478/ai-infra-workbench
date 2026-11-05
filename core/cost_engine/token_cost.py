"""$ per million tokens — the bridge between the roofline model and price data.

Given a GPU, a model, a decode throughput prediction and an hourly $ rate,
compute the cost of serving one million output tokens. Two headline numbers:

* ``cost_per_m_tokens_peak``  — assumes the GPU is busy decoding 100% of the hour.
  This is the floor. Never achievable in practice.
* ``cost_per_m_tokens_realistic`` — assumes a serving-time utilisation fraction
  (``busy_fraction``, default 60%) covering request gaps, batch starvation
  and reserved headroom. This is what deal-desk pricing should use.

Prefill is priced separately because the economics are different: prefill
produces no customer-visible output tokens but consumes the GPU. We amortise
prefill cost into the output-token price assuming an average input/output
length ratio.

The implied-margin helper takes an advertised API $/M-tokens figure and
backs out the gross margin against our cost model, so the Streamlit UI can
show "Together AI's $2.00/M for Llama-3-70B implies X% gross margin at Y%
utilisation on H100 SXM5" — with every input tagged measured / modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.cost_engine.kv_cache import ModelArch
from core.cost_engine.roofline import decode_tps, prefill_tps
from core.gpu_specs.catalog import GpuSpec

SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class TokenCost:
    """One cost estimate for one (gpu, model, quant, batch) combo."""

    gpu_id: str
    model_name: str
    dtype: str
    batch_size: int
    kv_seq_len: int

    decode_tps: float  # aggregate output tokens/sec across the batch
    prefill_tps: float  # aggregate prompt tokens/sec across the batch
    hourly_usd: float
    busy_fraction: float
    input_output_ratio: float  # prompt tokens per output token (avg)

    cost_per_m_tokens_peak: float
    cost_per_m_tokens_realistic: float
    prefill_share_usd: float  # $ of the per-M-output cost that is prefill
    bound: str  # the decode bound ('compute' or 'memory')

    def as_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "model": self.model_name,
            "dtype": self.dtype,
            "batch": self.batch_size,
            "kv_seq_len": self.kv_seq_len,
            "decode_tps": round(self.decode_tps, 1),
            "prefill_tps": round(self.prefill_tps, 1),
            "hourly_usd": round(self.hourly_usd, 4),
            "busy_fraction": self.busy_fraction,
            "io_ratio": self.input_output_ratio,
            "cost_per_m_tokens_peak": round(self.cost_per_m_tokens_peak, 4),
            "cost_per_m_tokens_realistic": round(self.cost_per_m_tokens_realistic, 4),
            "prefill_share_usd": round(self.prefill_share_usd, 4),
            "bound": self.bound,
        }


def cost_per_m_tokens(
    gpu: GpuSpec,
    arch: ModelArch,
    param_count_billions: float,
    hourly_usd: float,
    *,
    batch_size: int = 32,
    input_length: int = 512,
    output_length: int = 256,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
    busy_fraction: float = 0.60,
) -> TokenCost:
    """Cost model for one hour of serving the given workload.

    ``utilisation_frac`` is the *roofline* utilisation (what fraction of peak
    tensor-core FLOPs the kernel achieves — a hardware-kernel efficiency).
    ``busy_fraction`` is the *serving* utilisation (what fraction of the
    billed hour is actually decoding). Keep them separate: a 65% kernel on
    a 60% busy GPU costs ~2.5x the theoretical floor.

    ``input_length`` and ``output_length`` set the average request shape.
    Prefill cost is amortised into the output token price proportionally.
    """
    if hourly_usd < 0:
        raise ValueError(f"hourly_usd must be non-negative, got {hourly_usd}")
    if not 0 < busy_fraction <= 1:
        raise ValueError(f"busy_fraction must be in (0, 1], got {busy_fraction}")

    decode = decode_tps(
        gpu,
        arch,
        param_count_billions,
        batch_size=batch_size,
        kv_seq_len=input_length + output_length // 2,
        dtype=dtype,
        utilisation_frac=utilisation_frac,
    )
    prefill = prefill_tps(
        gpu,
        arch,
        param_count_billions,
        batch_size=batch_size,
        input_length=input_length,
        dtype=dtype,
        utilisation_frac=utilisation_frac,
    )

    io_ratio = input_length / output_length if output_length > 0 else 0.0

    # Hourly tokens at peak: entire hour spent decoding at predicted TPS.
    # Hourly tokens realistic: discount by busy_fraction (gaps, starvation).
    decode_tokens_per_hour_peak = decode.tokens_per_second * SECONDS_PER_HOUR
    decode_tokens_per_hour_real = decode_tokens_per_hour_peak * busy_fraction

    peak_cost = (hourly_usd / decode_tokens_per_hour_peak) * 1_000_000 if decode_tokens_per_hour_peak > 0 else float("inf")
    realistic_decode_cost = (
        (hourly_usd / decode_tokens_per_hour_real) * 1_000_000
        if decode_tokens_per_hour_real > 0
        else float("inf")
    )

    # Amortise prefill: each output token comes with ``io_ratio`` input tokens.
    # Prefill cost per million output tokens = (io_ratio / prefill_tps) * hourly / busy * 1e6 ...
    # More precisely: time to prefill 1M * io_ratio input tokens = 1e6 * io_ratio / prefill_tps sec.
    # That time is billed at (hourly / 3600 / busy_fraction).
    if prefill.tokens_per_second > 0 and busy_fraction > 0:
        prefill_seconds_per_m_output = (1_000_000 * io_ratio) / prefill.tokens_per_second
        prefill_cost_per_m_output = (
            prefill_seconds_per_m_output * (hourly_usd / SECONDS_PER_HOUR) / busy_fraction
        )
    else:
        prefill_cost_per_m_output = 0.0

    realistic_total = realistic_decode_cost + prefill_cost_per_m_output

    return TokenCost(
        gpu_id=gpu.id,
        model_name=arch.name,
        dtype=dtype,
        batch_size=batch_size,
        kv_seq_len=input_length + output_length // 2,
        decode_tps=decode.tokens_per_second,
        prefill_tps=prefill.tokens_per_second,
        hourly_usd=hourly_usd,
        busy_fraction=busy_fraction,
        input_output_ratio=io_ratio,
        cost_per_m_tokens_peak=peak_cost,
        cost_per_m_tokens_realistic=realistic_total,
        prefill_share_usd=prefill_cost_per_m_output,
        bound=decode.bound,
    )


@dataclass(frozen=True)
class ApiMargin:
    """Back-out of implied gross margin from an advertised API rate."""

    api_name: str
    api_price_per_m_tokens: float
    our_cost_per_m_tokens: float
    gross_margin_pct: float
    busy_fraction: float
    note: str


def implied_api_margin(
    api_name: str,
    api_price_per_m_tokens: float,
    our_cost: TokenCost,
) -> ApiMargin:
    """Gross margin a provider would make if their cost base matched ours.

    This is not an accusation that the provider makes that margin — it is
    a reasoned benchmark the Streamlit UI shows alongside the TCO page,
    with a clear note that reseller margin, inefficiencies and multi-model
    overhead are not deducted.
    """
    cost = our_cost.cost_per_m_tokens_realistic
    if api_price_per_m_tokens <= 0:
        raise ValueError("api_price_per_m_tokens must be > 0")
    margin_pct = (api_price_per_m_tokens - cost) / api_price_per_m_tokens * 100.0
    return ApiMargin(
        api_name=api_name,
        api_price_per_m_tokens=api_price_per_m_tokens,
        our_cost_per_m_tokens=cost,
        gross_margin_pct=margin_pct,
        busy_fraction=our_cost.busy_fraction,
        note=(
            f"Implied gross margin assumes the provider runs {our_cost.model_name} "
            f"on {our_cost.gpu_id} at {int(our_cost.busy_fraction * 100)}% busy; "
            "does not deduct reseller margin, multi-tenant overhead or SRE cost."
        ),
    )


def sweep_cost(
    gpu: GpuSpec,
    arch: ModelArch,
    param_count_billions: float,
    hourly_usd: float,
    *,
    batches: tuple[int, ...] = (1, 4, 16, 32, 64),
    input_length: int = 512,
    output_length: int = 256,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
    busy_fraction: float = 0.60,
) -> list[TokenCost]:
    """Sweep batch sizes for a single (gpu, model) pair.

    Used to build the 'optimal batch size for cost' chart on the Streamlit
    TCO page — cost usually falls fast with batch, then flattens or rises
    once queue latency makes larger batches unrealistic.
    """
    return [
        cost_per_m_tokens(
            gpu,
            arch,
            param_count_billions,
            hourly_usd,
            batch_size=b,
            input_length=input_length,
            output_length=output_length,
            dtype=dtype,
            utilisation_frac=utilisation_frac,
            busy_fraction=busy_fraction,
        )
        for b in batches
    ]
