"""3-year TCO: build vs rent vs API.

Five options are compared on the same denominator — $ per million output
tokens over a 3-year horizon:

* ``owned``       : buy the GPU, host it, pay power. Capex depreciated
                    straight-line over 36 months.
* ``reserved``    : hyperscaler 3-year reserved rate.
* ``on_demand``   : hyperscaler on-demand rate.
* ``neocloud``    : neocloud (Lambda / CoreWeave / Nebius / Nscale) typical
                    rate. These are quoted differently per vendor; the
                    caller passes one effective hourly $ rate and a note.
* ``api``         : per-token API price (Together, Fireworks, OpenRouter...).

For each option, we compute:

  - Total tokens the deployment can serve in 3 years at the configured
    busy_fraction. The owned/reserved/on-demand/neocloud options all run
    the same model on the same GPU with the same busy fraction, so the
    token-producing capacity is the same. The API option has zero capex
    and scales per-token.
  - Total $ spent over 3 years.
  - Effective $/M tokens.

Break-even:

  - owned vs reserved: busy_fraction above which owning beats renting.
  - any-build vs API: hourly $ at which renting equals API pricing.

Stress test:

  - "40% demand miss" = demand lands at 60% of forecast. The owned,
    reserved and neocloud options still pay for the capacity; API only
    pays for the tokens served.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.cost_engine.kv_cache import ModelArch
from core.cost_engine.roofline import decode_tps
from core.gpu_specs.catalog import GpuSpec

HOURS_PER_YEAR = 8760
MONTHS_PER_YEAR = 12
Option = Literal["owned", "reserved", "on_demand", "neocloud", "api"]


@dataclass(frozen=True)
class OwnedEconomics:
    """Capex + opex inputs for the 'buy the GPU' option.

    All figures are per-GPU per-year unless flagged otherwise. ``hosting_usd_per_month``
    covers rack, cross-connects and ops — colos quote this figure directly.
    """

    capex_usd: float  # GPU + server amortised per-GPU slot
    depreciation_years: int = 3
    power_draw_watts: float = 700.0  # H100 SXM5 TDP as example
    power_cost_usd_per_kwh: float = 0.10  # UK industrial ~ GBP 0.12 / USD 0.15; US colos cheaper
    pue: float = 1.4  # data centre power usage effectiveness
    hosting_usd_per_month: float = 1200.0  # rack space, cross-connect, remote hands
    financing_rate_pct: float = 8.0  # cost of capital
    maintenance_pct_of_capex_per_year: float = 3.0


@dataclass(frozen=True)
class TcoResult:
    option: Option
    label: str
    total_usd_3y: float
    total_output_tokens_3y: float
    usd_per_m_tokens: float
    monthly_usd: float
    notes: str

    def as_dict(self) -> dict:
        return {
            "option": self.option,
            "label": self.label,
            "total_usd_3y": round(self.total_usd_3y, 2),
            "total_output_tokens_3y": int(self.total_output_tokens_3y),
            "usd_per_m_tokens": round(self.usd_per_m_tokens, 4),
            "monthly_usd": round(self.monthly_usd, 2),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Serving capacity helper
# ---------------------------------------------------------------------------


def output_tokens_per_hour(
    gpu: GpuSpec,
    arch: ModelArch,
    param_count_billions: float,
    *,
    batch_size: int = 32,
    kv_seq_len: int = 640,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
    busy_fraction: float = 0.60,
) -> float:
    """Expected output tokens per hour of wall time for one GPU."""
    res = decode_tps(
        gpu,
        arch,
        param_count_billions,
        batch_size=batch_size,
        kv_seq_len=kv_seq_len,
        dtype=dtype,
        utilisation_frac=utilisation_frac,
    )
    return res.tokens_per_second * 3600 * busy_fraction


# ---------------------------------------------------------------------------
# Owned
# ---------------------------------------------------------------------------


def owned_hourly_cost(econ: OwnedEconomics) -> float:
    """Blended hourly $ for one GPU under ownership.

    Breakdown:
      - Depreciation: capex / (depreciation_years * HOURS_PER_YEAR)
      - Power: TDP (kW) * PUE * $/kWh
      - Hosting: monthly_usd * 12 / HOURS_PER_YEAR
      - Financing: capex * rate * 0.5 (avg over straight-line life) / HOURS_PER_YEAR
      - Maintenance: capex * maint_pct / HOURS_PER_YEAR
    """
    depreciation = econ.capex_usd / (econ.depreciation_years * HOURS_PER_YEAR)
    kw = econ.power_draw_watts / 1000.0
    power = kw * econ.pue * econ.power_cost_usd_per_kwh
    hosting = econ.hosting_usd_per_month * MONTHS_PER_YEAR / HOURS_PER_YEAR
    financing = econ.capex_usd * (econ.financing_rate_pct / 100.0) * 0.5 / HOURS_PER_YEAR
    maintenance = (
        econ.capex_usd * (econ.maintenance_pct_of_capex_per_year / 100.0) / HOURS_PER_YEAR
    )
    return depreciation + power + hosting + financing + maintenance


def owned_tco(
    econ: OwnedEconomics,
    output_tokens_per_hr: float,
    busy_fraction: float,
) -> TcoResult:
    hourly = owned_hourly_cost(econ)
    hours = HOURS_PER_YEAR * econ.depreciation_years
    total = hourly * hours
    tokens = output_tokens_per_hr * hours  # already scaled by busy_fraction in helper
    per_m = (total / tokens) * 1_000_000 if tokens > 0 else float("inf")
    monthly = hourly * HOURS_PER_YEAR / MONTHS_PER_YEAR
    return TcoResult(
        option="owned",
        label=f"Owned (capex ${econ.capex_usd:,.0f}, {econ.depreciation_years}y depreciation)",
        total_usd_3y=total,
        total_output_tokens_3y=tokens,
        usd_per_m_tokens=per_m,
        monthly_usd=monthly,
        notes=(
            f"Blended hourly ${hourly:.3f} = depreciation + power ({econ.power_draw_watts}W @ "
            f"PUE {econ.pue}) + hosting + financing + maintenance. "
            f"Assumes {int(busy_fraction * 100)}% busy."
        ),
    )


# ---------------------------------------------------------------------------
# Rented (reserved / on_demand / neocloud) — same math, different hourly input
# ---------------------------------------------------------------------------


def rented_tco(
    option: Option,
    label: str,
    hourly_usd: float,
    output_tokens_per_hr: float,
    busy_fraction: float,
    horizon_years: int = 3,
    notes: str | None = None,
) -> TcoResult:
    if option not in ("reserved", "on_demand", "neocloud"):
        raise ValueError(f"rented_tco expects reserved/on_demand/neocloud, got {option!r}")
    hours = HOURS_PER_YEAR * horizon_years
    total = hourly_usd * hours
    tokens = output_tokens_per_hr * hours
    per_m = (total / tokens) * 1_000_000 if tokens > 0 else float("inf")
    monthly = hourly_usd * HOURS_PER_YEAR / MONTHS_PER_YEAR
    return TcoResult(
        option=option,
        label=label,
        total_usd_3y=total,
        total_output_tokens_3y=tokens,
        usd_per_m_tokens=per_m,
        monthly_usd=monthly,
        notes=notes or f"{label} at ${hourly_usd:.3f}/hr for {horizon_years}y.",
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def api_tco(
    api_name: str,
    api_price_per_m_tokens: float,
    tokens_demanded_3y: float,
    input_price_per_m_tokens: float = 0.0,
    io_ratio: float = 2.0,
) -> TcoResult:
    """Cost of serving ``tokens_demanded_3y`` output tokens over 3y via API.

    Many providers quote input and output tokens at different rates (input
    typically 2-3x cheaper). ``io_ratio`` is prompt-tokens per output-token.
    """
    output_cost = (tokens_demanded_3y / 1_000_000) * api_price_per_m_tokens
    input_tokens = tokens_demanded_3y * io_ratio
    input_cost = (input_tokens / 1_000_000) * input_price_per_m_tokens
    total = output_cost + input_cost
    per_m_blended = (
        (total / tokens_demanded_3y) * 1_000_000 if tokens_demanded_3y > 0 else 0.0
    )
    monthly = total / (MONTHS_PER_YEAR * 3)
    return TcoResult(
        option="api",
        label=f"{api_name} API",
        total_usd_3y=total,
        total_output_tokens_3y=tokens_demanded_3y,
        usd_per_m_tokens=per_m_blended,
        monthly_usd=monthly,
        notes=(
            f"Output ${api_price_per_m_tokens}/M + input ${input_price_per_m_tokens}/M "
            f"at IO ratio {io_ratio}. API cost scales linearly with demand."
        ),
    )


# ---------------------------------------------------------------------------
# Break-even
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakEven:
    """Utilisation / price level at which two options become equal."""

    option_a: Option
    option_b: Option
    pivot_variable: str
    pivot_value: float
    note: str


def break_even_busy_fraction(
    econ: OwnedEconomics,
    rented_hourly_usd: float,
) -> BreakEven:
    """Busy fraction above which owning beats renting at the given hourly rate.

    The *hourly* cost comparison is independent of busy fraction — if owning
    is cheaper per hour than renting, it is cheaper at any utilisation.
    Busy fraction matters for the *per-token* comparison because owned
    carries fixed costs that still accrue when the GPU is idle. We return
    the busy fraction at which the per-token costs converge, assuming the
    rented hourly rate scales linearly with busy time (it does not — you
    pay for the reservation anyway — but this is the floor comparison).

    The useful intuition: at the crossover busy fraction, owned $/M tokens
    equals rented $/M tokens. Below it, rented is cheaper per token even
    though hourly rented may be more expensive.
    """
    owned_hourly = owned_hourly_cost(econ)
    if rented_hourly_usd <= 0:
        return BreakEven(
            option_a="owned",
            option_b="reserved",
            pivot_variable="busy_fraction",
            pivot_value=float("inf"),
            note="Rented hourly is zero — owned never breaks even.",
        )
    ratio = owned_hourly / rented_hourly_usd
    return BreakEven(
        option_a="owned",
        option_b="reserved",
        pivot_variable="busy_fraction",
        pivot_value=min(1.0, max(0.0, ratio)),
        note=(
            f"Owned hourly ${owned_hourly:.3f} vs rented ${rented_hourly_usd:.3f}. "
            f"Owning is cheaper on a $/token basis once busy_fraction >= {ratio:.2f} "
            "(and assumes rented is only paid for time used, which is only true for "
            "on-demand — reserved pays regardless)."
        ),
    )


def break_even_hourly_rate(
    cost_per_m_tokens_target: float,
    output_tokens_per_hr: float,
) -> BreakEven:
    """Hourly $ at which a rented GPU matches a target $/M-token cost (e.g. an API rate)."""
    if output_tokens_per_hr <= 0:
        return BreakEven(
            option_a="reserved",
            option_b="api",
            pivot_variable="hourly_usd",
            pivot_value=0.0,
            note="Zero output throughput — no break-even possible.",
        )
    hourly = (cost_per_m_tokens_target * output_tokens_per_hr) / 1_000_000
    return BreakEven(
        option_a="reserved",
        option_b="api",
        pivot_variable="hourly_usd",
        pivot_value=hourly,
        note=(
            f"At {int(output_tokens_per_hr):,} output tokens/hr, a rented GPU breaks even "
            f"with a ${cost_per_m_tokens_target}/M API at ${hourly:.3f}/hr."
        ),
    )


# ---------------------------------------------------------------------------
# Demand stress test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemandStressResult:
    """One option's cost under forecast vs. stressed demand.

    ``demand_multiplier = 0.6`` for the standard 40%-miss stress.
    """

    option: Option
    label: str
    base_usd_3y: float
    stressed_usd_3y: float
    base_usd_per_m: float
    stressed_usd_per_m: float
    tokens_delivered_3y: float
    stranded_capacity_usd: float  # >0 when capacity sat idle

    def as_dict(self) -> dict:
        return {
            "option": self.option,
            "label": self.label,
            "base_usd_3y": round(self.base_usd_3y, 2),
            "stressed_usd_3y": round(self.stressed_usd_3y, 2),
            "base_usd_per_m": round(self.base_usd_per_m, 4),
            "stressed_usd_per_m": round(self.stressed_usd_per_m, 4),
            "tokens_delivered_3y": int(self.tokens_delivered_3y),
            "stranded_capacity_usd": round(self.stranded_capacity_usd, 2),
        }


def demand_stress(
    result: TcoResult,
    demand_multiplier: float,
) -> DemandStressResult:
    """What happens to $/M-token cost if demand lands at ``demand_multiplier`` of forecast.

    For owned/reserved/neocloud the total $ stays fixed, tokens delivered
    falls, so $/M-token rises. For on-demand we assume capacity can be
    scaled down (rented hours shrink with demand). For API, cost scales
    linearly with demand so $/M-token is unchanged.
    """
    if not 0 < demand_multiplier <= 2.0:
        raise ValueError(f"demand_multiplier should be in (0, 2], got {demand_multiplier}")

    tokens_delivered = result.total_output_tokens_3y * demand_multiplier

    if result.option in ("owned", "reserved", "neocloud"):
        stressed_total = result.total_usd_3y  # fixed capacity, can't downscale
        stranded = result.total_usd_3y * (1 - demand_multiplier) if demand_multiplier < 1 else 0.0
    elif result.option in ("on_demand", "api"):
        stressed_total = result.total_usd_3y * demand_multiplier
        stranded = 0.0
    else:
        raise ValueError(f"Unknown option {result.option!r}")

    stressed_per_m = (
        (stressed_total / tokens_delivered) * 1_000_000 if tokens_delivered > 0 else float("inf")
    )
    return DemandStressResult(
        option=result.option,
        label=result.label,
        base_usd_3y=result.total_usd_3y,
        stressed_usd_3y=stressed_total,
        base_usd_per_m=result.usd_per_m_tokens,
        stressed_usd_per_m=stressed_per_m,
        tokens_delivered_3y=tokens_delivered,
        stranded_capacity_usd=stranded,
    )


# ---------------------------------------------------------------------------
# Scenario composer — the function the agent and the Streamlit TCO page call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TcoScenario:
    """One row per option in the headline comparison table."""

    label: str
    workload_tokens_per_year: float
    busy_fraction: float
    results: list[TcoResult]
    stress: list[DemandStressResult]
    break_evens: list[BreakEven]

    def summary(self) -> list[dict]:
        return [r.as_dict() for r in self.results]


def compare_options(
    gpu: GpuSpec,
    arch: ModelArch,
    param_count_billions: float,
    *,
    owned_econ: OwnedEconomics,
    reserved_hourly: float,
    on_demand_hourly: float,
    neocloud_hourly: float,
    api_name: str,
    api_output_price_per_m: float,
    api_input_price_per_m: float = 0.0,
    io_ratio: float = 2.0,
    batch_size: int = 32,
    input_length: int = 512,
    output_length: int = 256,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
    busy_fraction: float = 0.60,
    demand_stress_multiplier: float = 0.60,
) -> TcoScenario:
    """Compare all five options for one (gpu, model) workload.

    Designed to be the single function the agent calls. All numeric inputs
    must have come from the sources store upstream; this function does no
    citation handling itself.
    """
    tokens_per_hour = output_tokens_per_hour(
        gpu,
        arch,
        param_count_billions,
        batch_size=batch_size,
        kv_seq_len=input_length + output_length // 2,
        dtype=dtype,
        utilisation_frac=utilisation_frac,
        busy_fraction=busy_fraction,
    )
    tokens_per_year = tokens_per_hour * HOURS_PER_YEAR
    tokens_3y = tokens_per_year * 3

    owned = owned_tco(owned_econ, tokens_per_hour, busy_fraction)
    reserved = rented_tco(
        "reserved", "Reserved 3y cloud", reserved_hourly, tokens_per_hour, busy_fraction
    )
    on_demand = rented_tco(
        "on_demand", "On-demand cloud", on_demand_hourly, tokens_per_hour, busy_fraction
    )
    neocloud = rented_tco(
        "neocloud",
        "Neocloud (Lambda/Nebius/Nscale)",
        neocloud_hourly,
        tokens_per_hour,
        busy_fraction,
    )
    api = api_tco(
        api_name,
        api_output_price_per_m,
        tokens_3y,
        input_price_per_m_tokens=api_input_price_per_m,
        io_ratio=io_ratio,
    )

    results = [owned, reserved, on_demand, neocloud, api]
    stress = [demand_stress(r, demand_stress_multiplier) for r in results]
    break_evens = [
        break_even_busy_fraction(owned_econ, reserved_hourly),
        break_even_hourly_rate(api_output_price_per_m, tokens_per_hour),
    ]

    return TcoScenario(
        label=f"{arch.name} on {gpu.id}",
        workload_tokens_per_year=tokens_per_year,
        busy_fraction=busy_fraction,
        results=results,
        stress=stress,
        break_evens=break_evens,
    )
