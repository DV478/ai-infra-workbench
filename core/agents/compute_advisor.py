"""Compute advisor agent — the Build 1 headline agent.

Answers sizing questions ("serve a 70B model at 50 rps for 3 years — rent,
buy or API?") by calling tool-wrapped functions from ``core/cost_engine``.
Every figure in the response is the return value of a tool call, never an
LLM-produced number. The eval harness in ``core/eval`` grades the agent
against known answers and reports a pass rate.

Design choices:

- **Direct Anthropic SDK with prompt caching.** Haiku 4.5 by default to
  fit the 5-10 GBP/month API budget. System prompt + tool definitions are
  cached because they are stable; only the user turn varies.
- **Tool registry is just a dict of plain Python callables.** Tests call
  ``run_tool("decode_tps", {...})`` directly — no LLM in the loop. The
  LLM only sees JSON schemas via ``TOOL_SCHEMAS``.
- **Deterministic executor.** The agent can be run against a transcript
  without hitting the API by supplying ``executor=replay``, used by the
  eval harness to check tool-dispatch logic independently of model
  stochasticity.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.cost_engine.kv_cache import (
    LLAMA3_8B,
    LLAMA3_70B,
    QWEN2_5_1_5B,
    QWEN2_5_3B,
    TINYLLAMA_1_1B,
    ModelArch,
    fits_in_memory,
    kv_cache_bytes,
    weights_bytes,
)
from core.cost_engine.roofline import decode_tps, prefill_tps
from core.cost_engine.tco import (
    OwnedEconomics,
    api_tco,
    break_even_hourly_rate,
    compare_options,
    output_tokens_per_hour,
    owned_hourly_cost,
)
from core.cost_engine.token_cost import cost_per_m_tokens, implied_api_margin
from core.gpu_specs.catalog import load_catalog

# ---------------------------------------------------------------------------
# Model-arch lookup. Agent passes a string name; we resolve to the arch.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[ModelArch, float]] = {
    "tinyllama-1.1b": (TINYLLAMA_1_1B, 1.1),
    "qwen2.5-1.5b": (QWEN2_5_1_5B, 1.5),
    "qwen2.5-3b": (QWEN2_5_3B, 3.0),
    "llama-3-8b": (LLAMA3_8B, 8.0),
    "llama-3-70b": (LLAMA3_70B, 70.0),
}


def _resolve_model(name: str) -> tuple[ModelArch, float]:
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Known: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key]


def _resolve_gpu(gpu_id: str):
    catalog = load_catalog()
    try:
        return catalog.by_id(gpu_id)
    except KeyError as e:
        raise ValueError(str(e)) from e


# ---------------------------------------------------------------------------
# Tool implementations — thin wrappers that convert agent-friendly string
# inputs to core-library types and back to JSON-safe dicts.
# ---------------------------------------------------------------------------


def _tool_decode_tps(
    gpu_id: str,
    model: str,
    batch_size: int = 32,
    kv_seq_len: int = 1024,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
) -> dict:
    gpu = _resolve_gpu(gpu_id)
    arch, p = _resolve_model(model)
    res = decode_tps(gpu, arch, p, batch_size=batch_size, kv_seq_len=kv_seq_len, dtype=dtype, utilisation_frac=utilisation_frac)
    return {
        "gpu_id": gpu_id,
        "model": model,
        "tokens_per_second": round(res.tokens_per_second, 1),
        "bound": res.bound,
        "arithmetic_intensity": round(res.arithmetic_intensity, 2),
    }


def _tool_prefill_tps(
    gpu_id: str,
    model: str,
    batch_size: int = 32,
    input_length: int = 512,
    dtype: str = "fp16",
    utilisation_frac: float = 0.65,
) -> dict:
    gpu = _resolve_gpu(gpu_id)
    arch, p = _resolve_model(model)
    res = prefill_tps(gpu, arch, p, batch_size=batch_size, input_length=input_length, dtype=dtype, utilisation_frac=utilisation_frac)
    return {
        "tokens_per_second": round(res.tokens_per_second, 1),
        "bound": res.bound,
        "arithmetic_intensity": round(res.arithmetic_intensity, 2),
    }


def _tool_fits_in_memory(gpu_id: str, model: str, dtype: str, batch_size: int, seq_len: int) -> dict:
    gpu = _resolve_gpu(gpu_id)
    arch, p = _resolve_model(model)
    w = weights_bytes(p, dtype) / 1e9
    kv = kv_cache_bytes(arch, batch_size, seq_len, dtype) / 1e9
    fits, headroom = fits_in_memory(w, kv, gpu.memory_gb, activation_overhead_gb=2.0)
    return {
        "fits": fits,
        "weights_gb": round(w, 2),
        "kv_cache_gb": round(kv, 2),
        "headroom_gb": round(headroom, 2),
        "gpu_memory_gb": gpu.memory_gb,
    }


def _tool_cost_per_m_tokens(
    gpu_id: str,
    model: str,
    hourly_usd: float,
    batch_size: int = 32,
    input_length: int = 512,
    output_length: int = 256,
    dtype: str = "fp16",
    busy_fraction: float = 0.60,
    utilisation_frac: float = 0.65,
) -> dict:
    gpu = _resolve_gpu(gpu_id)
    arch, p = _resolve_model(model)
    tc = cost_per_m_tokens(
        gpu, arch, p, hourly_usd,
        batch_size=batch_size, input_length=input_length, output_length=output_length,
        dtype=dtype, utilisation_frac=utilisation_frac, busy_fraction=busy_fraction,
    )
    return tc.as_dict()


def _tool_compare_options(
    gpu_id: str,
    model: str,
    reserved_hourly: float,
    on_demand_hourly: float,
    neocloud_hourly: float,
    api_output_price_per_m: float,
    api_input_price_per_m: float = 0.0,
    api_name: str = "API",
    capex_usd: float = 32000.0,
    hosting_usd_per_month: float = 1200.0,
    power_cost_usd_per_kwh: float = 0.10,
    batch_size: int = 32,
    input_length: int = 512,
    output_length: int = 256,
    dtype: str = "fp16",
    busy_fraction: float = 0.60,
    demand_stress_multiplier: float = 0.60,
) -> dict:
    gpu = _resolve_gpu(gpu_id)
    arch, p = _resolve_model(model)
    econ = OwnedEconomics(
        capex_usd=capex_usd,
        power_draw_watts=gpu.tdp_watts or 700.0,
        power_cost_usd_per_kwh=power_cost_usd_per_kwh,
        hosting_usd_per_month=hosting_usd_per_month,
    )
    scenario = compare_options(
        gpu, arch, p,
        owned_econ=econ,
        reserved_hourly=reserved_hourly,
        on_demand_hourly=on_demand_hourly,
        neocloud_hourly=neocloud_hourly,
        api_name=api_name,
        api_output_price_per_m=api_output_price_per_m,
        api_input_price_per_m=api_input_price_per_m,
        batch_size=batch_size, input_length=input_length, output_length=output_length,
        dtype=dtype, busy_fraction=busy_fraction,
        demand_stress_multiplier=demand_stress_multiplier,
    )
    return {
        "options": [r.as_dict() for r in scenario.results],
        "stress": [s.as_dict() for s in scenario.stress],
        "break_evens": [
            {"a": be.option_a, "b": be.option_b, "var": be.pivot_variable, "value": round(be.pivot_value, 4), "note": be.note}
            for be in scenario.break_evens
        ],
        "workload_tokens_per_year": int(scenario.workload_tokens_per_year),
    }


def _tool_api_margin(
    gpu_id: str,
    model: str,
    hourly_usd: float,
    api_name: str,
    api_price_per_m_tokens: float,
    batch_size: int = 32,
    input_length: int = 512,
    output_length: int = 256,
    dtype: str = "fp16",
    busy_fraction: float = 0.60,
) -> dict:
    gpu = _resolve_gpu(gpu_id)
    arch, p = _resolve_model(model)
    tc = cost_per_m_tokens(
        gpu, arch, p, hourly_usd,
        batch_size=batch_size, input_length=input_length, output_length=output_length,
        dtype=dtype, busy_fraction=busy_fraction,
    )
    m = implied_api_margin(api_name, api_price_per_m_tokens, tc)
    return {
        "api_name": m.api_name,
        "api_price_per_m_tokens": m.api_price_per_m_tokens,
        "our_cost_per_m_tokens": round(m.our_cost_per_m_tokens, 4),
        "gross_margin_pct": round(m.gross_margin_pct, 1),
        "note": m.note,
    }


def _tool_output_tokens_per_hour(
    gpu_id: str,
    model: str,
    batch_size: int = 32,
    kv_seq_len: int = 1024,
    dtype: str = "fp16",
    busy_fraction: float = 0.60,
) -> dict:
    gpu = _resolve_gpu(gpu_id)
    arch, p = _resolve_model(model)
    tph = output_tokens_per_hour(gpu, arch, p, batch_size=batch_size, kv_seq_len=kv_seq_len, dtype=dtype, busy_fraction=busy_fraction)
    tph_int = int(tph)
    return {"tokens_per_hour": tph_int, "tokens_per_day": tph_int * 24}


def _tool_owned_hourly_cost(
    capex_usd: float,
    power_draw_watts: float = 700.0,
    power_cost_usd_per_kwh: float = 0.10,
    pue: float = 1.4,
    hosting_usd_per_month: float = 1200.0,
    financing_rate_pct: float = 8.0,
    maintenance_pct_of_capex_per_year: float = 3.0,
    depreciation_years: int = 3,
) -> dict:
    econ = OwnedEconomics(
        capex_usd=capex_usd,
        depreciation_years=depreciation_years,
        power_draw_watts=power_draw_watts,
        power_cost_usd_per_kwh=power_cost_usd_per_kwh,
        pue=pue,
        hosting_usd_per_month=hosting_usd_per_month,
        financing_rate_pct=financing_rate_pct,
        maintenance_pct_of_capex_per_year=maintenance_pct_of_capex_per_year,
    )
    return {"hourly_usd": round(owned_hourly_cost(econ), 4)}


def _tool_break_even_hourly(cost_per_m_tokens_target: float, output_tokens_per_hr: float) -> dict:
    be = break_even_hourly_rate(cost_per_m_tokens_target, output_tokens_per_hr)
    return {"hourly_usd": round(be.pivot_value, 4), "note": be.note}


def _tool_api_only_cost(
    api_output_price_per_m: float,
    tokens_demanded_3y: float,
    api_input_price_per_m: float = 0.0,
    io_ratio: float = 2.0,
    api_name: str = "API",
) -> dict:
    res = api_tco(api_name, api_output_price_per_m, tokens_demanded_3y, api_input_price_per_m, io_ratio)
    return res.as_dict()


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    "decode_tps": _tool_decode_tps,
    "prefill_tps": _tool_prefill_tps,
    "fits_in_memory": _tool_fits_in_memory,
    "cost_per_m_tokens": _tool_cost_per_m_tokens,
    "compare_options": _tool_compare_options,
    "api_margin": _tool_api_margin,
    "output_tokens_per_hour": _tool_output_tokens_per_hour,
    "owned_hourly_cost": _tool_owned_hourly_cost,
    "break_even_hourly": _tool_break_even_hourly,
    "api_only_cost": _tool_api_only_cost,
}


def run_tool(name: str, params: dict[str, Any]) -> dict:
    """Dispatch a tool call. Raises if the tool is unknown."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool {name!r}. Available: {sorted(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name](**params)


# ---------------------------------------------------------------------------
# Tool schemas (JSON schema fragments for the Anthropic API tool-use format)
# ---------------------------------------------------------------------------

_GPU_IDS = [
    "t4", "l4", "l40s", "a100_sxm_80gb",
    "h100_sxm5_80gb", "h100_pcie_80gb", "h200_sxm_141gb",
    "b100", "b200", "gb200", "b300", "r100",
    "tpu_v5e", "tpu_v5p", "tpu_trillium",
    "trn1", "trn2",
]

_MODEL_NAMES = sorted(MODEL_REGISTRY.keys())


def _enum(items: list[str]) -> dict:
    return {"type": "string", "enum": items}


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "decode_tps",
        "description": "Predicted decode (output token) throughput for a model on a GPU at a given batch size and KV-cache length. Returns tokens/second across the batch and whether the kernel is compute- or memory-bound.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gpu_id": _enum(_GPU_IDS),
                "model": _enum(_MODEL_NAMES),
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 512, "default": 32},
                "kv_seq_len": {"type": "integer", "minimum": 128, "maximum": 131072, "default": 1024},
                "dtype": _enum(["fp16", "fp8", "int8", "int4"]),
                "utilisation_frac": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.65},
            },
            "required": ["gpu_id", "model"],
        },
    },
    {
        "name": "prefill_tps",
        "description": "Predicted prefill (prompt-token) throughput. Usually compute-bound; slower ramp with batch than decode.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gpu_id": _enum(_GPU_IDS),
                "model": _enum(_MODEL_NAMES),
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 512, "default": 32},
                "input_length": {"type": "integer", "minimum": 64, "maximum": 32768, "default": 512},
                "dtype": _enum(["fp16", "fp8", "int8", "int4"]),
                "utilisation_frac": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.65},
            },
            "required": ["gpu_id", "model"],
        },
    },
    {
        "name": "fits_in_memory",
        "description": "Whether a model + KV cache + activation overhead fits in a single GPU's HBM. Returns headroom in GB.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gpu_id": _enum(_GPU_IDS),
                "model": _enum(_MODEL_NAMES),
                "dtype": _enum(["fp16", "fp8", "int8", "int4"]),
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 512},
                "seq_len": {"type": "integer", "minimum": 128, "maximum": 131072},
            },
            "required": ["gpu_id", "model", "dtype", "batch_size", "seq_len"],
        },
    },
    {
        "name": "cost_per_m_tokens",
        "description": "Cost per million output tokens on a rented GPU at a given hourly rate. Amortises prefill into the output price. Returns both peak and realistic (busy_fraction-discounted) costs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gpu_id": _enum(_GPU_IDS),
                "model": _enum(_MODEL_NAMES),
                "hourly_usd": {"type": "number", "minimum": 0.0},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 512, "default": 32},
                "input_length": {"type": "integer", "minimum": 64, "maximum": 32768, "default": 512},
                "output_length": {"type": "integer", "minimum": 32, "maximum": 8192, "default": 256},
                "dtype": _enum(["fp16", "fp8", "int8", "int4"]),
                "busy_fraction": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.60},
                "utilisation_frac": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.65},
            },
            "required": ["gpu_id", "model", "hourly_usd"],
        },
    },
    {
        "name": "compare_options",
        "description": "Headline rent-vs-buy-vs-API comparison over 3 years for a workload. Returns five options (owned/reserved/on_demand/neocloud/api) with total $ and $/M tokens, plus break-evens and a 40%-demand-miss stress row for each option.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gpu_id": _enum(_GPU_IDS),
                "model": _enum(_MODEL_NAMES),
                "reserved_hourly": {"type": "number", "minimum": 0.0},
                "on_demand_hourly": {"type": "number", "minimum": 0.0},
                "neocloud_hourly": {"type": "number", "minimum": 0.0},
                "api_output_price_per_m": {"type": "number", "minimum": 0.0},
                "api_input_price_per_m": {"type": "number", "minimum": 0.0, "default": 0.0},
                "api_name": {"type": "string", "default": "API"},
                "capex_usd": {"type": "number", "minimum": 0.0, "default": 32000.0},
                "hosting_usd_per_month": {"type": "number", "minimum": 0.0, "default": 1200.0},
                "power_cost_usd_per_kwh": {"type": "number", "minimum": 0.0, "default": 0.10},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 512, "default": 32},
                "input_length": {"type": "integer", "default": 512},
                "output_length": {"type": "integer", "default": 256},
                "dtype": _enum(["fp16", "fp8", "int8", "int4"]),
                "busy_fraction": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.60},
                "demand_stress_multiplier": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 0.60},
            },
            "required": [
                "gpu_id", "model",
                "reserved_hourly", "on_demand_hourly", "neocloud_hourly",
                "api_output_price_per_m",
            ],
        },
    },
    {
        "name": "api_margin",
        "description": "Implied gross margin a provider would make if their cost base matched ours, given an advertised API $/M-tokens rate. Not an accusation — a reasoned benchmark.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gpu_id": _enum(_GPU_IDS),
                "model": _enum(_MODEL_NAMES),
                "hourly_usd": {"type": "number", "minimum": 0.0},
                "api_name": {"type": "string"},
                "api_price_per_m_tokens": {"type": "number", "minimum": 0.01},
                "batch_size": {"type": "integer", "default": 32},
                "input_length": {"type": "integer", "default": 512},
                "output_length": {"type": "integer", "default": 256},
                "dtype": _enum(["fp16", "fp8", "int8", "int4"]),
                "busy_fraction": {"type": "number", "default": 0.60},
            },
            "required": ["gpu_id", "model", "hourly_usd", "api_name", "api_price_per_m_tokens"],
        },
    },
    {
        "name": "output_tokens_per_hour",
        "description": "Expected output tokens per wall-clock hour for one GPU at a given busy fraction. Used to convert rps / demand into token volumes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gpu_id": _enum(_GPU_IDS),
                "model": _enum(_MODEL_NAMES),
                "batch_size": {"type": "integer", "default": 32},
                "kv_seq_len": {"type": "integer", "default": 1024},
                "dtype": _enum(["fp16", "fp8", "int8", "int4"]),
                "busy_fraction": {"type": "number", "default": 0.60},
            },
            "required": ["gpu_id", "model"],
        },
    },
    {
        "name": "owned_hourly_cost",
        "description": "Blended hourly $ to own one GPU: depreciation + power + hosting + financing + maintenance. Depreciation is straight-line over 3 years.",
        "input_schema": {
            "type": "object",
            "properties": {
                "capex_usd": {"type": "number", "minimum": 0.0},
                "power_draw_watts": {"type": "number", "default": 700.0},
                "power_cost_usd_per_kwh": {"type": "number", "default": 0.10},
                "pue": {"type": "number", "default": 1.4},
                "hosting_usd_per_month": {"type": "number", "default": 1200.0},
                "financing_rate_pct": {"type": "number", "default": 8.0},
                "maintenance_pct_of_capex_per_year": {"type": "number", "default": 3.0},
                "depreciation_years": {"type": "integer", "default": 3},
            },
            "required": ["capex_usd"],
        },
    },
    {
        "name": "break_even_hourly",
        "description": "The rented $/hr at which a GPU matches a target $/M-token cost (e.g. an API rate), given expected output tokens per hour.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cost_per_m_tokens_target": {"type": "number", "minimum": 0.0},
                "output_tokens_per_hr": {"type": "number", "minimum": 0.0},
            },
            "required": ["cost_per_m_tokens_target", "output_tokens_per_hr"],
        },
    },
    {
        "name": "api_only_cost",
        "description": "3-year total cost of serving a given number of output tokens via an API. Scales linearly with demand.",
        "input_schema": {
            "type": "object",
            "properties": {
                "api_output_price_per_m": {"type": "number", "minimum": 0.0},
                "tokens_demanded_3y": {"type": "number", "minimum": 0.0},
                "api_input_price_per_m": {"type": "number", "default": 0.0},
                "io_ratio": {"type": "number", "default": 2.0},
                "api_name": {"type": "string", "default": "API"},
            },
            "required": ["api_output_price_per_m", "tokens_demanded_3y"],
        },
    },
]


SYSTEM_PROMPT = """You are the compute advisor inside the AI Compute Economics Workbench.

Your job: answer sizing and build-vs-rent questions about serving open-weight LLMs.

Rules you must follow:
1. You never calculate a figure yourself. Every number in your answer must come from a tool call. If a user question cannot be answered with the available tools, say so.
2. State your assumptions explicitly: GPU, model, quantisation, batch size, input/output lengths, utilisation, hourly rates. If the user didn't give an assumption, pick a sensible default and name it.
3. Label each figure as measured, modelled, or assumption. Spec figures from the GPU catalog are measured when confidence is high, modelled otherwise. Throughput predictions from decode_tps/prefill_tps are modelled. Hourly rates are assumptions unless the user cites a source.
4. For TCO questions, call compare_options once rather than piecing together owned/reserved/api separately. It also returns the demand-stress rows.
5. If a model does not fit the GPU you were given, say so before continuing — call fits_in_memory first for any 70B-class workload.
6. Keep final answers short: a single recommendation, 3-5 bullet points of reasoning with the tool-sourced numbers, one caveat.

Output format:
- **Recommendation**: 1-2 sentences.
- **Numbers**: bulleted figures with their source (tool name) and kind label.
- **Caveats**: one line on what would change the answer.
"""


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    params: dict[str, Any]
    result: dict | None = None
    error: str | None = None


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class ComputeAdvisor:
    """Thin wrapper around the Anthropic SDK that runs the tool loop.

    Instantiate once per session so the client is reused. Pass
    ``executor=run_tool`` (the default) in production; tests can supply
    a custom executor to intercept tool dispatch.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        executor: Callable[[str, dict], dict] = run_tool,
        max_tool_calls: int = 8,
    ):
        self._api_key = api_key
        self._model = model
        self._executor = executor
        self._max_tool_calls = max_tool_calls
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise RuntimeError("anthropic SDK not installed") from e
            self._client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
        return self._client

    def ask(self, question: str) -> AgentResponse:
        """Send one user turn and run the tool loop to completion."""
        client = self._get_client()
        tool_calls: list[ToolCall] = []
        messages: list[dict] = [{"role": "user", "content": question}]

        for _ in range(self._max_tool_calls):
            resp = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            if resp.stop_reason == "end_turn":
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
                return AgentResponse(
                    text=text,
                    tool_calls=tool_calls,
                    usage={
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
                    },
                )

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for block in resp.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    call = ToolCall(name=block.name, params=dict(block.input))
                    try:
                        call.result = self._executor(block.name, call.params)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(call.result)}
                        )
                    except Exception as e:
                        call.error = str(e)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({"error": str(e)}),
                                "is_error": True,
                            }
                        )
                    tool_calls.append(call)
                messages.append({"role": "user", "content": tool_results})
                continue

            # Any other stop reason: bail with whatever text we have.
            text = "".join(
                getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
            )
            return AgentResponse(text=text, tool_calls=tool_calls)

        raise RuntimeError(
            f"Exceeded max_tool_calls={self._max_tool_calls} without end_turn. "
            "Investigate: the model is probably stuck in a tool loop."
        )
