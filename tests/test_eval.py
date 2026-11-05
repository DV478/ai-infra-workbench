"""Tests for the eval harness.

The harness grades synthetic ``AgentResponse`` objects — we don't call
the Anthropic API here. Separate end-to-end evals live under
``scripts/run_eval.py`` (not in CI; needs an API key).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.agents.compute_advisor import run_tool
from core.eval.harness import grade, run_eval_set, summarise
from core.eval.questions import EVAL_QUESTIONS


# Local stand-ins for the AgentResponse / ToolCall dataclasses so this
# file has no runtime dependency on the agent class.
@dataclass
class _Call:
    name: str
    params: dict
    result: dict | None = None


@dataclass
class _Resp:
    text: str = ""
    tool_calls: list[_Call] = field(default_factory=list)


# ---------- Guardrail questions --------------------------------------------


def test_guardrail_question_requires_no_tool_call():
    q = next(q for q in EVAL_QUESTIONS if q.id == "out_of_scope_training")
    good = _Resp(text="Training is outside the scope of this advisor.")
    bad_tool = _Resp(text="Training scope...", tool_calls=[_Call("decode_tps", {})])
    bad_text = _Resp(text="4000 GPU-hours.")
    assert grade(q, good).passed is True
    assert grade(q, bad_tool).passed is False
    assert grade(q, bad_text).passed is False


# ---------- Tool-based questions -------------------------------------------


def test_tool_question_fails_without_tool_call():
    q = next(q for q in EVAL_QUESTIONS if q.id == "fit_70b_fp16_h100")
    resp = _Resp(text="Does not fit.")
    r = grade(q, resp)
    assert r.passed is False
    assert any("not called" in reason for reason in r.reasons)


def test_tool_question_passes_with_matching_params():
    q = next(q for q in EVAL_QUESTIONS if q.id == "fit_70b_fp16_h100")
    params = {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "dtype": "fp16", "batch_size": 1, "seq_len": 2048}
    result = run_tool("fits_in_memory", params)
    resp = _Resp(text="This does NOT fit.", tool_calls=[_Call("fits_in_memory", params, result)])
    r = grade(q, resp)
    assert r.tool_called is True
    assert r.params_matched is True
    assert r.passed is True
    # Numeric check should also pass: 140GB weights > 80GB → very negative headroom
    assert r.numeric_in_range is True


def test_tool_question_fails_on_wrong_param():
    q = next(q for q in EVAL_QUESTIONS if q.id == "fit_70b_fp16_h100")
    params = {"gpu_id": "a100_sxm_80gb", "model": "llama-3-70b", "dtype": "fp16", "batch_size": 1, "seq_len": 2048}
    result = run_tool("fits_in_memory", params)
    resp = _Resp(text="Does not fit.", tool_calls=[_Call("fits_in_memory", params, result)])
    r = grade(q, resp)
    assert r.passed is False
    assert r.params_matched is False
    assert any("gpu_id" in reason for reason in r.reasons)


def test_case_insensitive_string_match():
    q = next(q for q in EVAL_QUESTIONS if q.id == "fit_70b_fp16_h100")
    # Agent returns gpu_id capitalised; should still pass.
    params = {"gpu_id": "H100_SXM5_80GB", "model": "LLAMA-3-70B", "dtype": "FP16", "batch_size": 1, "seq_len": 2048}
    # Can't run_tool with uppercase model (resolver is case-insensitive too)
    result = run_tool("fits_in_memory", {k: (v.lower() if isinstance(v, str) else v) for k, v in params.items()})
    resp = _Resp(text="Does not fit.", tool_calls=[_Call("fits_in_memory", params, result)])
    r = grade(q, resp)
    assert r.params_matched is True


# ---------- Numeric range checks -------------------------------------------


def test_numeric_out_of_range_still_passes_structurally():
    """Numeric is informational — right tool with right params still passes.
    The point is to catch tool-dispatch failures; roofline error is expected."""
    q = next(q for q in EVAL_QUESTIONS if q.id == "decode_8b_h100_b1")
    params = {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "batch_size": 1, "kv_seq_len": 1024}
    # Fake a result outside the expected range
    fake_result = {"tokens_per_second": 1.0, "bound": "memory", "arithmetic_intensity": 2.0}
    resp = _Resp(text="Memory bound.", tool_calls=[_Call("decode_tps", params, fake_result)])
    r = grade(q, resp)
    assert r.passed is True  # tool + params correct
    assert r.numeric_in_range is False  # but the number was wrong
    assert any("outside" in reason for reason in r.reasons)


# ---------- End-to-end dry run --------------------------------------------


def test_run_eval_set_with_stub_ask():
    """Synthetic ask_fn that always returns an empty response; every question
    should fail. Tests that the harness runs without crashing on any
    question in the set."""
    def ask_empty(_q: str) -> _Resp:
        return _Resp(text="")

    results = run_eval_set(EVAL_QUESTIONS, ask_empty)
    assert len(results) == len(EVAL_QUESTIONS)
    summary = summarise(results)
    assert summary["n"] == 20
    assert summary["passed"] == 0
    assert summary["pass_rate_pct"] == 0.0


def test_run_eval_set_with_perfect_stub():
    """Synthetic ask_fn that calls the expected tool with the expected params
    for each question. Every tool-based question should pass. Guardrail
    questions pass if we leave tool_calls empty and include "scope"."""
    def ask_perfect(question_text: str) -> _Resp:
        for q in EVAL_QUESTIONS:
            if q.question == question_text:
                if q.expected_tool is None:
                    return _Resp(text="This is outside the scope of the advisor.")
                # Build minimally viable params for the tool.
                params = dict(q.expected_params)
                _fill_defaults(params, q.expected_tool)
                try:
                    result = run_tool(q.expected_tool, params)
                except Exception as e:
                    result = {"error": str(e)}
                # Include common expected text markers.
                text = "scope " + " ".join(q.expected_text_contains) + " fit memory api double"
                return _Resp(text=text, tool_calls=[_Call(q.expected_tool, params, result)])
        raise AssertionError(f"Unknown question: {question_text}")

    results = run_eval_set(EVAL_QUESTIONS, ask_perfect)
    summary = summarise(results)
    # All 20 should pass with a perfect stub
    assert summary["pass_rate_pct"] >= 95, f"Perfect stub only hit {summary['pass_rate_pct']}%: {summary['failures']}"


def _fill_defaults(params: dict, tool_name: str) -> None:
    """Fill the minimum required params for a tool so run_tool works."""
    defaults = {
        "fits_in_memory": {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "dtype": "fp16", "batch_size": 32, "seq_len": 2048},
        "decode_tps": {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b"},
        "prefill_tps": {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b"},
        "cost_per_m_tokens": {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "hourly_usd": 3.0},
        "compare_options": {
            "gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b",
            "reserved_hourly": 2.5, "on_demand_hourly": 4.5, "neocloud_hourly": 2.0,
            "api_output_price_per_m": 2.0,
        },
        "api_margin": {
            "gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "hourly_usd": 3.0,
            "api_name": "API", "api_price_per_m_tokens": 2.0,
        },
        "output_tokens_per_hour": {"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b"},
        "owned_hourly_cost": {"capex_usd": 32000.0},
        "break_even_hourly": {"cost_per_m_tokens_target": 2.0, "output_tokens_per_hr": 1_000_000.0},
        "api_only_cost": {"api_output_price_per_m": 2.0, "tokens_demanded_3y": 1e9},
    }
    for k, v in defaults.get(tool_name, {}).items():
        params.setdefault(k, v)


# ---------- Question set validation ----------------------------------------


def test_eval_set_has_exactly_20():
    assert len(EVAL_QUESTIONS) == 20


def test_every_expected_tool_exists_in_registry():
    for q in EVAL_QUESTIONS:
        if q.expected_tool is not None:
            from core.agents.compute_advisor import TOOL_REGISTRY
            assert q.expected_tool in TOOL_REGISTRY, (
                f"Question {q.id} references unknown tool {q.expected_tool!r}"
            )


def test_every_question_has_unique_id():
    ids = [q.id for q in EVAL_QUESTIONS]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {[i for i in ids if ids.count(i) > 1]}"
