"""Grading logic for the eval set.

The harness takes an ``AgentResponse`` and an ``EvalQuestion`` and returns
pass/fail plus a breakdown so failures are debuggable:

- Was the expected tool called? (``tool_called``)
- Did its params match the expected params? (``params_matched``)
- Does the numeric result fall in the expected range? (``numeric_in_range``)
- Does the final text contain the expected substrings? (``text_contains_all``)

A question passes if ``tool_called AND params_matched`` for tool-based
questions, OR ``text_contains_all`` for guardrail questions (which have
``expected_tool=None``). The numeric check is informational — the roofline
is a model with 15-25% error, and the point of the harness is to catch
wrong-tool / wrong-param failures, not to verify the physics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.eval.questions import EvalQuestion


@dataclass
class EvalResult:
    question_id: str
    passed: bool
    tool_called: bool = False
    params_matched: bool = False
    numeric_in_range: bool | None = None
    text_contains_all: bool | None = None
    actual_tool: str | None = None
    actual_params: dict | None = None
    actual_numeric: float | None = None
    reasons: list[str] = field(default_factory=list)


def _params_match(actual: dict, expected: dict) -> tuple[bool, list[str]]:
    """Every key in expected must be present in actual with an equal value.

    Strings are compared case-insensitively (the model sometimes capitalises
    gpu_id). Numerics must be exact; the agent is supposed to use the
    numbers from the question verbatim.
    """
    reasons: list[str] = []
    for k, exp in expected.items():
        if k not in actual:
            reasons.append(f"missing param {k!r}")
            continue
        a = actual[k]
        if isinstance(exp, str) and isinstance(a, str):
            if a.lower() != exp.lower():
                reasons.append(f"param {k}: expected {exp!r}, got {a!r}")
        elif a != exp:
            reasons.append(f"param {k}: expected {exp!r}, got {a!r}")
    return not reasons, reasons


def grade(question: EvalQuestion, response) -> EvalResult:
    """Grade one agent response against one question.

    ``response`` is a ``core.agents.AgentResponse`` (duck-typed to avoid a
    circular import when the harness is loaded standalone).
    """
    result = EvalResult(question_id=question.id, passed=False)

    # Guardrail-style question: no tool should be called, text must contain markers.
    if question.expected_tool is None:
        if question.expected_text_contains:
            text_lower = response.text.lower()
            missing = [s for s in question.expected_text_contains if s.lower() not in text_lower]
            result.text_contains_all = not missing
            if missing:
                result.reasons.append(f"text missing: {missing}")
        # Should not have called any heavy calculator tool.
        if response.tool_calls:
            result.reasons.append(f"guardrail question called tools: {[c.name for c in response.tool_calls]}")
        result.passed = (result.text_contains_all or False) and not response.tool_calls
        return result

    # Tool-use question: expected tool must have been called.
    matching = [c for c in response.tool_calls if c.name == question.expected_tool]
    if not matching:
        result.reasons.append(f"expected tool {question.expected_tool!r} not called")
        result.actual_tool = response.tool_calls[0].name if response.tool_calls else None
        return result

    result.tool_called = True
    # Pick the first matching call.
    call = matching[0]
    result.actual_tool = call.name
    result.actual_params = call.params

    ok, reasons = _params_match(call.params, question.expected_params)
    result.params_matched = ok
    result.reasons.extend(reasons)

    if question.expected_numeric is not None and call.result is not None:
        key, lo, hi = question.expected_numeric
        val = call.result.get(key)
        if val is None:
            result.reasons.append(f"result missing key {key!r}")
            result.numeric_in_range = False
        else:
            result.actual_numeric = float(val)
            result.numeric_in_range = lo <= float(val) <= hi
            if not result.numeric_in_range:
                result.reasons.append(f"{key}={val} outside [{lo}, {hi}]")

    if question.expected_text_contains:
        text_lower = response.text.lower()
        missing = [s for s in question.expected_text_contains if s.lower() not in text_lower]
        result.text_contains_all = not missing
        if missing:
            result.reasons.append(f"text missing: {missing}")

    # Pass criteria: right tool + right params. Numeric + text are informational.
    result.passed = result.tool_called and result.params_matched
    return result


def run_eval_set(
    questions: list[EvalQuestion],
    ask_fn: Callable[[str], object],
) -> list[EvalResult]:
    """Run every question through ``ask_fn`` and grade each response.

    ``ask_fn(question_text) -> AgentResponse``. Pass the agent's ``.ask`` method
    in production, or a mock in tests.
    """
    return [grade(q, ask_fn(q.question)) for q in questions]


def summarise(results: list[EvalResult]) -> dict:
    """Pass-rate + per-component breakdown."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    passed = sum(1 for r in results if r.passed)
    tool_called = sum(1 for r in results if r.tool_called)
    params_matched = sum(1 for r in results if r.params_matched)
    numeric_ok = sum(1 for r in results if r.numeric_in_range)
    numeric_evaluated = sum(1 for r in results if r.numeric_in_range is not None)
    return {
        "n": n,
        "passed": passed,
        "pass_rate_pct": round(100 * passed / n, 1),
        "tool_called_rate_pct": round(100 * tool_called / n, 1),
        "params_matched_rate_pct": round(100 * params_matched / n, 1),
        "numeric_in_range_rate_pct": (
            round(100 * numeric_ok / numeric_evaluated, 1) if numeric_evaluated else None
        ),
        "failures": [{"id": r.question_id, "reasons": r.reasons} for r in results if not r.passed],
    }
