"""Gold-set evaluation for the market intelligence extractor.

Mirrors ``core/eval/harness.py`` (the compute-advisor eval) but graded
on a different shape: one doc produces N facts, so we measure recall
(did we find every expected fact?), precision (did we invent any?),
and quote-validity (does every fact's quote appear verbatim in the
source body?).

A case specifies the minimum facts the extractor must emit from a doc.
Matching is deliberately loose on strings — `subject_keyword` is a
case-insensitive substring — and tight on numbers (±10% by default).
This matches how a human reviewer judges: "a capex fact about Nscale
Glasgow for roughly $2.5B" is a match for "Nscale Glasgow datacenter —
$2.5 billion", even if the subject strings differ.

Pass criterion: recall ≥ 70% AND quote-validity = 100%. False positives
are reported but don't fail the case on their own — an extractor that
over-pulls is less dangerous than one that misses or hallucinates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.market_intel.schema import FactKind, MarketIntelDoc, MarketIntelFact

# ---------------------------------------------------------------------------
# Case + Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExpectedFact:
    """Minimum spec an actual fact must satisfy to count as a match.

    - ``fact_kind``: exact match required.
    - ``subject_keyword``: must appear (case-insensitive) in actual subject.
    - ``value_num``/``value_tolerance_pct``: if ``value_num`` is set, actual
      value must be within ±tolerance%. Default tolerance 10%.
    """

    fact_kind: FactKind
    subject_keyword: str
    value_num: float | None = None
    value_unit: str | None = None
    value_tolerance_pct: float = 10.0


@dataclass
class EvalCase:
    id: str
    doc: MarketIntelDoc
    expected: list[ExpectedFact] = field(default_factory=list)
    notes: str | None = None


@dataclass
class CaseResult:
    case_id: str
    recall_pct: float
    precision_pct: float
    quote_valid_rate_pct: float
    passed: bool
    matched: list[tuple[ExpectedFact, MarketIntelFact]] = field(default_factory=list)
    missed: list[ExpectedFact] = field(default_factory=list)
    spurious: list[MarketIntelFact] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Matching + grading
# ---------------------------------------------------------------------------


def _fact_matches(exp: ExpectedFact, actual: MarketIntelFact) -> bool:
    if actual.fact_kind != exp.fact_kind:
        return False
    if exp.subject_keyword.casefold() not in actual.subject.casefold():
        return False
    if exp.value_num is not None:
        if actual.value_num is None:
            return False
        if exp.value_num == 0:
            if abs(actual.value_num) > 1e-6:
                return False
        else:
            err = abs(actual.value_num - exp.value_num) / abs(exp.value_num) * 100
            if err > exp.value_tolerance_pct:
                return False
    return True


def grade_case(case: EvalCase, actual_facts: list[MarketIntelFact]) -> CaseResult:
    matched: list[tuple[ExpectedFact, MarketIntelFact]] = []
    unmatched_actual = list(actual_facts)
    missed: list[ExpectedFact] = []

    for exp in case.expected:
        found: MarketIntelFact | None = None
        for a in unmatched_actual:
            if _fact_matches(exp, a):
                found = a
                break
        if found is not None:
            matched.append((exp, found))
            unmatched_actual.remove(found)
        else:
            missed.append(exp)

    n_expected = len(case.expected)
    n_actual = len(actual_facts)
    recall = (len(matched) / n_expected * 100) if n_expected else 100.0
    precision = (len(matched) / n_actual * 100) if n_actual else 100.0

    quote_valid = sum(1 for a in actual_facts if a.verify_quote_in(case.doc.body))
    quote_rate = (quote_valid / n_actual * 100) if n_actual else 100.0

    passed = recall >= 70.0 and quote_rate >= 100.0
    reasons: list[str] = []
    if recall < 70.0:
        reasons.append(f"recall {recall:.0f}% below 70% threshold")
    if quote_rate < 100.0:
        reasons.append(f"{n_actual - quote_valid}/{n_actual} facts have invalid quotes")

    return CaseResult(
        case_id=case.id,
        recall_pct=round(recall, 1),
        precision_pct=round(precision, 1),
        quote_valid_rate_pct=round(quote_rate, 1),
        passed=passed,
        matched=matched,
        missed=missed,
        spurious=unmatched_actual,
        reasons=reasons,
    )


def run_eval_set(
    cases: list[EvalCase],
    extract_fn: Callable[[MarketIntelDoc], list[MarketIntelFact]],
) -> list[CaseResult]:
    """Run each case through ``extract_fn`` and grade.

    ``extract_fn(doc) -> list[MarketIntelFact]``. In production, bind a
    partial over ``core.market_intel.extract_facts`` with a live
    ``AnthropicExtractorClient``.
    """
    return [grade_case(c, extract_fn(c.doc)) for c in cases]


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    passed = sum(1 for r in results if r.passed)
    avg_recall = sum(r.recall_pct for r in results) / n
    avg_precision = sum(r.precision_pct for r in results) / n
    avg_quote = sum(r.quote_valid_rate_pct for r in results) / n
    return {
        "n": n,
        "passed": passed,
        "pass_rate_pct": round(100 * passed / n, 1),
        "avg_recall_pct": round(avg_recall, 1),
        "avg_precision_pct": round(avg_precision, 1),
        "avg_quote_valid_rate_pct": round(avg_quote, 1),
        "failures": [{"id": r.case_id, "reasons": r.reasons} for r in results if not r.passed],
    }


# ---------------------------------------------------------------------------
# Gold set — 8 compact docs covering all six fact kinds plus guardrails.
# Deliberately written to span easy / hedged / guardrail difficulty so a
# single pass-rate number reflects real-world extractor behaviour.
# ---------------------------------------------------------------------------


def _doc(doc_id: str, **kw) -> MarketIntelDoc:
    return MarketIntelDoc(doc_id=doc_id, **kw)


GOLD_CASES: list[EvalCase] = [
    EvalCase(
        id="capex-simple",
        doc=_doc(
            doc_id="gold-01",
            doc_type="press_release",
            title="Nscale commits $2.5B to Glasgow",
            body=(
                "Nscale announced today that it has committed $2.5 billion to build a new AI "
                "datacenter in Glasgow, Scotland. The facility will reach 300 MW of IT load "
                "by the end of 2027."
            ),
            company="nscale",
            published_date="2026-10-14",
        ),
        expected=[
            ExpectedFact(FactKind.CAPEX, "nscale", value_num=2_500_000_000, value_unit="usd"),
            ExpectedFact(FactKind.CAPACITY, "glasgow", value_num=300, value_unit="mw"),
        ],
        notes="Baseline: one capex and one capacity in a clean paragraph.",
    ),
    EvalCase(
        id="gpu-price-simple",
        doc=_doc(
            doc_id="gold-02",
            doc_type="press_release",
            title="Nebius H200 availability",
            body=(
                "Nebius Group announced general availability of H200 SXM 141GB clusters. "
                "On-demand pricing is $3.49 per GPU-hour; 1-year reserved contracts start "
                "at $2.19 per GPU-hour."
            ),
            company="nebius",
            published_date="2026-09-30",
        ),
        expected=[
            ExpectedFact(FactKind.GPU_PRICE, "h200", value_num=3.49, value_unit="usd_per_hour"),
            ExpectedFact(FactKind.GPU_PRICE, "h200", value_num=2.19, value_unit="usd_per_hour"),
        ],
        notes="Two prices for the same subject with different commitment windows.",
    ),
    EvalCase(
        id="shipment-count",
        doc=_doc(
            doc_id="gold-03",
            doc_type="earnings_transcript",
            title="Hyperscaler Q3 call excerpt",
            body=(
                "During the quarter we took delivery of 48,000 H100 SXM5 GPUs, "
                "bringing our total installed base to 215,000 Hopper-class accelerators."
            ),
            company="cloudco",
            published_date="2026-10-28",
        ),
        expected=[
            ExpectedFact(FactKind.GPU_SHIPMENT, "h100", value_num=48000, value_unit="units"),
        ],
        notes="Explicit shipment; 'total installed base' is a different kind and not expected.",
    ),
    EvalCase(
        id="contract-named",
        doc=_doc(
            doc_id="gold-04",
            doc_type="press_release",
            title="CoreAI signs $800M compute deal with Nvidia",
            body=(
                "CoreAI announced an $800 million three-year compute supply agreement with "
                "Nvidia covering 20,000 Blackwell B200 GPUs delivered through 2028."
            ),
            company="coreai",
            published_date="2026-11-01",
        ),
        expected=[
            ExpectedFact(FactKind.CONTRACT, "nvidia", value_num=800_000_000, value_unit="usd"),
            ExpectedFact(FactKind.GPU_SHIPMENT, "b200", value_num=20_000, value_unit="units"),
        ],
        notes="Contract + shipment in one sentence; extractor should produce both kinds.",
    ),
    EvalCase(
        id="hedged-numbers",
        doc=_doc(
            doc_id="gold-05",
            doc_type="news",
            title="Startup plans roughly $500M buildout",
            body=(
                "The company said it expects to spend approximately $500 million over the "
                "next two years on datacenter buildout, with up to 40 MW of initial capacity."
            ),
            company="startupco",
            published_date="2026-08-20",
        ),
        expected=[
            ExpectedFact(
                FactKind.CAPEX, "startup", value_num=500_000_000, value_unit="usd",
                value_tolerance_pct=15.0,
            ),
            ExpectedFact(
                FactKind.CAPACITY, "capacity", value_num=40, value_unit="mw",
                value_tolerance_pct=15.0,
            ),
        ],
        notes="Hedged language — extractor should still emit the figures with low confidence.",
    ),
    EvalCase(
        id="roadmap-ship-date",
        doc=_doc(
            doc_id="gold-06",
            doc_type="blog",
            title="Rubin GA pushed to H1 2027",
            body=(
                "Nvidia confirmed Rubin GPUs will reach general availability in Q1 2027, "
                "with sampling to select customers beginning December 2026."
            ),
            company="nvidia",
            published_date="2026-10-02",
        ),
        expected=[
            ExpectedFact(FactKind.ROADMAP, "rubin"),
        ],
        notes="Date-only roadmap fact; no numeric value required to match.",
    ),
    EvalCase(
        id="guardrail-no-facts",
        doc=_doc(
            doc_id="gold-07",
            doc_type="news",
            title="Analyst opinion piece",
            body=(
                "In our view, AI infrastructure spend will continue to dominate hyperscaler "
                "capex narratives into next year. Investors should watch for signs of "
                "margin pressure as supply catches up with demand."
            ),
            company="analyst",
            published_date="2026-11-03",
        ),
        expected=[],
        notes="No concrete numbers or commitments — extractor should emit zero facts.",
    ),
    EvalCase(
        id="distractor-non-facts",
        doc=_doc(
            doc_id="gold-08",
            doc_type="press_release",
            title="FoundryCo expands Austin team",
            body=(
                "FoundryCo, founded in 2019 with 42 employees, announced plans to grow "
                "headcount to 180 by 2027. The company also confirmed it has committed "
                "$75 million to expand its Austin campus, adding 15 MW of capacity."
            ),
            company="foundryco",
            published_date="2026-10-30",
        ),
        expected=[
            ExpectedFact(FactKind.CAPEX, "austin", value_num=75_000_000, value_unit="usd"),
            ExpectedFact(FactKind.CAPACITY, "austin", value_num=15, value_unit="mw"),
        ],
        notes="Distractor numbers (2019, 42 employees, 180 by 2027) should NOT be emitted.",
    ),
]
