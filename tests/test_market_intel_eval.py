"""Tests for the market-intelligence gold-set grader.

Covers:
- Matching logic (kind + subject + value tolerance)
- Recall / precision math
- Quote validity enforcement
- Guardrail case (empty expected, empty actual)
- Gold-set sanity: every case's expected facts are internally consistent
  and the fake extractor floor is non-zero
"""

from __future__ import annotations

from core.market_intel import (
    GOLD_CASES,
    EvalCase,
    ExpectedFact,
    FactKind,
    MarketIntelDoc,
    MarketIntelFact,
    grade_case,
    summarise,
)


def _doc(body: str) -> MarketIntelDoc:
    return MarketIntelDoc(
        doc_type="news", title="t", body=body, doc_id="t1"
    )


def _fact(body: str, **kw) -> MarketIntelFact:
    defaults = {"doc_id": "t1", "subject": "X", "quote": body[:40], "confidence": "high"}
    defaults.update(kw)
    return MarketIntelFact(**defaults)


# ---------- Matching ------------------------------------------------------


def test_matches_same_kind_and_subject_keyword():
    doc = _doc("Nscale committed $2.5 billion to Glasgow")
    case = EvalCase(
        id="c",
        doc=doc,
        expected=[ExpectedFact(FactKind.CAPEX, "nscale", value_num=2_500_000_000)],
    )
    actual = [
        _fact(doc.body, fact_kind=FactKind.CAPEX, subject="Nscale Glasgow datacenter",
              value_num=2_500_000_000, value_unit="usd",
              quote="Nscale committed $2.5 billion to Glasgow")
    ]
    r = grade_case(case, actual)
    assert r.passed
    assert r.recall_pct == 100.0
    assert r.precision_pct == 100.0


def test_misses_on_wrong_kind():
    doc = _doc("Nscale committed $2.5 billion to Glasgow")
    case = EvalCase(
        id="c",
        doc=doc,
        expected=[ExpectedFact(FactKind.CAPEX, "nscale")],
    )
    actual = [
        _fact(doc.body, fact_kind=FactKind.CAPACITY, subject="Nscale",
              value_num=300, value_unit="mw",
              quote="Nscale committed $2.5 billion to Glasgow")
    ]
    r = grade_case(case, actual)
    assert not r.passed
    assert r.recall_pct == 0.0


def test_misses_on_subject_keyword_mismatch():
    doc = _doc("Nscale committed $2.5 billion to Glasgow")
    case = EvalCase(
        id="c",
        doc=doc,
        expected=[ExpectedFact(FactKind.CAPEX, "edinburgh")],   # body has Glasgow, not Edinburgh
    )
    actual = [
        _fact(doc.body, fact_kind=FactKind.CAPEX, subject="Glasgow",
              value_num=2_500_000_000,
              quote="Nscale committed $2.5 billion to Glasgow")
    ]
    r = grade_case(case, actual)
    assert not r.passed


def test_value_tolerance_respected():
    body = "Company spent $100 million on compute"
    doc = _doc(body)
    case = EvalCase(
        id="c", doc=doc,
        expected=[ExpectedFact(FactKind.CAPEX, "company", value_num=100_000_000, value_tolerance_pct=10.0)],
    )
    # within 10%
    close = [_fact(body, fact_kind=FactKind.CAPEX, subject="Company Y",
                   value_num=105_000_000, quote=body)]
    assert grade_case(case, close).passed
    # 15% off — should miss
    far = [_fact(body, fact_kind=FactKind.CAPEX, subject="Company Y",
                 value_num=115_000_000, quote=body)]
    assert not grade_case(case, far).passed


def test_missing_value_when_expected_fails_match():
    body = "Some text"
    case = EvalCase(
        id="c", doc=_doc(body),
        expected=[ExpectedFact(FactKind.CAPEX, "x", value_num=1_000_000)],
    )
    # actual fact has no value_num
    actual = [_fact(body, fact_kind=FactKind.CAPEX, subject="x",
                    value_num=None, quote=body)]
    r = grade_case(case, actual)
    assert not r.passed


# ---------- Quote validity ------------------------------------------------


def test_invalid_quote_fails_case():
    body = "Nscale committed $2.5 billion to Glasgow"
    case = EvalCase(
        id="c", doc=_doc(body),
        expected=[ExpectedFact(FactKind.CAPEX, "nscale")],
    )
    actual = [
        _fact(body, fact_kind=FactKind.CAPEX, subject="Nscale",
              value_num=2_500_000_000,
              quote="This phrase does not appear in the body at all"),
    ]
    r = grade_case(case, actual)
    assert r.quote_valid_rate_pct == 0.0
    assert not r.passed


def test_quote_valid_handles_whitespace_variance():
    body = "The  facility  reaches  300  MW  of  load"
    case = EvalCase(
        id="c", doc=_doc(body),
        expected=[],
    )
    actual = [
        _fact(body, fact_kind=FactKind.CAPACITY, subject="Site",
              value_num=300, value_unit="mw",
              quote="The facility reaches 300 MW of load"),   # whitespace normalised
    ]
    r = grade_case(case, actual)
    assert r.quote_valid_rate_pct == 100.0


# ---------- Guardrail + spurious facts ------------------------------------


def test_guardrail_empty_expected_empty_actual_passes():
    doc = _doc("Nothing to see here. Just an opinion.")
    case = EvalCase(id="c", doc=doc, expected=[])
    r = grade_case(case, [])
    assert r.passed
    assert r.recall_pct == 100.0


def test_guardrail_spurious_facts_lower_precision_but_still_pass_if_quote_valid():
    # Pass criterion ignores spurious — only recall + quote-valid matter.
    # A spurious fact with a valid quote lowers precision but should not
    # fail the case on its own (the extractor over-pulling is logged, not punished).
    body = "Company committed $100 million."
    case = EvalCase(id="c", doc=_doc(body), expected=[])
    actual = [
        _fact(body, fact_kind=FactKind.CAPEX, subject="Company",
              value_num=100_000_000, quote=body),
    ]
    r = grade_case(case, actual)
    # recall is 100% (no expected facts missed), quote valid, so pass
    assert r.passed
    assert r.precision_pct == 0.0
    assert len(r.spurious) == 1


# ---------- Gold set --------------------------------------------------------


def test_gold_set_covers_every_fact_kind():
    expected_kinds = {e.fact_kind for c in GOLD_CASES for e in c.expected}
    # Confirm at least one case tests each of the six kinds
    assert FactKind.CAPEX in expected_kinds
    assert FactKind.CAPACITY in expected_kinds
    assert FactKind.GPU_PRICE in expected_kinds
    assert FactKind.GPU_SHIPMENT in expected_kinds
    assert FactKind.CONTRACT in expected_kinds
    assert FactKind.ROADMAP in expected_kinds


def test_gold_set_ids_unique():
    ids = [c.id for c in GOLD_CASES]
    assert len(ids) == len(set(ids))


def test_gold_set_has_guardrail_case():
    guardrails = [c for c in GOLD_CASES if not c.expected]
    assert len(guardrails) >= 1, "Need at least one guardrail case (zero expected facts)"


# ---------- Summary --------------------------------------------------------


def test_summarise_empty():
    assert summarise([]) == {"n": 0}


def test_summarise_counts_and_averages():
    from core.market_intel import CaseResult
    results = [
        CaseResult(case_id="a", recall_pct=100, precision_pct=100, quote_valid_rate_pct=100, passed=True),
        CaseResult(case_id="b", recall_pct=50, precision_pct=66.7, quote_valid_rate_pct=100, passed=False,
                   reasons=["recall 50% below 70% threshold"]),
    ]
    s = summarise(results)
    assert s["n"] == 2
    assert s["passed"] == 1
    assert s["pass_rate_pct"] == 50.0
    assert s["avg_recall_pct"] == 75.0
    assert len(s["failures"]) == 1
    assert s["failures"][0]["id"] == "b"
