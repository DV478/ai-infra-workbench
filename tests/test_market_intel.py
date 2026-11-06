"""Tests for the market intelligence extractor.

Every test uses ``FakeExtractorClient`` so no Anthropic API call is
made. The real client is covered only by the live eval script.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from core.market_intel import (
    FACT_TOOL_SCHEMA,
    AnthropicExtractorClient,
    FactKind,
    FakeExtractorClient,
    MarketIntelDoc,
    MarketIntelFact,
    MarketIntelStore,
    OpenAIExtractorClient,
    extract_facts,
    make_extractor_client,
)
from core.market_intel.extractor import (
    _anthropic_to_openai_tool,
    detect_backend,
)

# ---------- Backend detection (dual-backend prep) --------------------------


def test_detect_backend_anthropic():
    assert detect_backend("sk-ant-api03-xxxx") == "anthropic"


def test_detect_backend_openai():
    assert detect_backend("sk-proj-xxxx") == "openai"
    assert detect_backend("sk-xxxxx") == "openai"


def test_detect_backend_rejects_unknown():
    with pytest.raises(ValueError, match="Unrecognised"):
        detect_backend("garbage-key")


def test_detect_backend_rejects_empty():
    with pytest.raises(ValueError, match="Empty"):
        detect_backend("")

# ---------- Fixtures --------------------------------------------------------


NSCALE_BODY = """Nscale announced today that it has committed $2.5 billion to build a new
AI datacenter in Glasgow, Scotland. The facility will reach 300 MW of IT load
by the end of 2027 and will host 50,000 Nvidia H100 SXM5 GPUs under a
multi-year supply deal with Nvidia. Nscale said hourly on-demand pricing for
H100 SXM5 instances starts at $2.49 per GPU-hour.
"""


def _nscale_doc() -> MarketIntelDoc:
    return MarketIntelDoc(
        doc_type="press_release",
        title="Nscale commits $2.5B to Glasgow AI datacenter",
        body=NSCALE_BODY,
        company="nscale",
        published_date="2026-10-14",
    )


# ---------- Schema ---------------------------------------------------------


def test_fact_rejects_empty_quote():
    with pytest.raises(ValueError):
        MarketIntelFact(
            doc_id="d1",
            fact_kind=FactKind.CAPEX,
            subject="X",
            quote="",
            confidence="medium",
        )


def test_fact_rejects_empty_body():
    with pytest.raises(ValueError):
        MarketIntelDoc(doc_type="news", title="x", body="   ")


def test_verify_quote_handles_whitespace():
    fact = MarketIntelFact(
        doc_id="d1",
        fact_kind=FactKind.CAPEX,
        subject="X",
        quote="committed $2.5 billion to build a new AI datacenter in Glasgow",
        confidence="high",
    )
    assert fact.verify_quote_in(NSCALE_BODY) is True


def test_verify_quote_rejects_hallucination():
    fact = MarketIntelFact(
        doc_id="d1",
        fact_kind=FactKind.CAPEX,
        subject="X",
        quote="committed $5 billion to build a new AI datacenter in Edinburgh",
        confidence="high",
    )
    assert fact.verify_quote_in(NSCALE_BODY) is False


# ---------- Tool schema ----------------------------------------------------


def test_fact_tool_schema_mentions_every_kind():
    enum_values = FACT_TOOL_SCHEMA["input_schema"]["properties"]["facts"]["items"][
        "properties"
    ]["fact_kind"]["enum"]
    assert set(enum_values) == {k.value for k in FactKind}


# ---------- Extractor ------------------------------------------------------


def test_extract_facts_happy_path():
    client = FakeExtractorClient(
        facts=[
            {
                "fact_kind": "capex",
                "subject": "Nscale Glasgow datacenter",
                "value_num": 2_500_000_000,
                "value_unit": "usd",
                "period": "2027",
                "quote": "committed $2.5 billion to build a new AI datacenter in Glasgow, Scotland",
                "confidence": "high",
            },
            {
                "fact_kind": "capacity",
                "subject": "Nscale Glasgow datacenter",
                "value_num": 300,
                "value_unit": "mw",
                "period": "2027",
                "quote": "The facility will reach 300 MW of IT load",
                "confidence": "high",
            },
            {
                "fact_kind": "gpu_shipment",
                "subject": "H100 SXM5",
                "value_num": 50_000,
                "value_unit": "units",
                "quote": "will host 50,000 Nvidia H100 SXM5 GPUs",
                "confidence": "high",
            },
            {
                "fact_kind": "gpu_price",
                "subject": "H100 SXM5",
                "value_num": 2.49,
                "value_unit": "usd_per_hour",
                "quote": "hourly on-demand pricing for H100 SXM5 instances starts at $2.49 per GPU-hour",
                "confidence": "high",
            },
        ]
    )
    facts = extract_facts(_nscale_doc(), client)
    assert len(facts) == 4
    kinds = {f.fact_kind for f in facts}
    assert kinds == {FactKind.CAPEX, FactKind.CAPACITY, FactKind.GPU_SHIPMENT, FactKind.GPU_PRICE}
    # Company inherited from doc when LLM didn't supply one
    assert all(f.company == "nscale" for f in facts)


def test_extract_facts_drops_unverifiable_quote():
    client = FakeExtractorClient(
        facts=[
            {
                "fact_kind": "capex",
                "subject": "Nscale Glasgow datacenter",
                "value_num": 5_000_000_000,
                "value_unit": "usd",
                "quote": "committed $5 billion to build a new fab",   # not in body
                "confidence": "high",
            },
            {
                "fact_kind": "capacity",
                "subject": "Nscale Glasgow datacenter",
                "value_num": 300,
                "value_unit": "mw",
                "quote": "The facility will reach 300 MW of IT load",
                "confidence": "high",
            },
        ]
    )
    facts = extract_facts(_nscale_doc(), client)
    assert len(facts) == 1
    assert facts[0].value_num == 300


def test_extract_facts_drops_malformed_rows():
    client = FakeExtractorClient(
        facts=[
            {"fact_kind": "capex", "subject": "Nscale"},   # missing quote + confidence
            {
                "fact_kind": "capacity",
                "subject": "Glasgow",
                "quote": "will reach 300 MW of IT load",
                "confidence": "high",
            },
        ]
    )
    facts = extract_facts(_nscale_doc(), client)
    assert len(facts) == 1
    assert facts[0].fact_kind == FactKind.CAPACITY


def test_extract_facts_empty_doc_returns_empty():
    client = FakeExtractorClient(facts=[])
    assert extract_facts(_nscale_doc(), client) == []


def test_extract_facts_skip_verification():
    client = FakeExtractorClient(
        facts=[
            {
                "fact_kind": "roadmap",
                "subject": "Blackwell Ultra",
                "quote": "unverifiable snippet not in body",
                "confidence": "low",
            }
        ]
    )
    facts = extract_facts(_nscale_doc(), client, require_quote_in_body=False)
    assert len(facts) == 1


# ---------- Store ----------------------------------------------------------


def test_store_round_trip():
    store = MarketIntelStore()
    doc = _nscale_doc()
    store.add_doc(doc)

    fact = MarketIntelFact(
        doc_id=doc.doc_id,
        fact_kind=FactKind.GPU_PRICE,
        subject="H100 SXM5",
        value_num=2.49,
        value_unit="usd_per_hour",
        quote="hourly on-demand pricing for H100 SXM5 instances starts at $2.49 per GPU-hour",
        confidence="high",
        company="nscale",
    )
    store.add_fact(fact)

    assert store.count() == 1
    assert store.count(FactKind.GPU_PRICE) == 1
    assert store.count(FactKind.CAPEX) == 0

    rows = store.facts_for_subject("H100 SXM5")
    assert len(rows) == 1
    assert rows[0]["value_num"] == 2.49

    summary = store.summary_by_kind()
    assert summary == {"gpu_price": 1}

    store.close()


def test_store_add_doc_idempotent():
    store = MarketIntelStore()
    doc = _nscale_doc()
    store.add_doc(doc)
    store.add_doc(doc)
    rows = store._con.execute("SELECT COUNT(*) FROM docs").fetchone()
    assert rows[0] == 1
    store.close()


def test_store_recent_facts_newest_first():
    from datetime import UTC, datetime, timedelta

    store = MarketIntelStore()
    doc = _nscale_doc()
    store.add_doc(doc)

    t0 = datetime.now(UTC)
    older = MarketIntelFact(
        doc_id=doc.doc_id,
        fact_kind=FactKind.CAPEX,
        subject="A",
        quote="A quote",
        confidence="medium",
        extracted_at=t0,
    )
    newer = MarketIntelFact(
        doc_id=doc.doc_id,
        fact_kind=FactKind.CAPEX,
        subject="B",
        quote="B quote",
        confidence="medium",
        extracted_at=t0 + timedelta(seconds=1),
    )
    store.add_fact(older)
    store.add_fact(newer)

    rows = store.recent_facts(limit=2)
    assert [r["subject"] for r in rows] == ["B", "A"]
    store.close()


# ---------- End-to-end: extract + persist ----------------------------------


# ---------- Dual-backend: factory + OpenAIExtractorClient -----------------


def test_tool_schema_converter_shape():
    openai_tool = _anthropic_to_openai_tool(FACT_TOOL_SCHEMA)
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "emit_facts"
    # Parameters must carry the Anthropic input_schema verbatim.
    assert openai_tool["function"]["parameters"] is FACT_TOOL_SCHEMA["input_schema"]


def _install_fake_openai_module(monkeypatch, response_facts: list[dict]) -> MagicMock:
    """Inject a fake `openai` module so OpenAIExtractorClient works offline.

    Returns the MagicMock standing in for ``OpenAI(...)`` instances so tests
    can inspect call arguments.
    """
    fake_client = MagicMock()
    fake_resp = MagicMock()
    tc = MagicMock()
    tc.function.name = "emit_facts"
    tc.function.arguments = json.dumps({"facts": response_facts})
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.tool_calls = [tc]
    fake_client.chat.completions.create.return_value = fake_resp

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return fake_client


def test_openai_extractor_emits_facts(monkeypatch):
    fake = _install_fake_openai_module(
        monkeypatch,
        response_facts=[
            {
                "fact_kind": "capex",
                "subject": "Nscale Glasgow",
                "value_num": 2_500_000_000,
                "value_unit": "usd",
                "quote": "committed $2.5 billion to build a new AI datacenter in Glasgow",
                "confidence": "high",
            }
        ],
    )
    client = OpenAIExtractorClient(api_key="sk-test", model="gpt-4o-mini")
    facts = client.emit(_nscale_doc())
    assert len(facts) == 1
    assert facts[0]["fact_kind"] == "capex"

    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "emit_facts"},
    }
    # System prompt + user prompt order, plus the tool schema wrapped
    # for OpenAI.
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["tools"][0]["type"] == "function"


def test_openai_extractor_returns_empty_when_no_tool_call(monkeypatch):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.tool_calls = None
    fake_client.chat.completions.create.return_value = fake_resp
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    client = OpenAIExtractorClient(api_key="sk-test")
    assert client.emit(_nscale_doc()) == []


def test_factory_picks_anthropic_for_sk_ant_key():
    client = make_extractor_client(api_key="sk-ant-api03-test")
    assert isinstance(client, AnthropicExtractorClient)


def test_factory_picks_openai_for_sk_key():
    client = make_extractor_client(api_key="sk-proj-test")
    assert isinstance(client, OpenAIExtractorClient)


def test_factory_reads_env_when_api_key_not_passed(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
    client = make_extractor_client()
    assert isinstance(client, OpenAIExtractorClient)


def test_factory_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        make_extractor_client()


def test_factory_round_trip_with_openai_backend(monkeypatch):
    """End-to-end: factory → OpenAIExtractorClient → extract_facts → verified."""
    _install_fake_openai_module(
        monkeypatch,
        response_facts=[
            {
                "fact_kind": "capacity",
                "subject": "Nscale Glasgow",
                "value_num": 300,
                "value_unit": "mw",
                "quote": "The facility will reach 300 MW of IT load",
                "confidence": "high",
            },
            {
                # Hallucinated — must be dropped by extract_facts.
                "fact_kind": "capex",
                "subject": "Nscale Glasgow",
                "value_num": 5_000_000_000,
                "value_unit": "usd",
                "quote": "committed $5 billion to a fab in Edinburgh",
                "confidence": "high",
            },
        ],
    )
    client = make_extractor_client(api_key="sk-proj-test")
    assert isinstance(client, OpenAIExtractorClient)

    facts = extract_facts(_nscale_doc(), client)
    assert len(facts) == 1
    assert facts[0].fact_kind == FactKind.CAPACITY


def test_end_to_end_extract_and_persist():
    doc = _nscale_doc()
    client = FakeExtractorClient(
        callback=lambda d: [
            {
                "fact_kind": "capex",
                "subject": f"{d.company} Glasgow",
                "value_num": 2_500_000_000,
                "value_unit": "usd",
                "quote": "committed $2.5 billion to build a new AI datacenter in Glasgow",
                "confidence": "high",
            }
        ]
    )
    facts = extract_facts(doc, client)
    assert len(facts) == 1
    assert facts[0].subject == "nscale Glasgow"

    store = MarketIntelStore()
    store.add_doc(doc)
    store.add_facts(facts)
    assert store.count(FactKind.CAPEX) == 1
    store.close()
