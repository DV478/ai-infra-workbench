"""Claude-powered fact extractor.

Mirrors the compute advisor's design:

- Haiku 4.5 default for budget.
- Prompt caching (ephemeral) on the system prompt and tool schema.
- Tool-use format for structured output — the model MUST call
  ``emit_facts`` once per doc rather than free-text JSON.
- A ``FakeExtractorClient`` makes extraction testable offline.

Each returned fact carries a ``quote`` that the caller verifies against
the source body — ``MarketIntelFact.verify_quote_in``. Hallucinated
quotes are dropped by ``extract_facts`` before they reach the store.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from core.market_intel.schema import FactKind, MarketIntelDoc, MarketIntelFact

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def detect_backend(api_key: str) -> str:
    """Return ``'anthropic'`` or ``'openai'`` from an API-key prefix.

    Anthropic keys start with ``sk-ant-``, OpenAI with ``sk-``. Raises
    on anything else so an unknown key fails loudly rather than silently
    picking a wrong backend.
    """
    if not api_key:
        raise ValueError("Empty API key")
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if api_key.startswith("sk-"):
        return "openai"
    raise ValueError(
        f"Unrecognised API key prefix {api_key[:7]!r}. "
        "Expected 'sk-ant-*' (Anthropic) or 'sk-*' (OpenAI)."
    )


# ---------------------------------------------------------------------------
# Prompt and tool schema
# ---------------------------------------------------------------------------


EXTRACT_SYSTEM_PROMPT = """You extract structured facts from AI-infrastructure documents (earnings calls, press releases, news, filings).

Your job: read the document and call the emit_facts tool ONCE with every eligible fact you find.

Eligible fact kinds — extract ONLY these:
- gpu_price       : a $/hr rate for a specific GPU or instance type (list, reserved, contract)
- gpu_shipment    : a count of GPUs shipped, delivered, allocated, or ordered
- capex           : a dollar amount committed to build datacenters, buy GPUs, or expand capacity
- capacity        : megawatts, square footage, or rack counts for a specific site or fleet
- contract        : a supply or compute-purchase deal with named counterparties and (ideally) a dollar amount or term
- roadmap        : a stated ship date, GA window, or product milestone for a specific GPU / instance

Rules:
1. Every fact must have a ``quote`` — a verbatim snippet from the source text that supports it. If you cannot quote the source, do not emit the fact.
2. If a number is approximate or hedged ("about", "roughly", "up to"), set confidence to "low" and keep the hedge in ``notes``.
3. Normalise units: USD for money, usd_per_hour for rates, units for counts, mw for power, sq_ft for area.
4. ``subject`` is the thing the fact is about ("H100 SXM5", "Nscale UK datacenter", "Blackwell"). Keep it short.
5. ``period`` is the time window the fact applies to, when stated ("Q3-2025", "FY2026", "2027").
6. If a doc contains no eligible facts, call emit_facts with an empty facts list. Do not invent facts.
7. Do not extract sentiment, analyst opinions, or market commentary — only concrete numbers and commitments.
"""


FACT_TOOL_SCHEMA: dict = {
    "name": "emit_facts",
    "description": "Emit the complete list of structured facts found in the document. Call this exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_kind": {
                            "type": "string",
                            "enum": [k.value for k in FactKind],
                        },
                        "subject": {"type": "string"},
                        "value_num": {"type": ["number", "null"]},
                        "value_unit": {
                            "type": ["string", "null"],
                            "enum": [
                                "usd", "usd_per_hour", "units", "mw", "sq_ft",
                                "percent", "months", "years", None,
                            ],
                        },
                        "value_text": {"type": ["string", "null"]},
                        "period": {"type": ["string", "null"]},
                        "company": {"type": ["string", "null"]},
                        "quote": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "notes": {"type": ["string", "null"]},
                    },
                    "required": ["fact_kind", "subject", "quote", "confidence"],
                },
            }
        },
        "required": ["facts"],
    },
}


# ---------------------------------------------------------------------------
# Client protocol — lets tests inject a fake
# ---------------------------------------------------------------------------


class ExtractorClient(Protocol):
    def emit(self, doc: MarketIntelDoc) -> list[dict[str, Any]]:
        """Return the raw list of fact dicts the LLM produced.

        Implementations are expected to parse the ``emit_facts`` tool-use
        block from the model response and return the ``facts`` array
        unchanged. ``extract_facts`` handles verification + schema
        validation downstream.
        """
        ...


class FakeExtractorClient:
    """Test double. Either static-replay or callback-driven."""

    def __init__(
        self,
        facts: list[dict[str, Any]] | None = None,
        callback: Callable[[MarketIntelDoc], list[dict[str, Any]]] | None = None,
    ):
        if facts is None and callback is None:
            raise ValueError("FakeExtractorClient needs either `facts` or `callback`")
        self._facts = facts
        self._callback = callback

    def emit(self, doc: MarketIntelDoc) -> list[dict[str, Any]]:
        if self._callback is not None:
            return self._callback(doc)
        return list(self._facts or [])


class AnthropicExtractorClient:
    """Real Anthropic SDK client. Lazy-imports the SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
    ):
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise RuntimeError("anthropic SDK not installed") from e
            self._client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
        return self._client

    def emit(self, doc: MarketIntelDoc) -> list[dict[str, Any]]:
        client = self._get_client()
        prompt = _render_user_prompt(doc)
        resp = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": EXTRACT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[FACT_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "emit_facts"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "emit_facts":
                return list(block.input.get("facts", []))
        return []


def _render_user_prompt(doc: MarketIntelDoc) -> str:
    header = [
        f"Document type: {doc.doc_type}",
        f"Title: {doc.title}",
    ]
    if doc.company:
        header.append(f"Company: {doc.company}")
    if doc.ticker:
        header.append(f"Ticker: {doc.ticker}")
    if doc.published_date:
        header.append(f"Published: {doc.published_date}")
    if doc.source_url:
        header.append(f"Source: {doc.source_url}")
    return "\n".join(header) + "\n\n---\n\n" + doc.body


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_facts(
    doc: MarketIntelDoc,
    client: ExtractorClient,
    *,
    require_quote_in_body: bool = True,
) -> list[MarketIntelFact]:
    """Extract facts from one doc. Drops unverifiable quotes.

    Raised fact dicts are validated by Pydantic; invalid rows are dropped
    rather than raising, so a single malformed fact does not lose the
    rest of a doc. Callers that want strict mode can set
    ``require_quote_in_body=False`` only to skip the citation check —
    schema validation is always on.
    """
    raw_facts = client.emit(doc)
    out: list[MarketIntelFact] = []
    for raw in raw_facts:
        try:
            fact = MarketIntelFact(doc_id=doc.doc_id, **raw)
        except Exception:
            continue
        if require_quote_in_body and not fact.verify_quote_in(doc.body):
            continue
        # Fallback: inherit company from doc if the LLM didn't supply one.
        if not fact.company and doc.company:
            fact.company = doc.company
        out.append(fact)
    return out
