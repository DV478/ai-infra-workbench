"""Run the market intelligence extractor over text files on disk.

Reads documents from ``data/raw/market/`` and writes facts to
``data/market_intel.duckdb``. File naming convention:

    <doc_type>__<company>__<YYYY-MM-DD>__<slug>.{txt,md}

Example: ``earnings_transcript__nscale__2026-10-14__q3-2026.txt``

If parts are missing, the parser falls back to sensible defaults:
``doc_type=news``, ``company=unknown``, ``published_date=None``.

Usage:
    # real extraction (requires ANTHROPIC_API_KEY)
    python scripts/run_market_intel.py

    # fake-client dry run (no API key, uses an inline sample extractor)
    python scripts/run_market_intel.py --fake

    # one-shot single file
    python scripts/run_market_intel.py --file data/raw/market/x.txt

Exit codes:
    0 — ran successfully (may have written zero facts if the docs had none)
    2 — no docs found to process
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from core.market_intel import (
    FakeExtractorClient,
    MarketIntelDoc,
    MarketIntelStore,
    extract_facts,
)
from core.market_intel.extractor import AnthropicExtractorClient

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "market"
STORE_PATH = ROOT / "data" / "market_intel.duckdb"


_FILENAME_RE = re.compile(
    r"^(?P<doc_type>[a-z_]+)__(?P<company>[a-z0-9\-]+)__(?P<date>\d{4}-\d{2}-\d{2})__(?P<slug>[a-z0-9\-]+)"
)


def parse_filename(path: Path) -> dict[str, str | None]:
    m = _FILENAME_RE.match(path.stem)
    if m:
        return m.groupdict()
    return {"doc_type": "news", "company": None, "date": None, "slug": path.stem}


def load_docs(folder: Path) -> list[MarketIntelDoc]:
    docs: list[MarketIntelDoc] = []
    for p in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.md")):
        parts = parse_filename(p)
        body = p.read_text(encoding="utf-8")
        title_line = body.splitlines()[0].strip() if body else parts["slug"] or p.stem
        docs.append(
            MarketIntelDoc(
                doc_type=parts["doc_type"] or "news",
                title=title_line[:200],
                body=body,
                company=parts["company"],
                published_date=parts["date"],
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Fake-client fallback — lets the pipeline run with zero API spend.
# Looks for substrings commonly found in the included sample docs and emits
# plausible facts. Real extraction uses AnthropicExtractorClient.
# ---------------------------------------------------------------------------


def _fake_callback(doc: MarketIntelDoc) -> list[dict]:
    """Deterministic rule-based extractor for demo / CI. Never hits Claude."""
    out: list[dict] = []
    body = doc.body

    # Capex dollar figures
    for m in re.finditer(r"\$(\d+(?:\.\d+)?)\s*(billion|million)", body, flags=re.IGNORECASE):
        amt = float(m.group(1)) * (1_000_000_000 if m.group(2).lower() == "billion" else 1_000_000)
        snippet = body[max(0, m.start() - 40): m.end() + 60]
        out.append({
            "fact_kind": "capex",
            "subject": doc.company or "unspecified",
            "value_num": amt,
            "value_unit": "usd",
            "quote": snippet.strip(),
            "confidence": "medium",
            "notes": "rule-based fake extractor; verify manually before citing",
        })
    # GPU prices
    for m in re.finditer(r"\$(\d+\.\d{2})\s*(?:per|/)\s*GPU-hour", body, flags=re.IGNORECASE):
        snippet = body[max(0, m.start() - 60): m.end() + 40]
        out.append({
            "fact_kind": "gpu_price",
            "subject": "unspecified",
            "value_num": float(m.group(1)),
            "value_unit": "usd_per_hour",
            "quote": snippet.strip(),
            "confidence": "medium",
        })
    # Capacity (MW)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*MW", body):
        snippet = body[max(0, m.start() - 40): m.end() + 40]
        out.append({
            "fact_kind": "capacity",
            "subject": doc.company or "unspecified",
            "value_num": float(m.group(1)),
            "value_unit": "mw",
            "quote": snippet.strip(),
            "confidence": "medium",
        })
    # GPU shipments
    for m in re.finditer(r"(\d{1,3}(?:,\d{3})+)\s+(?:Nvidia\s+)?(H100|H200|B100|B200|GB200)", body):
        units = int(m.group(1).replace(",", ""))
        snippet = body[max(0, m.start() - 30): m.end() + 40]
        out.append({
            "fact_kind": "gpu_shipment",
            "subject": m.group(2),
            "value_num": units,
            "value_unit": "units",
            "quote": snippet.strip(),
            "confidence": "medium",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fake", action="store_true",
                    help="use rule-based fake extractor instead of Anthropic SDK")
    ap.add_argument("--file", type=Path, default=None,
                    help="process only this file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print facts without writing to DuckDB")
    args = ap.parse_args()

    if args.file is not None:
        if not args.file.exists():
            print(f"file not found: {args.file}", file=sys.stderr)
            return 2
        docs = load_docs(args.file.parent)
        docs = [d for d in docs if d.title.startswith(args.file.read_text(encoding="utf-8").splitlines()[0][:50])]
        # Fallback — if filename-regex matching trimmed everything, parse directly
        if not docs:
            docs = load_docs(args.file.parent)
    else:
        if not RAW_DIR.exists():
            print(f"no raw dir at {RAW_DIR} — create it and drop .txt/.md files", file=sys.stderr)
            return 2
        docs = load_docs(RAW_DIR)

    if not docs:
        print("no docs found", file=sys.stderr)
        return 2

    if args.fake:
        client = FakeExtractorClient(callback=_fake_callback)
    else:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set — use --fake for a no-spend run", file=sys.stderr)
            return 2
        client = AnthropicExtractorClient()

    store = None if args.dry_run else MarketIntelStore(STORE_PATH)
    total_facts = 0
    for doc in docs:
        facts = extract_facts(doc, client)
        total_facts += len(facts)
        print(f"[{doc.doc_type}] {doc.company or '?'} :: {doc.title[:60]} -> {len(facts)} facts")
        if store is not None:
            store.add_doc(doc)
            store.add_facts(facts)

    if store is not None:
        print(f"wrote {total_facts} facts for {len(docs)} docs -> {STORE_PATH}")
        store.close()
    else:
        print(f"[dry-run] would have written {total_facts} facts for {len(docs)} docs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
