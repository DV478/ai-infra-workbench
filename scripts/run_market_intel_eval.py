"""Live eval runner for the market intelligence extractor.

Runs the 8-case gold set from ``core.market_intel.eval.GOLD_CASES``
against a real LLM backend, writes a markdown report, and prints a
one-line summary suitable for a README badge.

Needs ``ANTHROPIC_API_KEY`` in the environment (or ``--api-key``) for
the default Anthropic backend.

Usage:
    python scripts/run_market_intel_eval.py
    python scripts/run_market_intel_eval.py --limit 3           # smoke
    python scripts/run_market_intel_eval.py --fake              # deterministic
    python scripts/run_market_intel_eval.py --out report.md

Exit codes:
    0 — pass rate >= 70%
    1 — missing API key
    2 — pass rate below threshold
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from core.market_intel import (
    GOLD_CASES,
    FakeExtractorClient,
    MarketIntelDoc,
    extract_facts,
    summarise,
)
from core.market_intel.extractor import AnthropicExtractorClient


def _fake_callback(doc: MarketIntelDoc) -> list[dict]:
    """Minimal rule-based extractor so --fake gives a non-trivial baseline."""
    import re

    out: list[dict] = []
    body = doc.body

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
        })
    for m in re.finditer(r"\$(\d+\.\d{2})\s*(?:per|/)\s*GPU-hour", body, flags=re.IGNORECASE):
        snippet = body[max(0, m.start() - 60): m.end() + 40]
        out.append({
            "fact_kind": "gpu_price",
            "subject": "H200",
            "value_num": float(m.group(1)),
            "value_unit": "usd_per_hour",
            "quote": snippet.strip(),
            "confidence": "medium",
        })
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*MW", body):
        snippet = body[max(0, m.start() - 40): m.end() + 40]
        out.append({
            "fact_kind": "capacity",
            "subject": doc.company or "capacity",
            "value_num": float(m.group(1)),
            "value_unit": "mw",
            "quote": snippet.strip(),
            "confidence": "medium",
        })
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
    ap.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"))
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake", action="store_true",
                    help="use deterministic rule-based extractor (no API spend)")
    ap.add_argument("--out", default="market_intel_eval_report.md")
    args = ap.parse_args()

    cases = GOLD_CASES[: args.limit] if args.limit else GOLD_CASES

    if args.fake:
        client = FakeExtractorClient(callback=_fake_callback)
        model_label = "fake (rule-based)"
    else:
        if not args.api_key:
            print("No ANTHROPIC_API_KEY set. Use --api-key, --fake, or set env var.",
                  file=sys.stderr)
            return 1
        client = AnthropicExtractorClient(api_key=args.api_key, model=args.model)
        model_label = args.model

    def extract_fn(doc: MarketIntelDoc):
        return extract_facts(doc, client)

    t0 = time.perf_counter()
    results = []
    for c in cases:
        print(f"[{c.id}] {c.doc.title[:60]}", file=sys.stderr)
        try:
            actual = extract_fn(c.doc)
        except Exception as e:
            print(f"  extract FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        from core.market_intel.eval import grade_case
        r = grade_case(c, actual)
        results.append((c, actual, r))
        marker = "PASS" if r.passed else "FAIL"
        print(
            f"  {marker} recall={r.recall_pct}% precision={r.precision_pct}% "
            f"quote={r.quote_valid_rate_pct}%",
            file=sys.stderr,
        )
    elapsed = time.perf_counter() - t0

    summary = summarise([r for _, _, r in results])
    _write_report(Path(args.out), model_label, results, summary, elapsed)

    print(
        f"Pass: {summary['passed']}/{summary['n']} ({summary['pass_rate_pct']}%) — "
        f"avg recall {summary['avg_recall_pct']}%, precision {summary['avg_precision_pct']}%, "
        f"quote-valid {summary['avg_quote_valid_rate_pct']}% — wrote {args.out}"
    )
    return 0 if summary["pass_rate_pct"] >= 70 else 2


def _write_report(path: Path, model: str, results, summary, elapsed: float) -> None:
    lines = [
        "# Market intelligence eval report",
        "",
        f"- Backend: `{model}`",
        f"- Cases: {summary['n']}",
        f"- **Pass rate: {summary['pass_rate_pct']}% ({summary['passed']}/{summary['n']})**",
        f"- Avg recall: {summary['avg_recall_pct']}%",
        f"- Avg precision: {summary['avg_precision_pct']}%",
        f"- Avg quote-valid rate: {summary['avg_quote_valid_rate_pct']}%",
        f"- Elapsed: {elapsed:.1f}s",
        "",
        "Pass criterion per case: recall >= 70% AND quote-valid = 100%.",
        "",
        "## Per-case results",
        "",
    ]
    for case, actual, r in results:
        marker = "PASS" if r.passed else "FAIL"
        lines.append(f"### [{marker}] {case.id}")
        lines.append(f"**Doc:** {case.doc.title}")
        lines.append("")
        lines.append(
            f"- recall: {r.recall_pct}% ({len(r.matched)}/{len(case.expected)} expected facts matched)"
        )
        lines.append(
            f"- precision: {r.precision_pct}% ({len(r.matched)}/{len(actual)} actual facts were expected)"
        )
        lines.append(f"- quote-valid: {r.quote_valid_rate_pct}%")
        if r.missed:
            lines.append("- missed expected:")
            for e in r.missed:
                val = f" value~={e.value_num}" if e.value_num is not None else ""
                lines.append(f"  - {e.fact_kind.value} :: {e.subject_keyword}{val}")
        if r.spurious:
            lines.append("- spurious (unexpected) facts:")
            for a in r.spurious:
                lines.append(f"  - {a.fact_kind.value} :: {a.subject} = {a.value_num}")
        if r.reasons:
            lines.append("- reasons:")
            for reason in r.reasons:
                lines.append(f"  - {reason}")
        lines.append("")
        lines.append("<details><summary>Actual facts</summary>")
        lines.append("")
        for a in actual:
            lines.append(
                f"- `{a.fact_kind.value}` {a.subject} = {a.value_num} {a.value_unit or ''} "
                f"[{a.confidence}] — _{a.quote[:120]}_"
            )
        lines.append("</details>")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
