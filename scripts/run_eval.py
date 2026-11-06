"""Live eval runner for the compute advisor.

Runs the 20-question set from ``core/eval/questions.py`` against a live
LLM backend (Anthropic or OpenAI — picked automatically from the key
prefix). ``ANTHROPIC_API_KEY`` is the canonical env var and accepts
either an Anthropic (``sk-ant-*``) or OpenAI (``sk-*``) key.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --model claude-sonnet-4-6 --out eval_report.md
    python scripts/run_eval.py --limit 5         # smoke-test on first 5 questions

Output:
    - A markdown file with per-question result + reasons for failures.
    - Stdout: `Pass: 17/20 (85%)` so a CI summary job can `grep` it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from core.agents.compute_advisor import ComputeAdvisor
from core.eval.harness import grade, summarise
from core.eval.questions import EVAL_QUESTIONS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--api-key",
        default=os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="Anthropic ('sk-ant-*') or OpenAI ('sk-*') key — backend auto-picked from prefix",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="override model id; default is backend-appropriate (Haiku 4.5 / gpt-4o-mini)",
    )
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--out", default="eval_report.md")
    ap.add_argument("--max-tool-calls", type=int, default=8)
    args = ap.parse_args()

    if not args.api_key:
        print(
            "No API key set. Set ANTHROPIC_API_KEY (accepts an Anthropic or "
            "OpenAI key) or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    questions = EVAL_QUESTIONS[: args.limit] if args.limit else EVAL_QUESTIONS

    advisor = ComputeAdvisor(
        api_key=args.api_key, model=args.model, max_tool_calls=args.max_tool_calls
    )
    print(f"backend: {advisor.backend} :: {advisor.model}", file=sys.stderr)

    results = []
    total_in = total_out = total_cache_read = 0
    t0 = time.perf_counter()
    for q in questions:
        print(f"[{q.id}] {q.question[:80]}...", file=sys.stderr)
        try:
            resp = advisor.ask(q.question)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        r = grade(q, resp)
        results.append((q, resp, r))
        if resp.usage:
            total_in += resp.usage["input_tokens"]
            total_out += resp.usage["output_tokens"]
            total_cache_read += resp.usage.get("cache_read_input_tokens", 0)
        print(
            f"  {'PASS' if r.passed else 'FAIL'}"
            f" tool={r.actual_tool} params_ok={r.params_matched}"
            f" numeric_ok={r.numeric_in_range}",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - t0
    summary = summarise([r for _, _, r in results])

    _write_report(
        Path(args.out), f"{advisor.backend} :: {advisor.model}", results, summary, elapsed,
        total_in, total_out, total_cache_read,
    )

    print(
        f"Pass: {summary['passed']}/{summary['n']} ({summary['pass_rate_pct']}%) — "
        f"wrote {args.out}"
    )
    return 0 if summary["pass_rate_pct"] >= 70 else 2


def _write_report(path, model, results, summary, elapsed, t_in, t_out, t_cache):
    lines = [
        "# Compute advisor eval report",
        "",
        f"- Model: `{model}`",
        f"- Questions: {summary['n']}",
        f"- **Pass rate: {summary['pass_rate_pct']}% ({summary['passed']}/{summary['n']})**",
        f"- Tool-called rate: {summary['tool_called_rate_pct']}%",
        f"- Params-matched rate: {summary['params_matched_rate_pct']}%",
        f"- Numeric-in-range rate: {summary.get('numeric_in_range_rate_pct')}%",
        f"- Elapsed: {elapsed:.1f}s",
        f"- Token usage: in={t_in:,}, out={t_out:,}, cache_read={t_cache:,}",
        "",
        "## Per-question results",
        "",
    ]
    for q, resp, r in results:
        marker = "✅" if r.passed else "❌"
        lines.append(f"### {marker} {q.id}")
        lines.append(f"**Q:** {q.question}")
        lines.append("")
        lines.append(f"- Expected tool: `{q.expected_tool}`")
        lines.append(f"- Actual tool: `{r.actual_tool}`")
        lines.append(f"- Params matched: {r.params_matched}")
        if r.numeric_in_range is not None:
            lines.append(f"- Numeric in range: {r.numeric_in_range} ({r.actual_numeric})")
        if r.reasons:
            lines.append("- Reasons:")
            for reason in r.reasons:
                lines.append(f"  - {reason}")
        lines.append("")
        lines.append("<details><summary>Agent response</summary>")
        lines.append("")
        lines.append(resp.text)
        lines.append("")
        if resp.tool_calls:
            lines.append("**Tool calls:**")
            for tc in resp.tool_calls:
                lines.append(f"- `{tc.name}` with `{json.dumps(tc.params)}`")
        lines.append("</details>")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
