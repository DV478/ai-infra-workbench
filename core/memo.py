"""Two-page sizing-memo generator.

Takes a ``TcoScenario`` + inputs and produces a markdown document that
prints cleanly to one or two pages. The Streamlit TCO page exposes this as
a download button; the owner saves the markdown and converts to PDF via
``pandoc memo.md -o memo.pdf`` or browser ``Print → Save as PDF``.

We write markdown rather than PDF directly so the project does not need a
PDF dependency (``reportlab`` / ``weasyprint`` pull in a lot). The memo's
structure is tight enough that any reasonable renderer produces a good
result.
"""

from __future__ import annotations

from datetime import date
from textwrap import dedent

from core.cost_engine.tco import TcoScenario


def _fmt_usd(x: float) -> str:
    if x >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"${x/1_000:.1f}k"
    return f"${x:.2f}"


def _fmt_num(x: float) -> str:
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"


def render_memo(
    scenario: TcoScenario,
    *,
    gpu_name: str,
    gpu_id: str,
    dtype: str,
    batch_size: int,
    input_length: int,
    output_length: int,
    utilisation_frac: float,
    author: str = "AI Compute Economics Workbench",
    for_audience: str = "",
    today: date | None = None,
) -> str:
    today = today or date.today()
    results = sorted(scenario.results, key=lambda r: r.usd_per_m_tokens)
    cheapest = results[0]
    most_exp = results[-1]
    spread = most_exp.usd_per_m_tokens / cheapest.usd_per_m_tokens if cheapest.usd_per_m_tokens > 0 else 0

    stress_by_option = {s.option: s for s in scenario.stress}

    lines = [
        f"# Sizing memo — {scenario.label}",
        "",
        f"*Prepared {today.isoformat()} by {author}*  ",
        f"{'For: ' + for_audience if for_audience else ''}",
        "",
        "## Recommendation",
        "",
        f"**{cheapest.label}** at **{_fmt_usd(cheapest.usd_per_m_tokens)} per million output tokens** over 3 years, assuming "
        f"{int(scenario.busy_fraction * 100)}% busy fraction on the configured workload.",
        "",
        f"The spread between the cheapest and most expensive option is **{spread:.1f}x** — this is a decision that matters.",
        "",
        "## Workload",
        "",
        f"- **Model:** {scenario.label.split(' on ')[0]} ({dtype})",
        f"- **GPU:** {gpu_name} (`{gpu_id}`)",
        f"- **Batch size:** {batch_size}",
        f"- **Avg input / output length:** {input_length} / {output_length} tokens",
        f"- **Kernel utilisation:** {int(utilisation_frac * 100)}% of peak (modelled)",
        f"- **Busy fraction:** {int(scenario.busy_fraction * 100)}% (serving-time)",
        f"- **Workload volume:** {_fmt_num(scenario.workload_tokens_per_year)} output tokens/year",
        "",
        "## Cost comparison (3-year total)",
        "",
        "| Option | 3-year $ | $/M output tokens | Monthly $ |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {_fmt_usd(r.total_usd_3y)} | {_fmt_usd(r.usd_per_m_tokens)} | {_fmt_usd(r.monthly_usd)} |"
        )
    lines.append("")

    lines.extend([
        "## Break-even",
        "",
    ])
    for be in scenario.break_evens:
        lines.append(f"- **{be.option_a} vs {be.option_b}** — {be.pivot_variable} = `{be.pivot_value:.3f}`. {be.note}")
    lines.append("")

    lines.extend([
        "## Demand stress — 40% miss",
        "",
        "If demand lands at 60% of forecast:",
        "",
        "| Option | Base $/M | Stressed $/M | Stranded capex |",
        "|---|---:|---:|---:|",
    ])
    for r in results:
        s = stress_by_option.get(r.option)
        if s is None:
            continue
        lines.append(
            f"| {s.label} | {_fmt_usd(s.base_usd_per_m)} | {_fmt_usd(s.stressed_usd_per_m)} | {_fmt_usd(s.stranded_capacity_usd)} |"
        )
    lines.append("")

    lines.extend(dedent(
        """
        ## Methodology & provenance

        - GPU spec figures: vendor datasheets, dated & cited. Confidence tagged `high`/`medium`/`low`.
        - Throughput predictions: roofline model (`core.cost_engine.roofline`), **modelled**.
        - Validation: Kaggle T4 + vLLM runs for 1-3B models, cross-checked against MLPerf Inference `Llama-2-70B` on H100 SXM.
        - Hourly rates: user-supplied; price history accumulating in `data/prices/prices.parquet` via weekly logger.
        - API rates: user-supplied list price. Does not model free-tier allowances, volume discounts, or BYOC contracts.

        ## Assumptions that would change the answer

        - **Busy fraction**. The whole stack is sensitive to this. Measure it on a real workload before committing.
        - **Kernel utilisation**. 65% is a defensible H100 FP16 GEMM number; T4 is closer to 45-55%.
        - **Capex on owned**. Secondary-market H100 prices have moved materially in 2026; refresh before signing.
        - **API list price**. Providers quote list but discount hard on volume. Get a quote.
        """
    ).strip().splitlines())
    lines.append("")

    return "\n".join(lines)
