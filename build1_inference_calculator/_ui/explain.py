"""Plain-English glossary — one source of truth for every tooltip and caption.

Audience: a non-technical reader (recruiter, GTM lead, strategy hire). Rule of
thumb — a business student should understand each entry without googling.

Usage:
    from _ui.explain import tooltip, caption
    st.slider("Busy fraction", 0.1, 1.0, 0.6, help=tooltip("busy_fraction"))
    st.caption(caption("memory_fit"))
"""

from __future__ import annotations

from dataclasses import dataclass

# Short tooltips rendered inside ``help=`` on widgets. One sentence, ≤140 chars.
TOOLTIPS: dict[str, str] = {
    "roofline": (
        "A chart showing the ceiling on how fast a GPU can run a model. "
        "Points above the line are physically impossible."
    ),
    "kv_cache": (
        "Memory the GPU uses to remember earlier tokens in a conversation "
        "so it doesn't redo work. Longer prompts = larger cache."
    ),
    "kernel_utilisation": (
        "How much of the GPU's theoretical max speed you actually get in "
        "practice. 65% is realistic for production inference."
    ),
    "busy_fraction": (
        "Of the hours you pay for a GPU, how many are actually serving "
        "tokens. 60% is a reasonable production average."
    ),
    "pue": (
        "Power usage effectiveness — a datacenter's electrical overhead. "
        "1.3 means 30% extra on top of the GPUs themselves."
    ),
    "fp16": "16-bit floating point. The default precision for most model weights.",
    "fp8": "8-bit floating point. Halves memory use with a small accuracy trade-off.",
    "int8": "8-bit integer. Even smaller than FP8; common for KV cache compression.",
    "int4": "4-bit integer. Smallest commonly used weight precision; saves the most memory.",
    "break_even": (
        "The level at which two options cost the same. Above it one wins; "
        "below it the other does."
    ),
    "stranded_capex": (
        "Money spent on GPU capacity you can't fill. Hurts owned and "
        "reserved options; doesn't touch on-demand or API."
    ),
    "demand_stress": (
        "A stress test: what happens to your cost per token if real demand "
        "lands 40% below your plan?"
    ),
    "arithmetic_intensity": (
        "FLOPs per byte of memory moved. Low intensity = bottlenecked by "
        "memory bandwidth, not compute."
    ),
    "mlperf": (
        "An industry benchmark round where vendors publish measured "
        "inference numbers on standard hardware."
    ),
    "tco": (
        "Total cost of ownership over 3 years — capex + power + hosting + "
        "financing rolled into one number per million tokens."
    ),
    "fact_kind": (
        "The category of a fact pulled from a document: price, capex, "
        "shipment, capacity, contract, or roadmap."
    ),
    "cached_tokens": (
        "Repeated prompt text charged at ~10% of the normal rate. Big "
        "savings on long system prompts across many turns."
    ),
    "input_tokens": (
        "Everything the model reads — your question, the system prompt, "
        "prior turns, and any tool results."
    ),
    "output_tokens": (
        "The model's reply. Usually priced about 4× higher than input."
    ),
    "tool_call": (
        "The agent pausing to run a real calculation (e.g. compare TCO) "
        "instead of guessing the number itself."
    ),
    "verbatim_quote": (
        "The exact sentence from the source document the fact was pulled "
        "from. Guards against invented numbers."
    ),
    "measured_tag": "Numbers from an actual hardware run.",
    "modelled_tag": "Numbers computed from physics or formulas.",
    "simulated_tag": "Numbers from a traced workload replay.",
    "assumption_tag": "Numbers chosen by the author as a reasonable default.",
    "confidence_badge": (
        "🟢 vendor datasheet cited. 🟡 web/press material; revisit as "
        "benchmarks mature. 🔴 announcement-only; may be modelled."
    ),
    "hourly_usd": (
        "What you'd pay per GPU per hour. Market rate for an H100 SXM "
        "sits around $2–3/hr on neocloud."
    ),
    "throughput_tps": (
        "Output tokens per second — the bandwidth of how fast the system "
        "generates text."
    ),
    "batch_size": (
        "Number of requests served in parallel per GPU step. Higher batch "
        "= more throughput, more memory, more latency per request."
    ),
    "input_length": "Average prompt length in tokens.",
    "output_length": "Average reply length in tokens.",
    "horizon_years": "How many years of ownership you're amortising capex over. 3 is standard.",
    "financing_pct": "Annual interest rate if you borrowed to buy the GPUs.",
    "hosting_usd_per_month": "Rack space, cooling, cross-connects you pay the DC per GPU per month.",
    "electricity_usd_per_kwh": "Industrial power rate. UK ~$0.09/kWh; US ~$0.07; EU ~$0.11.",
    "depreciation_pct": "Fraction of capex written off by end of horizon. 70% = 30% residual value.",
}

# Longer captions rendered as ``st.caption(...)`` or inline markdown. 1-2 sentences.
CAPTIONS: dict[str, str] = {
    "memory_fit": (
        "Weights, activations and the KV cache share the GPU's memory at the "
        "same time. If they sum above HBM, the model won't fit at this batch size."
    ),
    "roofline_chart": (
        "Points above the line are impossible. Points on the line are the physical "
        "ceiling — compute-bound on the right, memory-bound on the left."
    ),
    "tco_comparison": (
        "Same workload costed five ways on one ruler: $ per million output tokens, "
        "amortised over 3 years."
    ),
    "demand_stress_chart": (
        "What if real demand lands 40% below your plan? Owned and reserved capacity "
        "still gets paid for. API and on-demand scale down with you."
    ),
    "break_even_header": (
        "The tipping point where one option stops being cheaper than another. "
        "Use it to pressure-test which choice survives a demand shift."
    ),
    "verbatim_quote": (
        "Every fact carries the exact sentence from the source document it was "
        "extracted from. If the quote isn't there, the fact isn't recorded."
    ),
    "cost_tracker_hero": (
        "Every Advisor chat turn is logged here with its token breakdown and the "
        "£ it cost. Switch models on the Advisor page to compare side by side."
    ),
    "pricing_levers": (
        "LLM pricing has three levers — input tokens, output tokens, and cached "
        "tokens. Output is the most expensive; cache is the cheapest."
    ),
    "advisor_intro": (
        "Ask a sizing or build-vs-rent question. Every number in the answer comes "
        "from a tool call against the engine — the model never calculates a figure itself."
    ),
    "market_intel_intro": (
        "Live prices and capex that back the TCO assumptions. Every row carries "
        "the exact quote it was pulled from."
    ),
}


def tooltip(key: str) -> str:
    """Short ``help=`` string. Empty string if the key is unknown — safe
    to pass straight into Streamlit without conditional logic."""
    return TOOLTIPS.get(key, "")


def caption(key: str) -> str:
    """Longer caption text. Empty string if unknown."""
    return CAPTIONS.get(key, "")


@dataclass(frozen=True)
class _BreakEvenShape:
    """Duck-typed shape of ``core.cost_engine.tco.BreakEven``.

    We don't import the real one here — keeps this module streamlit-free
    and avoids a cost_engine import chain in the UI helpers. The real
    BreakEven dataclass has the same attribute names.
    """

    option_a: str
    option_b: str
    pivot_variable: str
    pivot_value: float
    note: str


_OPTION_LABELS: dict[str, str] = {
    "owned": "Owning",
    "reserved": "Renting reserved",
    "on_demand": "Renting on-demand",
    "neocloud": "Using neocloud",
    "api": "Paying per-token API",
}


def _label(option: str) -> str:
    return _OPTION_LABELS.get(option, option.replace("_", " ").title())


def break_even_sentence(be) -> str:  # noqa: ANN001 — duck-typed BreakEven
    """Render a BreakEven as plain English.

    Examples
    --------
    pivot_variable='busy_fraction', pivot_value=0.47 →
      "Owning beats renting reserved once you're decoding more than 47% of
       the hours you pay for."
    """
    pct = f"{be.pivot_value * 100:.0f}%"
    a = _label(be.option_a)
    b = _label(be.option_b)
    templates: dict[str, str] = {
        "busy_fraction": (
            f"{a} beats {b} once you're decoding more than {pct} of the hours you pay for."
        ),
        "utilisation_frac": f"{a} beats {b} once sustained kernel utilisation stays above {pct}.",
        "hourly_usd": (
            f"{a} ties {b} at a rental rate of ${be.pivot_value:.2f}/GPU-hour."
        ),
    }
    return templates.get(be.pivot_variable, f"{a} beats {b} once {be.pivot_variable} crosses {pct}.")
