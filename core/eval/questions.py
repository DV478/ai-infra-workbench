"""Twenty known-answer questions for the compute advisor.

Each ``EvalQuestion`` has:

- ``question`` — the user's natural-language ask (what the agent sees).
- ``expected_tool`` — the single core tool a correct answer must call.
  ``None`` means multi-tool is acceptable.
- ``expected_params`` — key-value pairs we expect in that tool's input.
  Checks that the agent picked the right GPU, model, dtype, etc.
- ``expected_numeric`` — ``(key, lower, upper)``. The tool result dict must
  contain ``key`` with a value in ``[lower, upper]``. Loose bounds —
  catches "right ballpark, right bound regime" failures.
- ``expected_text_contains`` — substrings that must appear in the agent's
  final text. Keeps answers honest without over-constraining prose.

The harness runs each question twice (same temperature) and reports the
pass rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_tool: str | None  # The one tool a correct answer must call
    expected_params: dict  # Subset of params that must match
    expected_numeric: tuple[str, float, float] | None  # (key, lo, hi)
    expected_text_contains: tuple[str, ...] = ()
    notes: str = ""


EVAL_QUESTIONS: list[EvalQuestion] = [
    # -----------------------------------------------------------------------
    # Memory fit questions
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="fit_70b_fp16_h100",
        question="Does Llama-3-70B at FP16 fit on a single H100 SXM5 80GB with batch 1 and 2048-token context?",
        expected_tool="fits_in_memory",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "dtype": "fp16"},
        expected_numeric=("headroom_gb", -200.0, -20.0),  # 140GB weights alone exceed 80GB — negative headroom
        expected_text_contains=("no", "fit"),
    ),
    EvalQuestion(
        id="fit_70b_int4_h100",
        question="Can Llama-3-70B at INT4 fit on a single H100 SXM5 80GB with batch 1, 2048 tokens?",
        expected_tool="fits_in_memory",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "dtype": "int4"},
        expected_numeric=("headroom_gb", 20.0, 80.0),  # INT4 ~35 GB; large headroom
        expected_text_contains=("fit",),
    ),
    EvalQuestion(
        id="fit_8b_fp16_h100",
        question="Does Llama-3-8B FP16 fit on H100 SXM5 80GB with batch 32 at 2048 context?",
        expected_tool="fits_in_memory",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "dtype": "fp16"},
        expected_numeric=("headroom_gb", 40.0, 70.0),
        expected_text_contains=("fit",),
    ),
    # -----------------------------------------------------------------------
    # Decode throughput
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="decode_8b_h100_b1",
        question="What decode throughput can a single H100 SXM5 produce for Llama-3-8B FP16 at batch 1, 1024-token KV cache?",
        expected_tool="decode_tps",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "batch_size": 1},
        expected_numeric=("tokens_per_second", 50.0, 250.0),
        expected_text_contains=("memory",),
        notes="Roofline-only prediction; memory-bound at batch 1.",
    ),
    EvalQuestion(
        id="decode_8b_h100_b32",
        question="What's the aggregate decode throughput for Llama-3-8B FP16 on H100 SXM5 at batch 32, 1024 KV length?",
        expected_tool="decode_tps",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "batch_size": 32},
        expected_numeric=("tokens_per_second", 1500.0, 7000.0),
    ),
    EvalQuestion(
        id="decode_tinyllama_t4",
        question="Predicted decode tok/s for TinyLlama-1.1B FP16 on a T4 at batch 1, 512 KV length?",
        expected_tool="decode_tps",
        expected_params={"gpu_id": "t4", "model": "tinyllama-1.1b", "batch_size": 1},
        expected_numeric=("tokens_per_second", 60.0, 150.0),
    ),
    EvalQuestion(
        id="decode_70b_h100_fp8",
        question="On an H100 SXM5, what decode throughput can I get for Llama-3-70B at FP8, batch 32, 1024 KV?",
        expected_tool="decode_tps",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "batch_size": 32, "dtype": "fp8"},
        expected_numeric=("tokens_per_second", 300.0, 2500.0),
    ),
    # -----------------------------------------------------------------------
    # Cost per million tokens
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="cost_8b_h100_3usd",
        question="If I rent an H100 SXM5 for $3/hr and serve Llama-3-8B FP16 at batch 32 with 60% busy fraction, what does a million output tokens cost?",
        expected_tool="cost_per_m_tokens",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "hourly_usd": 3.0},
        expected_numeric=("cost_per_m_tokens_realistic", 0.1, 3.0),
    ),
    EvalQuestion(
        id="cost_70b_h100_fp8",
        question="Cost per million output tokens for Llama-3-70B FP8 on H100 SXM5 at $3/hr, batch 32, 60% busy?",
        expected_tool="cost_per_m_tokens",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "hourly_usd": 3.0, "dtype": "fp8"},
        expected_numeric=("cost_per_m_tokens_realistic", 0.5, 10.0),
    ),
    EvalQuestion(
        id="cost_scales_with_hourly",
        question="If my hourly rate doubles from $2 to $4 on an H100 SXM5 running Llama-3-8B FP16 batch 32, what does cost per million tokens do?",
        expected_tool="cost_per_m_tokens",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b"},
        expected_numeric=None,
        expected_text_contains=("double", ),
    ),
    # -----------------------------------------------------------------------
    # Rent-vs-buy / compare_options
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="compare_70b_default",
        question="For Llama-3-70B FP8 on H100 SXM5, compare owning (capex $32k), 3y reserved at $2.50/hr, on-demand $4.50/hr, neocloud $2.00/hr and Together AI at $0.90/M output / $0.90/M input. Batch 32, 60% busy, 512 in / 256 out.",
        expected_tool="compare_options",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-70b", "dtype": "fp8"},
        expected_numeric=None,
        expected_text_contains=("owned", "api"),
    ),
    EvalQuestion(
        id="compare_8b_low_util",
        question="Llama-3-8B FP16 on H100 SXM5 — at 20% busy fraction, is owning ($32k capex) or a $2/M-token API cheaper over 3 years? Reserved $2.50/hr, on-demand $4.50, neocloud $2.00.",
        expected_tool="compare_options",
        expected_params={"model": "llama-3-8b", "busy_fraction": 0.2},
        expected_numeric=None,
        expected_text_contains=("api",),
        notes="At 20% busy fraction, owning wastes capex — API should win.",
    ),
    # -----------------------------------------------------------------------
    # API margin
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="margin_together_8b",
        question="At $3/hr on H100 SXM5 serving Llama-3-8B FP16, what gross margin does Together AI's $0.20/M-tokens imply?",
        expected_tool="api_margin",
        expected_params={"api_price_per_m_tokens": 0.20, "model": "llama-3-8b"},
        expected_numeric=None,
    ),
    EvalQuestion(
        id="margin_expensive_api",
        question="If an API charges $10/M tokens for Llama-3-8B and we can serve it at $3/hr on H100 SXM5 FP16 batch 32, what's the implied gross margin?",
        expected_tool="api_margin",
        expected_params={"api_price_per_m_tokens": 10.0},
        expected_numeric=("gross_margin_pct", 80.0, 100.0),
    ),
    # -----------------------------------------------------------------------
    # Output tokens per hour
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="tph_8b_h100",
        question="How many Llama-3-8B FP16 output tokens per day can one H100 SXM5 produce at batch 32, 60% busy, 1024 KV?",
        expected_tool="output_tokens_per_hour",
        expected_params={"gpu_id": "h100_sxm5_80gb", "model": "llama-3-8b", "batch_size": 32, "busy_fraction": 0.60},
        expected_numeric=("tokens_per_day", 50_000_000, 400_000_000),
    ),
    # -----------------------------------------------------------------------
    # Owned hourly
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="owned_hourly_32k",
        question="What's the all-in hourly cost of owning one H100 at $32,000 capex, 700W TDP, 1.4 PUE, $0.10/kWh, $1,200/month hosting, 8% financing, 3% maintenance, 3-year depreciation?",
        expected_tool="owned_hourly_cost",
        expected_params={"capex_usd": 32000.0},
        expected_numeric=("hourly_usd", 2.5, 4.0),
    ),
    EvalQuestion(
        id="owned_hourly_50k",
        question="Same inputs as default but capex is $50,000 — what's the hourly?",
        expected_tool="owned_hourly_cost",
        expected_params={"capex_usd": 50000.0},
        expected_numeric=("hourly_usd", 3.5, 5.5),
    ),
    # -----------------------------------------------------------------------
    # Break-even
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="break_even_hourly",
        question="If I can serve 1,000,000 output tokens per hour and want to match a $2/M-token API, what's the maximum hourly rate I can pay?",
        expected_tool="break_even_hourly",
        expected_params={"cost_per_m_tokens_target": 2.0, "output_tokens_per_hr": 1_000_000.0},
        expected_numeric=("hourly_usd", 1.9, 2.1),
    ),
    # -----------------------------------------------------------------------
    # API-only
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="api_only_500m",
        question="How much does serving 500M output tokens per year via a $2/M-output / $1/M-input API with IO ratio 2 cost over 3 years?",
        expected_tool="api_only_cost",
        expected_params={"api_output_price_per_m": 2.0, "tokens_demanded_3y": 1_500_000_000.0},
        expected_numeric=("total_usd_3y", 5_000_000, 7_000_000),
    ),
    # -----------------------------------------------------------------------
    # Out-of-scope guardrail
    # -----------------------------------------------------------------------
    EvalQuestion(
        id="out_of_scope_training",
        question="How many GPU-hours does it take to train Llama-3-70B from scratch?",
        expected_tool=None,
        expected_params={},
        expected_numeric=None,
        expected_text_contains=("scope", ),
        notes="Training is out of scope; agent should decline.",
    ),
]


assert len(EVAL_QUESTIONS) == 20, f"Eval set should have 20 questions, has {len(EVAL_QUESTIONS)}"
