"""Agent evaluation harness.

Twenty sizing questions with known tool-level answers. Running the harness
against the compute advisor reports:

- **Tool-call correctness**: did the agent call the right tool with inputs
  that produce an answer in the expected range?
- **Answer-text correctness**: does the final text contain the expected
  numeric answer within tolerance?

Both scores go in the README so the build's evaluation pass rate is stated
alongside the headline claim.
"""

from core.eval.harness import EvalResult, grade, run_eval_set
from core.eval.questions import EVAL_QUESTIONS, EvalQuestion

__all__ = ["EVAL_QUESTIONS", "EvalQuestion", "EvalResult", "grade", "run_eval_set"]
