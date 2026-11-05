"""Read MLPerf Inference result CSVs and cross-check against roofline predictions.

Workflow:
1. User downloads a round's CSV bundle from https://mlcommons.org/en/inference-datacenter-50/
   and drops the file in ``data/raw/mlperf/``. The schema is unstable round
   to round; ``read_results_csv`` is deliberately defensive.
2. ``normalise_system`` maps the submitter's system string (e.g.
   "Dell PowerEdge XE9680, 8x H100 SXM5-80GB") to our ``gpu_specs`` id.
3. ``cross_check`` joins MLPerf measured tokens/s against the roofline
   prediction for the same (gpu, model, scenario) and returns the
   percentage error. The Streamlit Validation page shows this table.

Scope: we focus on the ``Llama2-70B`` task because it is the only LLM-shaped
workload currently in both the MLPerf Inference and MLPerf Inference Mobile
rounds. ``read_results_csv`` can read any MLPerf CSV; the filters live in
the loader functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MLPERF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "mlperf"


@dataclass(frozen=True)
class MlperfRow:
    """One row of our normalised MLPerf view.

    Not every original MLPerf row has a value for every field — missing
    numerics are ``None`` so the Streamlit renderer can show '-' instead of
    a sentinel number.
    """

    round: str  # "v4.1", "v5.0", ...
    submitter: str
    system: str
    gpu_id: str | None  # mapped to our catalog id (None if unmappable)
    accelerator_count: int | None
    model_task: str  # 'Llama2-70B', 'Mixtral-8x7B', 'GPT-J 6B', ...
    scenario: str  # 'Server', 'Offline', 'SingleStream', 'MultiStream'
    tokens_per_second: float | None
    availability: str | None  # 'available', 'preview', 'rdi'
    notes: str | None = None


# ---------------------------------------------------------------------------
# System string -> gpu_id normalisation
# ---------------------------------------------------------------------------

# Common patterns MLPerf submitters use. Order matters — more specific first.
_SYSTEM_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bh100[-_\s]*sxm5", re.I), "h100_sxm5_80gb"),
    (re.compile(r"\bh100[-_\s]*sxm", re.I), "h100_sxm5_80gb"),
    (re.compile(r"\bh100[-_\s]*pcie", re.I), "h100_pcie_80gb"),
    (re.compile(r"\bh100[-_\s]*nvl", re.I), "h100_pcie_80gb"),
    (re.compile(r"\bh200[-_\s]*(sxm|hgx|141)", re.I), "h200_sxm_141gb"),
    (re.compile(r"\bh200\b", re.I), "h200_sxm_141gb"),
    (re.compile(r"\bb200\b", re.I), "b200"),
    (re.compile(r"\bgb200\b", re.I), "gb200"),
    (re.compile(r"\bb100\b", re.I), "b100"),
    (re.compile(r"\bb300\b", re.I), "b300"),
    (re.compile(r"\bl40s\b", re.I), "l40s"),
    (re.compile(r"\bl4\b", re.I), "l4"),
    (re.compile(r"\ba100[-_\s]*(sxm|80)", re.I), "a100_sxm_80gb"),
    (re.compile(r"\ba100\b", re.I), "a100_sxm_80gb"),
    (re.compile(r"\btpu[-_\s]*v5e\b", re.I), "tpu_v5e"),
    (re.compile(r"\btpu[-_\s]*v5p\b", re.I), "tpu_v5p"),
    (re.compile(r"\btrillium|tpu[-_\s]*v6e\b", re.I), "tpu_trillium"),
    (re.compile(r"\btrn2\b|trainium2", re.I), "trn2"),
    (re.compile(r"\btrn1\b|trainium\b", re.I), "trn1"),
)

_ACCEL_COUNT = re.compile(r"\b(\d+)\s*x\s*", re.I)


def normalise_system(system_string: str) -> tuple[str | None, int | None]:
    """Parse ``"Dell PowerEdge XE9680, 8x H100 SXM5-80GB"`` → ``("h100_sxm5_80gb", 8)``.

    Returns ``(gpu_id_or_None, accel_count_or_None)``. The caller decides
    whether an unmappable row is a bug or an acceptable skip.
    """
    gpu_id: str | None = None
    for pat, gid in _SYSTEM_PATTERNS:
        if pat.search(system_string):
            gpu_id = gid
            break

    count: int | None = None
    m = _ACCEL_COUNT.search(system_string)
    if m:
        try:
            count = int(m.group(1))
        except ValueError:
            count = None
    return gpu_id, count


# ---------------------------------------------------------------------------
# CSV read
# ---------------------------------------------------------------------------

# The MLPerf results CSV column names drift round to round. We try several.
_COL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "submitter": ("Submitter", "submitter", "Organization"),
    "system": ("System", "system", "System Name", "Platform"),
    "model_task": ("Benchmark", "Model", "benchmark", "Task"),
    "scenario": ("Scenario", "scenario"),
    "result": ("Result", "Performance", "Tokens/s", "performance", "result"),
    "availability": ("Availability", "availability", "Submission Type"),
    "units": ("Units", "units"),
}


def _first_column(df: pd.DataFrame, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in df.columns:
            return k
    return None


def read_results_csv(csv_path: str | Path, round_tag: str) -> list[MlperfRow]:
    """Read one MLPerf results CSV. Rows that can't be normalised are skipped.

    ``round_tag`` is a free-form label ("v5.0", "v5.0-datacenter") that
    propagates to every row so downstream joins keep track of provenance.
    """
    df = pd.read_csv(csv_path)

    submitter_col = _first_column(df, _COL_CANDIDATES["submitter"])
    system_col = _first_column(df, _COL_CANDIDATES["system"])
    model_col = _first_column(df, _COL_CANDIDATES["model_task"])
    scenario_col = _first_column(df, _COL_CANDIDATES["scenario"])
    result_col = _first_column(df, _COL_CANDIDATES["result"])
    avail_col = _first_column(df, _COL_CANDIDATES["availability"])

    if not all((submitter_col, system_col, model_col, scenario_col, result_col)):
        missing = [
            k
            for k, v in {
                "submitter": submitter_col,
                "system": system_col,
                "model": model_col,
                "scenario": scenario_col,
                "result": result_col,
            }.items()
            if not v
        ]
        raise ValueError(
            f"MLPerf CSV {csv_path} missing required column(s): {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    rows: list[MlperfRow] = []
    for _, r in df.iterrows():
        gpu_id, count = normalise_system(str(r[system_col]))
        result_val = r[result_col]
        try:
            result_f = float(result_val)
        except (TypeError, ValueError):
            result_f = None  # type: ignore[assignment]

        rows.append(
            MlperfRow(
                round=round_tag,
                submitter=str(r[submitter_col]),
                system=str(r[system_col]),
                gpu_id=gpu_id,
                accelerator_count=count,
                model_task=str(r[model_col]),
                scenario=str(r[scenario_col]),
                tokens_per_second=result_f,
                availability=str(r[avail_col]) if avail_col else None,
            )
        )
    return rows


def load_bundled_results(mlperf_dir: Path = MLPERF_DIR) -> list[MlperfRow]:
    """Read every ``*.csv`` under ``data/raw/mlperf/`` and merge.

    File-naming convention: ``<round>__<anything>.csv`` where ``<round>`` is
    the tag we want on each row (e.g. ``v5.0__datacenter.csv``).
    """
    if not mlperf_dir.exists():
        return []
    all_rows: list[MlperfRow] = []
    for csv_path in sorted(mlperf_dir.glob("*.csv")):
        round_tag = csv_path.stem.split("__", 1)[0]
        all_rows.extend(read_results_csv(csv_path, round_tag))
    return all_rows


# ---------------------------------------------------------------------------
# Cross-check against roofline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossCheck:
    gpu_id: str
    model_task: str
    scenario: str
    submitter: str
    accelerator_count: int | None
    mlperf_tps_total: float
    mlperf_tps_per_gpu: float | None
    predicted_tps_per_gpu: float
    error_pct: float  # (predicted - measured) / measured * 100

    def as_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "model": self.model_task,
            "scenario": self.scenario,
            "submitter": self.submitter,
            "accel_count": self.accelerator_count,
            "mlperf_tps_total": round(self.mlperf_tps_total, 1),
            "mlperf_tps_per_gpu": (
                round(self.mlperf_tps_per_gpu, 1) if self.mlperf_tps_per_gpu is not None else None
            ),
            "predicted_tps_per_gpu": round(self.predicted_tps_per_gpu, 1),
            "error_pct": round(self.error_pct, 1),
        }


def cross_check(
    rows: list[MlperfRow],
    predict_per_gpu_fn,
) -> list[CrossCheck]:
    """Join MLPerf rows against a prediction function.

    ``predict_per_gpu_fn`` takes ``(gpu_id, model_task, scenario)`` and
    returns the roofline predicted tokens/s for ONE GPU, or ``None`` to
    skip the row. The caller owns model-arch lookup so this module stays
    free of model-specific logic.
    """
    out: list[CrossCheck] = []
    for r in rows:
        if r.gpu_id is None or r.tokens_per_second is None:
            continue
        predicted = predict_per_gpu_fn(r.gpu_id, r.model_task, r.scenario)
        if predicted is None:
            continue
        per_gpu = (
            r.tokens_per_second / r.accelerator_count
            if r.accelerator_count and r.accelerator_count > 0
            else r.tokens_per_second
        )
        error_pct = (predicted - per_gpu) / per_gpu * 100.0 if per_gpu > 0 else 0.0
        out.append(
            CrossCheck(
                gpu_id=r.gpu_id,
                model_task=r.model_task,
                scenario=r.scenario,
                submitter=r.submitter,
                accelerator_count=r.accelerator_count,
                mlperf_tps_total=r.tokens_per_second,
                mlperf_tps_per_gpu=per_gpu,
                predicted_tps_per_gpu=predicted,
                error_pct=error_pct,
            )
        )
    return out
