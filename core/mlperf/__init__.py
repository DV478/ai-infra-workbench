"""MLPerf Inference result ingestion and cross-check.

MLPerf Inference publishes CSV result bundles each round at
https://mlcommons.org/en/inference-datacenter-50/. We ingest the Llama-2-70B
server and offline rows we care about, normalise them into our GPU id space,
and provide a cross-check function so the Streamlit Validation page can
show "predicted vs measured vs MLPerf-reported".
"""

from core.mlperf.ingest import (
    MlperfRow,
    cross_check,
    load_bundled_results,
    normalise_system,
    read_results_csv,
)

__all__ = [
    "MlperfRow",
    "cross_check",
    "load_bundled_results",
    "normalise_system",
    "read_results_csv",
]
