"""Build 1 — inference cost + build-vs-rent calculator.

The landing page. Multi-page via ``pages/`` so Streamlit renders the
sidebar automatically.
"""

from __future__ import annotations

# Streamlit Cloud entry-point shim: ensure the repo root is on sys.path so
# ``from core.X import ...`` resolves even if the editable install from
# requirements.txt (``-e .``) silently fails. Harmless locally (duplicate
# entries get deduped).
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="AI Compute Economics — Build 1",
    page_icon="⚙️",
    layout="wide",
)

st.title("Inference cost & build-vs-rent calculator")

st.markdown(
    """
This workbench models the cost of serving open-weight LLMs from hardware
specs to cost per million tokens, and compares owning versus renting versus
per-token APIs.

**Every number is tagged `measured`, `modelled`, `simulated` or `assumption`** —
hover any figure for its source.

#### Pages

- **Spec** — GPU catalog with datasheet citations. Memory fit for a model
  and batch at a glance.
- **Validation** — predicted versus measured tokens/s on free-tier T4,
  cross-checked against MLPerf-reported figures.
- **TCO** — 3-year total cost of ownership across own/reserved/on-demand/
  neocloud/API, with break-even utilisation and a 40%-demand-miss stress
  test.

#### Status

| Page | Powered by | Data quality |
|---|---|---|
| Spec | `core/gpu_specs/data/gpu_specs.yaml` | datasheet-high for Turing–Hopper; medium for Blackwell; low for Rubin |
| Validation | `core/cost_engine/roofline.py` + `data/processed/t4_smoke_test.json` | measured T4 pending first Kaggle run |
| TCO | `core/cost_engine/tco.py` + `data/prices/prices.parquet` | prices begin accumulating once the weekly cron runs |
"""
)

st.info(
    "See the project spec in [AI_INFRA_PROJECT.md](../AI_INFRA_PROJECT.md) "
    "for the full set of rules enforced here."
)
