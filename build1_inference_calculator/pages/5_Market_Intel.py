"""Market intelligence facts — Build 1.

Renders every extracted fact in ``data/market_intel.duckdb`` with filters
by kind and company. Each row shows the verbatim quote — the citation the
extractor refuses to invent.

To populate, drop text files into ``data/raw/market/`` and run:

    python scripts/run_market_intel.py --fake          # no API spend
    python scripts/run_market_intel.py                 # real Claude extraction
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.market_intel import FactKind, MarketIntelStore  # noqa: E402

ROOT = _REPO_ROOT
STORE_PATH = ROOT / "data" / "market_intel.duckdb"

st.set_page_config(page_title="Market Intel", page_icon="📊", layout="wide")
st.title("Market intelligence")

st.caption(
    "Structured facts extracted from earnings transcripts, press releases, and news. "
    "Every row carries the verbatim quote the extractor used — if the quote is not "
    "in the source document, the fact never makes it here."
)

if not STORE_PATH.exists():
    st.info(
        "No `data/market_intel.duckdb` yet. Drop text files into "
        "`data/raw/market/` and run `python scripts/run_market_intel.py --fake` "
        "(no API spend) or `python scripts/run_market_intel.py` (real extraction)."
    )
    st.stop()

store = MarketIntelStore(STORE_PATH)
try:
    summary = store.summary_by_kind()
    rows = store.recent_facts(limit=500)
finally:
    store.close()

if not rows:
    st.warning("Store exists but is empty. Run the extractor.")
    st.stop()

df = pd.DataFrame(rows)

# ---- Headline tiles -------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total facts", len(df))
c2.metric("GPU prices", summary.get("gpu_price", 0))
c3.metric("Capex rows", summary.get("capex", 0))
c4.metric("Shipments", summary.get("gpu_shipment", 0))
c5.metric("Companies", df["company"].dropna().nunique())

# ---- Filters --------------------------------------------------------------
st.sidebar.header("Filters")
kind_options = ["(all)"] + [k.value for k in FactKind]
kind_pick = st.sidebar.selectbox("Fact kind", options=kind_options)
company_options = ["(all)"] + sorted(df["company"].dropna().unique().tolist())
company_pick = st.sidebar.selectbox("Company", options=company_options)
confidence_pick = st.sidebar.multiselect(
    "Confidence",
    options=["high", "medium", "low"],
    default=["high", "medium", "low"],
)

filtered = df.copy()
if kind_pick != "(all)":
    filtered = filtered[filtered["fact_kind"] == kind_pick]
if company_pick != "(all)":
    filtered = filtered[filtered["company"] == company_pick]
if confidence_pick:
    filtered = filtered[filtered["confidence"].isin(confidence_pick)]

st.markdown(f"**Showing {len(filtered)} of {len(df)} facts**")

# ---- Quote-level view -----------------------------------------------------
display_cols = [
    "fact_kind", "subject", "value_num", "value_unit", "period",
    "company", "confidence", "quote", "notes",
]
st.dataframe(
    filtered[display_cols],
    hide_index=True,
    use_container_width=True,
    column_config={
        "quote": st.column_config.TextColumn("Quote", width="large"),
        "value_num": st.column_config.NumberColumn("Value", format="%.2f"),
    },
)

# ---- Quick aggregates -----------------------------------------------------
st.subheader("GPU price snapshots")
gpu_price = df[(df["fact_kind"] == "gpu_price") & (df["value_num"].notna())]
if len(gpu_price):
    agg = (
        gpu_price.groupby("subject")
        .agg(latest_usd_per_hour=("value_num", "min"),
             rows=("value_num", "count"),
             companies=("company", "nunique"))
        .reset_index()
        .sort_values("latest_usd_per_hour")
    )
    st.dataframe(agg, hide_index=True, use_container_width=True)
else:
    st.caption("No GPU-price facts in the store yet.")

st.subheader("Capex by company")
capex = df[(df["fact_kind"] == "capex") & (df["value_num"].notna())]
if len(capex):
    cap_agg = (
        capex.groupby("company", dropna=False)
        .agg(total_usd=("value_num", "sum"), rows=("value_num", "count"))
        .reset_index()
        .sort_values("total_usd", ascending=False)
    )
    cap_agg["total_usd_billions"] = cap_agg["total_usd"] / 1e9
    st.dataframe(
        cap_agg[["company", "total_usd_billions", "rows"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "total_usd_billions": st.column_config.NumberColumn("Total ($B)", format="%.2f")
        },
    )
else:
    st.caption("No capex facts in the store yet.")
