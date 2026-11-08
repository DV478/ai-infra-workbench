"""Market prices — live prices + capex backing the TCO assumptions.

Reframed from the old 'empty-state → go run a script' page. Now the page
auto-populates on first load from the committed demo corpus so a visitor
never sees an empty box.

Flow:
1. Cold-start auto-populate — if data/market_intel.duckdb is missing,
   walk data/raw/market/*.{txt,md} through the FakeExtractorClient and
   write facts. Zero API spend.
2. page_header + headline tiles.
3. Price-over-time chart (hero visualisation).
4. Filterable fact table with verbatim quotes.
5. GPU price snapshots — min observed, max observed, latest citation date.
6. Capex-by-company with a period filter (fixes the old 'sum across periods' bug).
7. Methodology expander with gold-eval + quote-valid numbers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD1 = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _BUILD1):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from _ui.display_names import fact_kind_display  # noqa: E402
from _ui.explain import caption, tooltip  # noqa: E402
from _ui.page_header import page_header  # noqa: E402
from _ui.provenance import badge, confidence_badge  # noqa: E402

from core.market_intel import (  # noqa: E402
    FactKind,
    FakeExtractorClient,
    MarketIntelDoc,
    MarketIntelStore,
    extract_facts,
)

ROOT = _REPO_ROOT
RAW_DIR = ROOT / "data" / "raw" / "market"
STORE_PATH = ROOT / "data" / "market_intel.duckdb"

st.title("Market prices")

page_header(
    what="Live prices and capex figures extracted from AI-infra press releases, "
         "earnings transcripts, and news.",
    why="Every row backs a TCO assumption. 'Neocloud at $2.49/hr' isn't a guess "
        "if there's a dated quote for it.",
    start="Filter by fact kind or company. Click a row for the exact source quote.",
    numbers="One row = one extracted fact. Each carries the verbatim sentence "
            "it was pulled from.",
)

# ---------------------------------------------------------------------------
# Cold-start auto-populate (zero API spend)
# ---------------------------------------------------------------------------
_FILENAME_RE = re.compile(
    r"^(?P<doc_type>[a-z_]+)__(?P<company>[a-z0-9\-]+)__(?P<date>\d{4}-\d{2}-\d{2})__(?P<slug>[a-z0-9\-]+)"
)


def _fake_callback(doc: MarketIntelDoc) -> list[dict]:
    """Mirror of scripts/run_market_intel.py::_fake_callback — rule-based
    extraction for zero-spend cold-start. Kept inline rather than imported
    so the page has no dependency on scripts/.
    """
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
            "subject": "unspecified",
            "value_num": float(m.group(1)),
            "value_unit": "usd_per_hour",
            "quote": snippet.strip(),
            "confidence": "medium",
        })
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*MW", body):
        snippet = body[max(0, m.start() - 40): m.end() + 40]
        out.append({
            "fact_kind": "capacity",
            "subject": doc.company or "unspecified",
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


@st.cache_resource
def _cold_start_populate(_raw_dir: Path, _store_path: Path) -> bool:
    """Run the fake extractor over the committed demo corpus if the DB is missing.

    Cached with ``st.cache_resource`` so we only do it once per server lifetime.
    Returns True if we populated the DB; False if it already existed.
    """
    if _store_path.exists():
        return False
    if not _raw_dir.exists():
        return False
    raw_files = sorted(_raw_dir.glob("*.txt")) + sorted(_raw_dir.glob("*.md"))
    if not raw_files:
        return False

    _store_path.parent.mkdir(parents=True, exist_ok=True)
    store = MarketIntelStore(_store_path)
    client = FakeExtractorClient(callback=_fake_callback)
    try:
        for p in raw_files:
            m = _FILENAME_RE.match(p.stem)
            parts = m.groupdict() if m else {
                "doc_type": "news", "company": None, "date": None, "slug": p.stem,
            }
            body = p.read_text(encoding="utf-8")
            title_line = body.splitlines()[0].strip() if body else (parts["slug"] or p.stem)
            doc = MarketIntelDoc(
                doc_type=parts["doc_type"] or "news",
                title=title_line[:200],
                body=body,
                company=parts["company"],
                published_date=parts["date"],
            )
            store.add_doc(doc)
            facts = extract_facts(doc, client=client)
            for f in facts:
                store.add_fact(f)
    finally:
        store.close()
    return True


populated = _cold_start_populate(RAW_DIR, STORE_PATH)

if not STORE_PATH.exists():
    st.warning(
        "No `data/market_intel.duckdb` and no committed demo corpus found. "
        "Drop text files into `data/raw/market/` and reload."
    )
    st.stop()

if populated:
    st.info(
        f"{badge('simulated')} Showing facts from the committed **synthetic demo "
        "corpus** (zero API spend). In production, set `ANTHROPIC_API_KEY` or "
        "`OPENAI_API_KEY` and run `scripts/run_market_intel.py` for real extraction."
    )
else:
    st.success(f"{badge('measured')} Showing facts extracted from the real corpus.")

# ---------------------------------------------------------------------------
# Load facts
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Headline tiles
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total facts", len(df), help="Every fact extracted across every document.")
c2.metric("GPU prices", summary.get("gpu_price", 0),
          help="Price observations — hourly rates per GPU type.")
c3.metric("Capex rows", summary.get("capex", 0),
          help="Announced datacenter investments.")
c4.metric("Shipments", summary.get("gpu_shipment", 0),
          help="Unit counts for GPU deliveries.")
c5.metric("Companies", df["company"].dropna().nunique(),
          help="Distinct companies named across the corpus.")

# ---------------------------------------------------------------------------
# Hero chart — GPU prices observed over time
# ---------------------------------------------------------------------------
gpu_price = df[(df["fact_kind"] == "gpu_price") & (df["value_num"].notna())].copy()
if len(gpu_price):
    gpu_price["company"] = gpu_price["company"].fillna("unspecified")

    st.markdown("## GPU prices observed")
    fig = go.Figure()
    for company in sorted(gpu_price["company"].unique()):
        subset = gpu_price[gpu_price["company"] == company]
        fig.add_trace(go.Scatter(
            x=subset["extracted_at"],
            y=subset["value_num"],
            mode="markers",
            name=company,
            marker={"size": 11},
            hovertemplate="<b>%{text}</b><br>$%{y:.2f}/GPU-hr<extra></extra>",
            text=subset["subject"],
        ))
    fig.update_layout(
        title="$ per GPU-hour by company (all observations)",
        xaxis_title="Extracted at",
        yaxis_title="USD / GPU-hour",
        height=360,
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(caption("verbatim_quote"))

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
kind_labels = {"(all)": "(all)"} | {k.value: fact_kind_display(k) for k in FactKind}
kind_choice_label = st.sidebar.selectbox(
    "Fact kind",
    options=list(kind_labels.values()),
    help=tooltip("fact_kind"),
)
# Reverse-lookup pretty label → raw value
kind_pick = next(k for k, v in kind_labels.items() if v == kind_choice_label)

company_options = ["(all)"] + sorted(df["company"].dropna().unique().tolist())
company_pick = st.sidebar.selectbox("Company", options=company_options)
confidence_pick = st.sidebar.multiselect(
    "Confidence",
    options=["high", "medium", "low"],
    default=["high", "medium", "low"],
    help=tooltip("confidence_badge"),
)

filtered = df.copy()
if kind_pick != "(all)":
    filtered = filtered[filtered["fact_kind"] == kind_pick]
if company_pick != "(all)":
    filtered = filtered[filtered["company"] == company_pick]
if confidence_pick:
    filtered = filtered[filtered["confidence"].isin(confidence_pick)]

st.markdown(f"**Showing {len(filtered)} of {len(df)} facts**")

# ---------------------------------------------------------------------------
# Fact table — quote prominent
# ---------------------------------------------------------------------------
display_df = filtered.copy()
display_df["Kind"] = display_df["fact_kind"].apply(fact_kind_display)
display_df["Confidence"] = display_df["confidence"].apply(confidence_badge)

ordered_cols = {
    "Kind": "Kind",
    "subject": "Subject",
    "value_num": "Value",
    "value_unit": "Unit",
    "period": "Period",
    "company": "Company",
    "Confidence": "Confidence",
    "quote": "Quote (verbatim)",
    "notes": "Notes",
}
show_cols = [c for c in ordered_cols if c in display_df.columns]

st.dataframe(
    display_df[show_cols].rename(columns=ordered_cols),
    hide_index=True,
    width="stretch",
    column_config={
        "Quote (verbatim)": st.column_config.TextColumn("Quote (verbatim)", width="large"),
        "Value": st.column_config.NumberColumn("Value", format="%.2f"),
    },
)

# ---------------------------------------------------------------------------
# GPU price snapshots — fixed labelling (min observed, max observed, latest date)
# ---------------------------------------------------------------------------
st.markdown("## GPU price snapshots")
if len(gpu_price):
    gpu_price["extracted_at"] = pd.to_datetime(gpu_price["extracted_at"], errors="coerce")
    snapshot = (
        gpu_price.groupby("subject")
        .agg(
            min_usd_per_hour_observed=("value_num", "min"),
            max_usd_per_hour_observed=("value_num", "max"),
            observations=("value_num", "count"),
            companies=("company", "nunique"),
            latest_observation=("extracted_at", "max"),
        )
        .reset_index()
        .sort_values("min_usd_per_hour_observed")
    )
    st.dataframe(
        snapshot.rename(columns={
            "subject": "GPU",
            "min_usd_per_hour_observed": "Min observed ($/hr)",
            "max_usd_per_hour_observed": "Max observed ($/hr)",
            "observations": "Observations",
            "companies": "Companies citing",
            "latest_observation": "Latest observation",
        }),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "**Min / max** are the lowest / highest hourly rates observed in the corpus — "
        "not a current market quote. Dates reflect when the quote was extracted."
    )
else:
    st.caption("No GPU-price facts in the store yet.")

# ---------------------------------------------------------------------------
# Capex by company — period filter fixes the 'sum across periods' bug
# ---------------------------------------------------------------------------
st.markdown("## Capex by company")
capex = df[(df["fact_kind"] == "capex") & (df["value_num"].notna())].copy()

if len(capex):
    period_options = ["(most recent period only)", "(all periods — not meaningful)"] + sorted(
        p for p in capex["period"].dropna().unique() if p
    )
    period_pick = st.selectbox(
        "Period filter",
        options=period_options,
        index=0,
        help="Summing capex across different periods adds numbers that don't mean "
             "anything together. Default is the most recent observation per company.",
    )

    if period_pick == "(most recent period only)":
        latest_per_company = (
            capex.sort_values("extracted_at", ascending=False)
            .drop_duplicates(subset=["company"], keep="first")
        )
        cap_agg = (
            latest_per_company.assign(total_usd_billions=lambda d: d["value_num"] / 1e9)
            [["company", "total_usd_billions", "period"]]
            .rename(columns={
                "company": "Company",
                "total_usd_billions": "Most recent capex ($B)",
                "period": "Period",
            })
            .sort_values("Most recent capex ($B)", ascending=False)
        )
        st.dataframe(cap_agg, hide_index=True, width="stretch")
    elif period_pick == "(all periods — not meaningful)":
        st.warning(
            "Summing capex across different periods ($10B 2025 + $5B 2026) produces a "
            "number that doesn't mean anything. Shown for exploration only."
        )
        cap_agg = (
            capex.groupby("company", dropna=False)
            .agg(summed_usd=("value_num", "sum"), observations=("value_num", "count"))
            .reset_index()
        )
        cap_agg["summed_usd_billions"] = cap_agg["summed_usd"] / 1e9
        st.dataframe(
            cap_agg[["company", "summed_usd_billions", "observations"]].rename(columns={
                "company": "Company",
                "summed_usd_billions": "Summed capex ($B) — do not cite",
                "observations": "Observations",
            }),
            hide_index=True,
            width="stretch",
        )
    else:
        period_rows = capex[capex["period"] == period_pick]
        cap_agg = (
            period_rows.assign(usd_billions=lambda d: d["value_num"] / 1e9)
            [["company", "usd_billions"]]
            .rename(columns={"company": "Company", "usd_billions": f"Capex ({period_pick}, $B)"})
            .sort_values(f"Capex ({period_pick}, $B)", ascending=False)
        )
        st.dataframe(cap_agg, hide_index=True, width="stretch")
else:
    st.caption("No capex facts in the store yet.")

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with st.expander("Methodology"):
    st.markdown(
        """
- **Extraction pipeline.** Rule-based `FakeExtractorClient` (zero API spend) for the
  committed demo corpus. Production runs use `AnthropicExtractorClient` or
  `OpenAIExtractorClient` with `tool_choice=emit_facts` forced.
- **Quote verification.** Every fact's `quote` must appear verbatim in the source
  document body (case-folded, whitespace-normalised). If the quote isn't literally
  there, the fact fails verification and isn't stored.
- **Gold evaluation.** 12-case gold set in `core/market_intel/eval.py`. Baseline
  fake-extractor pass rate: 4/12 (33%, 50% recall). Real-LLM runs land ~70%+.
- **Corpus.** 8 synthetic documents under `data/raw/market/` tagged inline as
  `[SYNTHETIC — demo corpus]`. Replace with real press releases, transcripts,
  and news filings to produce production facts.
        """
    )
