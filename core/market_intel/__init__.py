"""Market intelligence extractor.

Takes earnings-transcript / press-release / news text the owner drops into
``data/raw/market/`` and turns it into structured facts (GPU prices,
capex, shipment volumes, capacity, contracts). Facts feed the price
tracker and the Build-1 Streamlit pages.

No scraping — input is text already on disk. Sibling repo
``AI Investment RAG`` handles acquisition; this module only extracts.
"""

from core.market_intel.extractor import (
    EXTRACT_SYSTEM_PROMPT,
    FACT_TOOL_SCHEMA,
    FakeExtractorClient,
    extract_facts,
)
from core.market_intel.schema import (
    FactKind,
    MarketIntelDoc,
    MarketIntelFact,
)
from core.market_intel.store import MarketIntelStore

__all__ = [
    "EXTRACT_SYSTEM_PROMPT",
    "FACT_TOOL_SCHEMA",
    "FactKind",
    "FakeExtractorClient",
    "MarketIntelDoc",
    "MarketIntelFact",
    "MarketIntelStore",
    "extract_facts",
]
