"""Sources discipline: every number in the UI traces back to a Source row.

Agents return ``(value, source_id)`` pairs, never raw floats. The UI renders
each figure with its ``kind`` badge (measured / modelled / simulated / assumption)
and a tooltip linking to the citation.
"""

from core.sources.schema import Confidence, Kind, Source
from core.sources.store import SourceStore

__all__ = ["Confidence", "Kind", "Source", "SourceStore"]
