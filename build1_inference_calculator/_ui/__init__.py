"""Shared UI primitives for Build 1.

Streamlit pages consume these so copy, display names, glossary, cost
calc and tool-call rendering stay consistent across the five pages.

Pure-logic modules (no streamlit import) so tests run headless:
  - display_names
  - explain
  - cost_counter

Streamlit rendering modules:
  - page_header
  - presets
  - provenance
  - tool_card
  - sidebar
"""
