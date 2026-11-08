"""Unified sidebar — called from ``app.py`` so copy/order stay consistent.

Streamlit renders the page list automatically; we add a header block above
it with the project pitch and the current demo scenario, plus a link row
at the bottom (GitHub, deployed site, methodology).
"""

from __future__ import annotations

import streamlit as st

GITHUB_URL = "https://github.com/DV478/ai-infra-workbench"
DEPLOYED_URL = "https://ai-infra-workbench.streamlit.app"


def render_sidebar() -> None:
    """Call once from ``app.py``. Pages inherit the sidebar automatically."""
    with st.sidebar:
        st.markdown("### AI Compute Economics")
        st.caption(
            "Should you rent, buy, or API an H100 for Llama-3-70B? "
            "A calculator + two AI agents that answer with real tool calls, not guesses."
        )
        st.markdown("---")
        st.caption("Jump to:")
        # Streamlit renders pages below automatically — we just add context above.
        st.markdown(
            f"🌐 [Deployed demo]({DEPLOYED_URL})  \n"
            f"🧪 [Source on GitHub]({GITHUB_URL})"
        )
