"""Standard page header used on every Build 1 page.

Four short lines — what / why / where to start / what numbers mean — rendered
inside a bordered container so a non-technical reader always has the same
orientation block in the same place.
"""

from __future__ import annotations

import streamlit as st


def page_header(
    *,
    what: str,
    why: str,
    start: str,
    numbers: str,
) -> None:
    """Render the standard four-line page orientation block.

    Parameters
    ----------
    what    : One sentence. "What is this page?"
    why     : One sentence. "Why should I care?"
    start   : One sentence. "Where do I click first?"
    numbers : One sentence. "What do the numbers mean / what unit?"
    """
    with st.container(border=True):
        st.markdown(
            f"**What:** {what}  \n"
            f"**Why:** {why}  \n"
            f"**Start:** {start}  \n"
            f"**Numbers:** {numbers}"
        )
