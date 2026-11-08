"""Scenario preset chips.

A horizontal row of one-click scenarios at the top of Spec, Validation and
TCO. Clicking writes the scenario to ``st.session_state`` under a per-page
prefix; widgets read back via ``st.session_state.get(..., default)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Scenario:
    """One preset. Pages use the fields that matter to them and ignore the rest."""

    key: str                  # stable id, e.g. "nscale_70b_h100"
    label: str                # chip label, e.g. "Nscale 70B on H100"
    description: str          # tooltip text
    gpu_id: str
    model_name: str           # "Llama-3-70B" etc., keyed to each page's model dict
    batch: int
    busy: float
    in_len: int
    out_len: int
    util: float


# Headline + two alternatives. Default = Nscale/Nebius neocloud angle.
NSCALE_70B_H100 = Scenario(
    key="nscale_70b_h100",
    label="Nscale 70B on H100 (default)",
    description=(
        "Llama-3-70B served on 8× H100 SXM 80GB at $2.49/GPU-hour, 60% busy. "
        "The Nscale/Nebius neocloud unit-economics demo."
    ),
    gpu_id="h100_sxm5_80gb",
    model_name="Llama-3-70B",
    batch=32,
    busy=0.60,
    in_len=512,
    out_len=256,
    util=0.65,
)

MIDSIZE_8B_L40S = Scenario(
    key="midsize_8b_l40s",
    label="Midsize 8B on L40S",
    description="Llama-3-8B on a single L40S — the 'small team shipping production' slot.",
    gpu_id="l40s",
    model_name="Llama-3-8B",
    batch=16,
    busy=0.55,
    in_len=1024,
    out_len=384,
    util=0.65,
)

EDGE_3B_L4 = Scenario(
    key="edge_3b_l4",
    label="Edge 3B on L4",
    description="Qwen2.5-3B on a budget L4 — cheapest-serving-anything scenario.",
    gpu_id="l4",
    model_name="Qwen2.5-3B",
    batch=8,
    busy=0.45,
    in_len=512,
    out_len=256,
    util=0.55,
)


SCENARIOS: list[Scenario] = [NSCALE_70B_H100, MIDSIZE_8B_L40S, EDGE_3B_L4]


def preset_chips(
    key_prefix: str,
    scenarios: list[Scenario] | None = None,
) -> Scenario | None:
    """Render a row of preset chips. Returns the Scenario the user just clicked.

    The clicked scenario is also persisted in session state under
    ``{key_prefix}__active_scenario`` so a page reload remembers the last pick.

    Pages read widget defaults from ``st.session_state.get(f"{key_prefix}__{field}", fallback)``.
    """
    options = scenarios if scenarios is not None else SCENARIOS
    st.caption("Try a scenario — click once, tune below.")
    cols = st.columns(len(options))
    clicked: Scenario | None = None
    for col, s in zip(cols, options, strict=True):
        with col:
            if st.button(s.label, help=s.description, key=f"{key_prefix}__btn__{s.key}"):
                clicked = s

    if clicked is not None:
        _apply_scenario(key_prefix, clicked)

    return clicked


def _apply_scenario(key_prefix: str, s: Scenario) -> None:
    """Write scenario fields to session_state under the page's prefix."""
    st.session_state[f"{key_prefix}__active_scenario"] = s.key
    st.session_state[f"{key_prefix}__gpu_id"] = s.gpu_id
    st.session_state[f"{key_prefix}__model_name"] = s.model_name
    st.session_state[f"{key_prefix}__batch"] = s.batch
    st.session_state[f"{key_prefix}__busy"] = s.busy
    st.session_state[f"{key_prefix}__in_len"] = s.in_len
    st.session_state[f"{key_prefix}__out_len"] = s.out_len
    st.session_state[f"{key_prefix}__util"] = s.util


def scenario_default(key_prefix: str, field: str, fallback):  # noqa: ANN001, ANN201
    """Read a widget default — scenario pick if the user clicked, otherwise fallback."""
    return st.session_state.get(f"{key_prefix}__{field}", fallback)
