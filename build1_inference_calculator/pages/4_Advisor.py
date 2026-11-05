"""Compute advisor chat panel — Build 1 agent.

Transparent tool-calling UI. For every user turn:
- The agent's recommendation is shown as markdown.
- Every tool call it made is listed with its name, inputs and result.

If no ANTHROPIC_API_KEY is set, the page renders a tool explorer instead
so the UI is still usable without live API spend.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from core.agents.compute_advisor import (  # noqa: E402
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    ComputeAdvisor,
    run_tool,
)

st.set_page_config(page_title="Compute Advisor", page_icon="🛠️", layout="wide")
st.title("Compute advisor")

st.caption(
    "Ask a sizing or build-vs-rent question. Every number in the answer is a tool call "
    "against `core/cost_engine` — the agent never calculates a figure. "
    "The tool-call panel below shows exactly what was called and with what inputs."
)

api_key = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
if not api_key:
    st.warning(
        "No `ANTHROPIC_API_KEY` set. Showing the tool explorer so the page is "
        "usable without live API spend. Set the key to enable the chat."
    )
    st.header("Tool explorer")
    tool_names = sorted(TOOL_REGISTRY.keys())
    selected = st.selectbox("Tool", options=tool_names)
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == selected)
    st.markdown(f"**{schema['description']}**")
    st.json(schema["input_schema"])
    raw = st.text_area(
        "Parameters (JSON)",
        value=json.dumps(
            {
                k: v.get("default", "" if v["type"] == "string" else 0)
                for k, v in schema["input_schema"]["properties"].items()
                if k in schema["input_schema"].get("required", [])
            },
            indent=2,
        ),
        height=200,
    )
    if st.button("Run tool", type="primary"):
        try:
            params = json.loads(raw)
            result = run_tool(selected, params)
            st.success("Tool call succeeded")
            st.json(result)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Live chat
# ---------------------------------------------------------------------------
if "advisor" not in st.session_state:
    st.session_state.advisor = ComputeAdvisor(api_key=api_key)
if "history" not in st.session_state:
    st.session_state.history = []

st.header("Chat")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("tool_calls"):
            with st.expander(f"Tool calls ({len(turn['tool_calls'])})"):
                for tc in turn["tool_calls"]:
                    st.markdown(f"**{tc['name']}**")
                    st.json({"input": tc["params"], "output": tc["result"], "error": tc.get("error")})

question = st.chat_input("Serve 70B at 50 rps for 3 years — rent, buy or API?")
if question:
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = st.session_state.advisor.ask(question)
            except Exception as e:
                st.error(f"Agent failed: {type(e).__name__}: {e}")
                st.stop()
        st.markdown(resp.text)
        tool_dicts = [
            {"name": tc.name, "params": tc.params, "result": tc.result, "error": tc.error}
            for tc in resp.tool_calls
        ]
        if tool_dicts:
            with st.expander(f"Tool calls ({len(tool_dicts)})"):
                for tc in tool_dicts:
                    st.markdown(f"**{tc['name']}**")
                    st.json({"input": tc["params"], "output": tc["result"], "error": tc.get("error")})
        if resp.usage:
            st.caption(
                f"Tokens — in: {resp.usage['input_tokens']}, out: {resp.usage['output_tokens']}, "
                f"cache read: {resp.usage['cache_read_input_tokens']}, "
                f"cache create: {resp.usage['cache_creation_input_tokens']}"
            )
    st.session_state.history.append(
        {"role": "assistant", "text": resp.text, "tool_calls": tool_dicts}
    )
