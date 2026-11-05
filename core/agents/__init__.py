"""Agents that answer sizing questions by calling tool-wrapped core functions.

The one rule: the agent never calculates a figure. Every number in its
output is the return value of a tool call. The tool registry makes that
testable — the eval harness runs the agent against known answers and
reports a pass rate.
"""

from core.agents.compute_advisor import (
    TOOL_REGISTRY,
    ComputeAdvisor,
    ToolCall,
    run_tool,
)

__all__ = ["TOOL_REGISTRY", "ComputeAdvisor", "ToolCall", "run_tool"]
