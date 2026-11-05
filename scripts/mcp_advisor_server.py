"""MCP server exposing the compute advisor's tools.

Wraps ``core.agents.compute_advisor.TOOL_REGISTRY`` as an MCP server over
stdio so the agent can run from Claude Desktop or Claude Code inside the
existing coding-tool plan — **no Anthropic API spend**. This is the
budget-first path from the project spec.

Setup (one-time):

    pip install "mcp>=1.0"

Register with Claude Desktop (``~/Library/Application Support/Claude/
claude_desktop_config.json`` on macOS, ``%APPDATA%\\Claude\\claude_desktop_config.json``
on Windows):

    {
      "mcpServers": {
        "compute-advisor": {
          "command": "python",
          "args": ["c:/Users/DV/OneDrive/Desktop/Claude/AI_INFRA_PROJECT/scripts/mcp_advisor_server.py"],
          "env": {}
        }
      }
    }

Restart Claude Desktop; the ten tools appear in the tool picker.

The server has no model, no system prompt, no agent loop — it is a thin
translator from MCP's JSON-RPC to ``run_tool(name, params)``. The system
prompt that drives the sizing agent lives client-side (you can paste
``SYSTEM_PROMPT`` from ``core/agents/compute_advisor.py`` into Claude
Desktop's project instructions).
"""

from __future__ import annotations

import asyncio
import json
import sys

from core.agents.compute_advisor import TOOL_REGISTRY, TOOL_SCHEMAS, run_tool


def _format_result(result: dict) -> str:
    """MCP tool results are strings; return JSON so Claude can parse it back."""
    return json.dumps(result, indent=2)


async def _run_server() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError:
        print(
            "mcp SDK not installed. Run: pip install 'mcp>=1.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    server = Server("compute-advisor")

    @server.list_tools()
    async def list_tools() -> list:
        return [
            Tool(
                name=s["name"],
                description=s["description"],
                inputSchema=s["input_schema"],
            )
            for s in TOOL_SCHEMAS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        if name not in TOOL_REGISTRY:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Unknown tool {name!r}. "
                            f"Available: {sorted(TOOL_REGISTRY.keys())}"
                        }
                    ),
                )
            ]
        try:
            result = run_tool(name, arguments)
            return [TextContent(type="text", text=_format_result(result))]
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"{type(e).__name__}: {e}"}),
                )
            ]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> int:
    asyncio.run(_run_server())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
