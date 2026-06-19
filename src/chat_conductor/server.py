from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from .rehearse import DEFAULT_REHEARSE_BUDGET, DEFAULT_REHEARSE_LIMIT, DEFAULT_NEIGHBOR_RADIUS, rehearse
from .store import connect, init_schema, search

MCP_TOOL_NAMES = ("search", "search_chat_history")


def create_mcp_server(index_path: Path) -> FastMCP:
    server = FastMCP("chat-conductor")

    @server.tool(name="search", description="Search archived chat segments.")
    def search_tool(
        query: str,
        limit: int = 10,
        after: str | None = None,
        before: str | None = None,
        role: Literal["human", "assistant"] | None = None,
        conv: str | None = None,
    ) -> str:
        connection = connect(index_path)
        try:
            init_schema(connection)
            results = search(
                connection,
                query,
                limit=limit,
                after=after,
                before=before,
                role=role,
                conv=conv,
            )
        finally:
            connection.close()
        return json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2)

    @server.tool(
        name="search_chat_history",
        description="Return a budgeted archived recall block for the current task.",
    )
    def search_chat_history(
        query: str,
        limit: int = DEFAULT_REHEARSE_LIMIT,
        token_budget: int = DEFAULT_REHEARSE_BUDGET,
        neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
        after: str | None = None,
        before: str | None = None,
        role: Literal["human", "assistant"] | None = None,
        conv: str | None = None,
    ) -> str:
        connection = connect(index_path)
        try:
            init_schema(connection)
            return rehearse(
                connection,
                query,
                limit=limit,
                token_budget=token_budget,
                neighbor_radius=neighbor_radius,
                after=after,
                before=before,
                role=role,
                conv=conv,
            )
        finally:
            connection.close()

    return server


def run_server(
    index_path: Path,
    *,
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
) -> None:
    create_mcp_server(index_path).run(transport=transport)
