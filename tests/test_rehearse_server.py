from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

from chat_conductor.indexer import index_export
from chat_conductor.rehearse import estimate_tokens, rehearse
from chat_conductor.server import MCP_TOOL_NAMES, create_mcp_server
from chat_conductor.store import connect, init_schema
from test_index_search import _message, _write_export


class RehearseServerTests(unittest.TestCase):
    def test_rehearse_expands_neighbors_and_respects_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            _write_export(export_dir, [_conversation()])
            index_path = root / "index.sqlite3"
            index_export(export_dir, index_path)

            connection = connect(index_path)
            init_schema(connection)
            try:
                block = rehearse(
                    connection,
                    "needle",
                    limit=3,
                    token_budget=500,
                    neighbor_radius=1,
                )
            finally:
                connection.close()

            self.assertLessEqual(estimate_tokens(block), 500)
            self.assertIn("Archived recall for: needle", block)
            self.assertIn("archived view as of", block)
            self.assertIn("turn 0", block)
            self.assertIn("turn 1", block)
            self.assertIn("turn 2", block)
            self.assertIn("needle center", block)
            self.assertIn("treat as a lead", block)

    def test_mcp_exposes_search_and_rehearse_alias_but_not_mine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = create_mcp_server(Path(tmp) / "index.sqlite3")
            tools = asyncio.run(server.list_tools())
            names = {tool.name for tool in tools}
            self.assertEqual(names, set(MCP_TOOL_NAMES))
            self.assertNotIn("mine", names)
            self.assertIn("search_chat_history", names)
            self.assertIn("search", names)

    def test_mcp_search_tool_returns_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            _write_export(export_dir, [_conversation()])
            index_path = root / "index.sqlite3"
            index_export(export_dir, index_path)

            server = create_mcp_server(index_path)
            result = asyncio.run(server.call_tool("search", {"query": "needle", "limit": 2}))
            payload = json.loads(result[1]["result"])
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["conv_uuid"], "conv-rehearse")

    def test_mcp_search_chat_history_returns_recall_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            _write_export(export_dir, [_conversation()])
            index_path = root / "index.sqlite3"
            index_export(export_dir, index_path)

            server = create_mcp_server(index_path)
            result = asyncio.run(server.call_tool(
                "search_chat_history",
                {"query": "needle", "limit": 2, "token_budget": 500},
            ))
            block = result[1]["result"]
            self.assertIn("Archived recall for: needle", block)
            self.assertIn("archived view as of", block)
            self.assertIn("needle center", block)


def _conversation() -> dict:
    return {
        "uuid": "conv-rehearse",
        "name": "Rehearse",
        "created_at": "2026-05-01T00:00:00Z",
        "updated_at": "2026-05-01T00:00:00Z",
        "chat_messages": [
            _message("h1", "human", "before question", "2026-05-01T00:00:01Z"),
            _message("a1", "assistant", "before answer", "2026-05-01T00:00:02Z"),
            _message("h2", "human", "needle center", "2026-05-01T00:00:03Z"),
            _message("a2", "assistant", "center answer", "2026-05-01T00:00:04Z"),
            _message("h3", "human", "after question", "2026-05-01T00:00:05Z"),
            _message("a3", "assistant", "after answer", "2026-05-01T00:00:06Z"),
        ],
    }


if __name__ == "__main__":
    unittest.main()
