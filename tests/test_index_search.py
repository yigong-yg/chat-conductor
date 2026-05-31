from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from chat_conductor.export import conversations_json_from_export
from chat_conductor.indexer import index_export
from chat_conductor.store import connect, init_schema, search, status


class IndexSearchTests(unittest.TestCase):
    def test_indexes_searches_and_is_idempotent_with_timestamp_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            _write_export(export_dir, _fixture("hello keyword"))
            index_path = root / "index.sqlite3"

            first = index_export(export_dir, index_path)
            second = index_export(export_dir, index_path)

            self.assertEqual(first.conversations, 2)
            self.assertEqual(first.messages, 6)
            self.assertEqual(first.appended, 4)
            self.assertEqual(first.superseded, 0)
            self.assertEqual(second.appended, 0)
            self.assertEqual(second.skipped, 4)

            connection = connect(index_path)
            init_schema(connection)
            try:
                results = search(connection, "keyword", limit=5)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].conv_title, "Alpha")
                self.assertEqual(results[0].role, "turn")
                self.assertIn("hello keyword", results[0].text)

                human_results = search(connection, "hello", role="human")
                self.assertEqual(len(human_results), 1)
                self.assertEqual(human_results[0].role, "human")
                self.assertIn("hello keyword", human_results[0].text)

                assistant_results = search(connection, "answer", role="assistant")
                self.assertEqual(len(assistant_results), 1)
                self.assertEqual(assistant_results[0].role, "assistant")

                filtered = search(connection, "keyword", conv="Alpha")
                self.assertEqual(len(filtered), 1)
                content_block = search(connection, "content", role="human")
                self.assertEqual(len(content_block), 1)
                self.assertIn("content block text", content_block[0].text)
                self.assertEqual(status(connection)["active_turns"], 4)
            finally:
                connection.close()

    def test_supersedes_when_same_message_identity_changes_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            index_path = root / "index.sqlite3"

            _write_export(export_dir, _fixture("original keyword"))
            first = index_export(export_dir, index_path)
            _write_export(export_dir, _fixture("edited keyword"))
            second = index_export(export_dir, index_path)

            self.assertEqual(first.appended, 4)
            self.assertEqual(second.superseded, 1)
            self.assertEqual(second.skipped, 3)

            connection = connect(index_path)
            init_schema(connection)
            try:
                current = search(connection, "edited", limit=5)
                old = search(connection, "original", limit=5)
                self.assertEqual(len(current), 1)
                self.assertEqual(len(old), 0)
                index_status = status(connection)
                self.assertEqual(index_status["active_turns"], 4)
                self.assertEqual(index_status["superseded_turns"], 1)
            finally:
                connection.close()

    def test_accepts_zip_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            _write_export(export_dir, _fixture("zip keyword"))
            zip_path = root / "export.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(export_dir / "conversations.json", "data/conversations.json")

            with conversations_json_from_export(zip_path) as conversations_path:
                self.assertEqual(conversations_path.name, "conversations.json")

            stats = index_export(zip_path, root / "index.sqlite3")
            self.assertEqual(stats.appended, 4)

    def test_short_and_cjk_terms_are_searchable(self) -> None:
        # MAJOR #1 regression lock. FTS5 trigram returns nothing for terms shorter
        # than 3 codepoints ("AI", "ML", 2-char CJK like 信仰) — the archive's most
        # common query shapes. Such terms must route to a LIKE broad-net so search
        # never silently returns zero, and a short term must not AND-zero a query.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            convs = [
                {
                    "uuid": "c1",
                    "name": "C1",
                    "created_at": "2026-05-01T00:00:00Z",
                    "updated_at": "2026-05-01T00:00:00Z",
                    "chat_messages": [
                        _message("m1", "human", "Does AI model faith 信仰 well?", "2026-05-01T00:00:01Z"),
                        _message("m2", "assistant", "人工智能 and ML pipelines here", "2026-05-01T00:00:02Z"),
                    ],
                }
            ]
            _write_export(export_dir, convs)
            index_path = root / "index.sqlite3"
            index_export(export_dir, index_path)

            connection = connect(index_path)
            init_schema(connection)
            try:
                for term in ("AI", "ML", "信仰"):
                    hits = search(connection, term, limit=5)
                    self.assertGreaterEqual(len(hits), 1, f"no hits for short term {term!r}")
                    self.assertTrue(hits[0].turn_id)
                    self.assertTrue(hits[0].conv_uuid)
                    self.assertTrue(hits[0].date)
                # mixed short + long: the short term must narrow, not zero, the query.
                mixed = search(connection, "AI model", limit=5)
                self.assertGreaterEqual(len(mixed), 1, "AND-poisoning: short term zeroed the query")
            finally:
                connection.close()

    def test_message_set_change_currently_orphans_old_turn(self) -> None:
        # CHARACTERIZATION of MAJOR #2 (spec §10 turn-boundary supersede). When a
        # turn's message SET changes — e.g. the assistant reply is regenerated and
        # gets a NEW uuid — the turn's source_key changes, so upsert_turn cannot
        # match the old active row and APPENDS a new turn, leaving the old one
        # active=1. Two active turns then represent one logical position; a
        # re-export duplicates instead of superseding.
        #
        # This pins the CURRENT (buggy) behavior as a tripwire. DESIRED post-fix:
        # active_turns == 1 and a search for the stale reply returns 0. The fix
        # shape (overlap-deactivate vs position-key) is deferred pending confirmed
        # real regen behavior — when fixed, flip the assertions below.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            index_path = root / "index.sqlite3"

            base = {
                "uuid": "conv-x",
                "name": "X",
                "created_at": "2026-05-01T00:00:00Z",
                "updated_at": "2026-05-01T00:00:00Z",
            }
            v1 = [dict(base, chat_messages=[
                _message("h1", "human", "the question", "2026-05-01T00:00:01Z"),
                _message("a1", "assistant", "reply alpha original", "2026-05-01T00:00:02Z"),
            ])]
            v2 = [dict(base, chat_messages=[
                _message("h1", "human", "the question", "2026-05-01T00:00:01Z"),
                _message("a2", "assistant", "reply beta regenerated", "2026-05-01T00:00:02Z"),
            ])]

            _write_export(export_dir, v1)
            index_export(export_dir, index_path)
            _write_export(export_dir, v2)
            index_export(export_dir, index_path)

            connection = connect(index_path)
            init_schema(connection)
            try:
                # CURRENT behavior: regenerated turn appended, old left active.
                self.assertEqual(status(connection)["active_turns"], 2)
                self.assertEqual(len(search(connection, "alpha", limit=5)), 1)
                self.assertEqual(len(search(connection, "beta", limit=5)), 1)
            finally:
                connection.close()


def _write_export(root: Path, conversations: list[dict]) -> None:
    (root / "conversations.json").write_text(
        json.dumps(conversations, ensure_ascii=False),
        encoding="utf-8",
    )


def _fixture(first_human_text: str) -> list[dict]:
    return [
        {
            "uuid": "conv-alpha",
            "name": "Alpha",
            "created_at": "2026-05-01T00:00:00Z",
            "updated_at": "2026-05-01T00:00:00Z",
            "chat_messages": [
                _message("h1", "human", first_human_text, "2026-05-01T00:00:01Z"),
                _message("h2", "human", "second same timestamp", "2026-05-01T00:00:01Z"),
                _message("a1", "assistant", "assistant answer", "2026-05-01T00:00:02Z"),
                _message("h3", "human", "中文测试", "2026-05-01T00:00:03Z"),
            ],
        },
        {
            "uuid": "conv-beta",
            "name": "Beta",
            "created_at": "2026-05-02T00:00:00Z",
            "updated_at": "2026-05-02T00:00:00Z",
            "chat_messages": [
                _message("a2", "assistant", "leading assistant", "2026-05-02T00:00:01Z"),
                {
                    "uuid": "h4",
                    "sender": "human",
                    "text": "",
                    "content": [{"type": "text", "text": "content block text"}],
                    "created_at": "2026-05-02T00:00:02Z",
                },
            ],
        },
    ]


def _message(uuid: str, sender: str, text: str, created_at: str) -> dict:
    return {
        "uuid": uuid,
        "sender": sender,
        "text": text,
        "created_at": created_at,
        "content": [],
    }


if __name__ == "__main__":
    unittest.main()
