from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

from chat_conductor.evaluator import load_eval_cases, report_to_dict, run_eval
from chat_conductor.indexer import index_export
from chat_conductor.store import connect, init_schema, search
from test_index_search import _message, _write_export


class EvalTests(unittest.TestCase):
    def test_eval_reports_bilingual_recall_at_k(self) -> None:
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
                ai_turn = search(connection, "AI", limit=1)[0].turn_id
                cjk_turn = search(connection, "信仰", limit=1)[0].turn_id
                cases_path = root / "eval.json"
                cases_path.write_text(
                    json.dumps([
                        {"query": "AI", "expected_turn_id": ai_turn},
                        {"query": "信仰", "expected_turn_ids": [cjk_turn], "k": 3},
                    ], ensure_ascii=False),
                    encoding="utf-8",
                )
                report = run_eval(connection, load_eval_cases(cases_path), default_k=5)
            finally:
                connection.close()

            payload = report_to_dict(report)
            self.assertEqual(payload["cases"], 2)
            self.assertEqual(payload["hits"], 2)
            self.assertEqual(payload["recall_at_k"], 1.0)
            self.assertEqual(payload["misses"], [])


def _conversation() -> dict:
    return {
        "uuid": "conv-eval",
        "name": "Eval",
        "created_at": "2026-05-01T00:00:00Z",
        "updated_at": "2026-05-01T00:00:00Z",
        "chat_messages": [
            _message("h1", "human", "AI eval case", "2026-05-01T00:00:01Z"),
            _message("a1", "assistant", "信仰 and 因果 eval case", "2026-05-01T00:00:02Z"),
        ],
    }


if __name__ == "__main__":
    unittest.main()
