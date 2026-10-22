from __future__ import annotations

import itertools
import json
import sqlite3
import unittest
from unittest import mock

import _path  # noqa: F401

from evidence_support import NAMESPACE, EvidenceCase, message, numbered, reseal

from chat_conductor.evidence import budget
from chat_conductor.evidence.errors import BudgetExceeded, InvalidInput
from chat_conductor.evidence.receipts import load_receipt
from chat_conductor.evidence.sqlquery import SqlLimits, sql_query
from chat_conductor.evidence.sqlsandbox import open_user_sql_connection
from chat_conductor.evidence.verify import verify_run

SCOPE = {
    "definition": "messages in the synthetic test conversation",
    "unit": "exported message records",
    "filters": "none beyond the SQL",
    "missing_data": "only imported synthetic records are counted",
}
SENTINEL = "SENTINEL-LATER-REVISION 哨兵"


class SqlIsolationTests(EvidenceCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.new_dataset()
        self.ingest(self.dataset, {"conv-a": numbered("a", 30)})
        self.run = self.run_for(self.dataset)
        self.ingest(self.dataset, {"conv-a": [message("late", "ana", "2026-03-01T12:00:00", SENTINEL, order="999999")]}, name="later.json")

    def _sql(self, sql: str, params: dict | None = None, *, limits: SqlLimits = SqlLimits(), **options) -> dict:
        result = sql_query(self.dataset, self.run.run_id, sql, params or {}, options.pop("scope", SCOPE), limits=limits, **options)
        page = json.loads(result.output)
        self.assertEqual(page["response_bytes"], len(result.output))
        return page

    def test_aggregates_run_over_the_full_pinned_snapshot_before_caps(self) -> None:
        page = self._sql("SELECT COUNT(*) AS n FROM visible_messages", limits=SqlLimits(max_rows=1))
        self.assertEqual((page["status"], page["complete"], page["rows"]), ("complete", True, [[30]]))
        grouped = self._sql(
            "SELECT sender_id, COUNT(*) AS n, AVG(length(text)) AS mean_length FROM visible_messages "
            "GROUP BY sender_id ORDER BY sender_id"
        )
        self.assertEqual([row[:2] for row in grouped["rows"]], [["s_ana", 10], ["s_ben", 10], ["s_chen", 10]])
        self.assertIsInstance(grouped["rows"][0][2], float)
        receipt = load_receipt(self.run, page["receipt_id"])[0]
        self.assertEqual(receipt["snapshot"]["rows"], 30)
        self.assertEqual(receipt["snapshot"]["derivation"], "visible-messages-v1")
        self.assertEqual(receipt["arguments"]["scope"], SCOPE)
        self.assertEqual((receipt["columns"], receipt["rows"]), (["n"], [[30]]))
        parameterized = self._sql(
            "SELECT COUNT(*) AS n FROM visible_messages WHERE sender_id = :sender AND text LIKE :pattern ESCAPE '\\'",
            {"sender": "s_ana", "pattern": "%note%"},
        )
        self.assertEqual(parameterized["rows"], [[10]])
        counted = self._sql("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT COUNT(*) AS n FROM c")
        self.assertEqual(counted["rows"], [[5]])
        flattened = self._sql("WITH mine AS (SELECT * FROM visible_messages WHERE sender_id = 's_chen') SELECT COUNT(*) AS n FROM mine")
        self.assertEqual(flattened["rows"], [[10]])

    def test_writes_schema_access_attach_pragmas_and_unsafe_functions_are_denied(self) -> None:
        precheck = {
            "INSERT INTO visible_messages (id) VALUES ('x')": "sql.not_query",
            "UPDATE visible_messages SET text = ''": "sql.not_query",
            "DELETE FROM visible_messages": "sql.not_query",
            "CREATE TABLE t (x)": "sql.not_query",
            "DROP TABLE visible_messages": "sql.not_query",
            "PRAGMA query_only = OFF": "sql.not_query",
            "ATTACH DATABASE 'other.sqlite3' AS other": "sql.not_query",
            "DETACH DATABASE main": "sql.not_query",
            "VACUUM INTO 'copy.sqlite3'": "sql.not_query",
            "BEGIN": "sql.not_query",
            "SELECT 1; SELECT 2": "sql.multiple_statements",
            "SELECT 1; DROP TABLE visible_messages": "sql.multiple_statements",
            "SELECT ?": "sql.parameter_style",
            "SELECT @who": "sql.parameter_style",
        }
        for sql, code in precheck.items():
            with self.subTest(sql=sql):
                with self.assertRaises(InvalidInput) as caught:
                    self._sql(sql)
                self.assertEqual(caught.exception.code, code)
        denied = {
            "SELECT * FROM sqlite_master": "sqlite_master",
            "SELECT count(*) AS n FROM sqlite_schema": "sqlite_schema",
            "SELECT * FROM pragma_table_info('visible_messages')": "pragma_table_info",
            "SELECT * FROM json_each('[1]')": "json_each",
            "SELECT load_extension('evil') AS x": "load_extension",
            "SELECT random() AS x": "random",
            "SELECT date('now') AS x": "date",
            "SELECT group_concat(text) AS x FROM visible_messages": "group_concat",
            "SELECT zeroblob(10) AS x": "zeroblob",
            "SELECT sqlite_version() AS x": "sqlite_version",
            "WITH x AS (SELECT name FROM sqlite_master) SELECT * FROM x": "sqlite_master",
            "WITH sqlite_master AS (SELECT 1 AS x) SELECT COUNT(*) AS n FROM main.sqlite_master": "sqlite_master",
            "WITH c AS (SELECT 1 AS x) SELECT COUNT(*) AS n FROM sqlite_temp_schema": "sqlite_temp_schema",
        }
        for sql, target in denied.items():
            with self.subTest(sql=sql):
                with self.assertRaises(InvalidInput) as caught:
                    self._sql(sql)
                self.assertEqual(caught.exception.code, "sql.not_authorized")
                self.assertEqual(caught.exception.details["denied"][0]["object"], target)
                receipt = load_receipt(self.run, caught.exception.details["receipt_id"])[0]
                self.assertEqual((receipt["status"], receipt["rows"], receipt["error"]["code"]), ("error", [], "sql.not_authorized"))
        for sql in ("SELECT * FROM messages", "SELECT * FROM message_origins", "SELECT * FROM import_batches"):
            with self.subTest(sql=sql):
                with self.assertRaises(InvalidInput) as caught:
                    self._sql(sql)
                self.assertEqual(caught.exception.code, "sql.error")
                self.assertNotIn("messages", caught.exception.message)

    def test_the_locked_connection_itself_refuses_bypasses(self) -> None:
        self._sql("SELECT COUNT(*) AS n FROM visible_messages")
        connection, denials = open_user_sql_connection(self.run.snapshot_path)
        try:
            for sql in (
                "ATTACH DATABASE ':memory:' AS other",
                "PRAGMA query_only = OFF",
                "PRAGMA writable_schema = ON",
                "INSERT INTO visible_messages (id) VALUES ('x')",
                "DELETE FROM visible_messages",
                "CREATE TEMP TABLE t (x)",
                "SELECT * FROM sqlite_temp_master",
                "BEGIN",
            ):
                with self.subTest(sql=sql):
                    with self.assertRaises(sqlite3.DatabaseError):
                        connection.execute(sql)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM visible_messages").fetchone()[0], 30)
        finally:
            connection.close()
        self.assertTrue(denials)

    def test_cte_named_visible_messages_cannot_reveal_a_later_revision_sentinel(self) -> None:
        with self.assertRaises(InvalidInput) as caught:
            self._sql("WITH visible_messages AS (SELECT * FROM messages) SELECT COUNT(*) AS n FROM visible_messages WHERE text LIKE '%SENTINEL%'")
        self.assertEqual(caught.exception.code, "sql.error")
        spoof = self._sql(
            "WITH visible_messages AS (SELECT * FROM main.visible_messages) "
            "SELECT COUNT(*) AS n FROM visible_messages WHERE text LIKE :pattern",
            {"pattern": "%SENTINEL%"},
        )
        self.assertEqual(spoof["rows"], [[0]])
        direct = self._sql("SELECT COUNT(*) AS n, MAX(first_seen_revision) AS newest FROM visible_messages WHERE instr(text, :term) > 0 OR 1", {"term": "哨兵"})
        self.assertEqual(direct["rows"], [[30, 1]])
        snapshot = self.run.snapshot_path.read_bytes()
        self.assertNotIn(b"SENTINEL-LATER-REVISION", snapshot)
        self.assertNotIn("哨兵".encode("utf-8"), snapshot)
        canonical = b"".join(path.read_bytes() for path in self.dataset.root.glob("dataset.sqlite3*"))
        self.assertIn(b"SENTINEL-LATER-REVISION", canonical)
        connection = sqlite3.connect(self.run.snapshot_path)
        try:
            self.assertEqual([row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")], ["visible_messages"])
        finally:
            connection.close()

    def test_resource_exhaustion_is_stopped_explicitly(self) -> None:
        cases = {
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) AS n FROM c": "budget.steps",
            "SELECT COUNT(*) AS n FROM visible_messages a, visible_messages b, visible_messages c, visible_messages d": "budget.steps",
        }
        for sql, code in cases.items():
            with self.subTest(sql=sql):
                with self.assertRaises(BudgetExceeded) as caught:
                    self._sql(sql, limits=SqlLimits(max_steps=200_000))
                self.assertEqual(caught.exception.code, code)
                receipt = load_receipt(self.run, caught.exception.details["receipt_id"])[0]
                self.assertEqual((receipt["status"], receipt["rows"]), ("error", []))
        with self.assertRaises(InvalidInput) as caught:
            self._sql(
                "WITH RECURSIVE s(x) AS (SELECT 'abcdefgh' UNION ALL SELECT x || x FROM s WHERE length(x) < 100000000) "
                "SELECT MAX(length(x)) AS n FROM s"
            )
        self.assertEqual(caught.exception.code, "sql.too_big")
        ticks = itertools.count()
        with mock.patch.object(budget, "clock", lambda: next(ticks) * 0.001):
            with self.assertRaises(BudgetExceeded) as caught:
                self._sql(
                    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) AS n FROM c",
                    limits=SqlLimits(deadline_ms=50, max_steps=10**11),
                )
        self.assertEqual(caught.exception.code, "budget.deadline")

    def test_snapshot_budgets_fail_without_sampling(self) -> None:
        with self.assertRaises(BudgetExceeded) as caught:
            self._sql("SELECT COUNT(*) AS n FROM visible_messages", limits=SqlLimits(snapshot_max_bytes=16 * 1024))
        self.assertEqual(caught.exception.code, "sql.snapshot_size")
        receipt = load_receipt(self.run, caught.exception.details["receipt_id"])[0]
        self.assertEqual((receipt["status"], receipt["rows"], receipt["snapshot"]), ("error", [], None))
        ticks = itertools.count()
        with mock.patch.object(budget, "clock", lambda: next(ticks) * 100.0):
            with self.assertRaises(BudgetExceeded) as caught:
                self._sql("SELECT COUNT(*) AS n FROM visible_messages")
        self.assertEqual(caught.exception.code, "sql.snapshot_time")
        leftovers = [path.name for path in self.run.directory.iterdir() if path.name.startswith(".")]
        self.assertEqual(leftovers, [])

    def test_multirow_results_need_unique_columns_and_top_level_order(self) -> None:
        failures = {
            "SELECT sender_id FROM visible_messages": "sql.unordered_multirow",
            "SELECT * FROM (SELECT id FROM visible_messages ORDER BY id)": "sql.unordered_multirow",
            "SELECT row_number() OVER (ORDER BY id) AS r FROM visible_messages": "sql.unordered_multirow",
            "SELECT sender_id, sender_id FROM visible_messages ORDER BY 1": "sql.duplicate_columns",
        }
        for sql, code in failures.items():
            with self.subTest(sql=sql):
                with self.assertRaises(InvalidInput) as caught:
                    self._sql(sql)
                self.assertEqual(caught.exception.code, code)
        ordered = self._sql("SELECT id FROM visible_messages ORDER BY ts_utc, id", limits=SqlLimits(max_rows=5))
        self.assertEqual((ordered["status"], ordered["complete"], ordered["truncation"], ordered["row_count"]), ("incomplete", False, "max_rows", 5))
        clipped = self._sql("SELECT id, text FROM visible_messages ORDER BY ts_utc, id", limits=SqlLimits(max_bytes=2048))
        self.assertEqual((clipped["complete"], clipped["truncation"]), (False, "max_bytes"))
        self.assertLessEqual(clipped["response_bytes"], 2048)
        whole = self._sql("SELECT id FROM visible_messages ORDER BY ts_utc, id")
        self.assertEqual((whole["complete"], whole["row_count"]), (True, 30))

    def test_parameters_and_scope_are_validated(self) -> None:
        sql = "SELECT COUNT(*) AS n FROM visible_messages WHERE sender_id = :sender"
        for params in ({}, {"sender": "s_ana", "unused": 1}, {"sender": True}, {"sender": [1]}):
            with self.subTest(params=params):
                with self.assertRaises(InvalidInput) as caught:
                    self._sql(sql, params)
                self.assertEqual(caught.exception.code, "sql.parameters")
        for scope in ({}, {**SCOPE, "notes": "x"}, {**SCOPE, "unit": ""}, {**SCOPE, "timezone": "localtime"}):
            with self.subTest(scope=scope):
                with self.assertRaises(InvalidInput):
                    self._sql(sql, {"sender": "s_ana"}, scope=scope)

    def test_local_dates_convert_to_utc_parameters_once(self) -> None:
        page = self._sql(
            "SELECT COUNT(*) AS n FROM visible_messages WHERE ts_utc >= :start AND ts_utc < :end",
            utc_assignments=["start=2026-03-01T08:10", "end=2026-03-01T08:20"],
            tz="Asia/Shanghai",
        )
        self.assertEqual(page["rows"], [[10]])
        receipt = load_receipt(self.run, page["receipt_id"])[0]
        self.assertEqual(receipt["arguments"]["parameters"]["start"], "2026-03-01T00:10:00.000000Z")
        self.assertEqual(receipt["arguments"]["utc_parameters"][0], {
            "name": "start", "local": "2026-03-01T08:10", "zone": "Asia/Shanghai", "utc": "2026-03-01T00:10:00.000000Z", "kind": "datetime",
        })
        with self.assertRaises(InvalidInput) as caught:
            self._sql("SELECT :start AS s", utc_assignments=["start=2026-03-01"])
        self.assertEqual(caught.exception.code, "time.zone_required")

    def test_verifier_rebuilds_its_snapshot_and_ignores_the_run_copy(self) -> None:
        page = self._sql("SELECT COUNT(*) AS n FROM visible_messages WHERE sender_id = :sender", {"sender": "s_ben"})
        self.assertEqual(page["rows"], [[10]])
        draft = {
            "schema_version": 1,
            "dataset_id": self.run.dataset_id,
            "run_id": self.run.run_id,
            "corpus_revision": self.run.corpus_revision,
            "question": self.run.question,
            "claims": [{"id": "c_count", "type": "statistic", "receipt_id": page["receipt_id"], "row": 0, "column": "n", "value": 10}],
        }
        draft_path = self.write_json("draft.json", draft)
        self.run.snapshot_path.write_bytes(b"not a database any more")
        outcome = verify_run(self.dataset, self.run.run_id, draft_source=draft_path)
        self.assertEqual(outcome.status, "pass", outcome.record["failures"])
        self.assertEqual(self.run.snapshot_path.read_bytes(), b"not a database any more")
        self.assertEqual([path.name for path in self.run.directory.iterdir() if path.name.startswith(".")], [])
        receipt = load_receipt(self.run, page["receipt_id"])[0]
        receipt["rows"] = [[11]]
        self.write_receipt_raw(self.run, reseal(receipt))
        tampered = verify_run(self.dataset, self.run.run_id, draft_source=self.write_json("draft-11.json", {**draft, "claims": [{**draft["claims"][0], "value": 11}]}))
        self.assertEqual(tampered.status, "fail")
        self.assertEqual(tampered.record["failures"][0]["code"], "receipt.replay_mismatch")


if __name__ == "__main__":
    unittest.main()
