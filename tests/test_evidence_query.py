from __future__ import annotations

import itertools
import json
import shutil
import threading
import unittest
from unittest import mock

import _path  # noqa: F401

from evidence_support import NAMESPACE, EvidenceCase, message, numbered

from chat_conductor.evidence import budget
from chat_conductor.evidence.errors import BudgetExceeded, InvalidInput
from chat_conductor.evidence.importer import ingest
from chat_conductor.evidence.query import Limits, context, range_query, search
from chat_conductor.evidence.receipts import load_receipt


def _search_args(terms: list[str], **filters: object) -> dict:
    arguments = {
        "terms": terms,
        "case": "ascii-insensitive",
        "source_namespace": None,
        "conversation_id": None,
        "sender_id": None,
        "kinds": None,
        "text_origins": None,
        "from": None,
        "to": None,
        "tz": None,
    }
    arguments.update(filters)
    return arguments


def _range_args(conversation: str = "conv-a", **filters: object) -> dict:
    arguments = _search_args([], source_namespace=NAMESPACE, conversation_id=conversation, **filters)
    del arguments["terms"], arguments["case"]
    return arguments


class QueryTests(EvidenceCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.new_dataset()
        conversation = numbered("a", 60, text="garden note {i}")
        conversation[5] = message("a-0005", "chen", conversation[5]["ts"], kind="image", placeholder="[image]", order=conversation[5]["order"])
        conversation[7] = message("a-0007", "ben", conversation[7]["ts"], kind="voice", transcript="明天浇水", order=conversation[7]["order"])
        extra = [
            ("AI and 信仰 in one line", "ana"),
            ("Market: 50% off_seeds\\discount", "ben"),
            ("5000 off seedsXdiscount", "ben"),
            ("Water first", "ana"),
            ("water later", "ben"),
            ("WATER always", "chen"),
            ("Équipe meeting", "ana"),
            ("équipe meeting", "ben"),
            ("好", "chen"),
        ]
        for offset, (text, sender) in enumerate(extra):
            index = 10 + offset * 3
            conversation[index] = message(f"a-{index:04d}", sender, conversation[index]["ts"], text, order=conversation[index]["order"])
        self.conversation = conversation
        self.ingest(self.dataset, {"conv-a": conversation, "conv-b": numbered("b", 5, day="2026-03-02", text="other {i}")})
        self.run = self.run_for(self.dataset)

    def _search(self, terms: list[str], *, run=None, limits: Limits = Limits(), cursor: str | None = None, **filters: object) -> tuple[dict, dict]:
        result = search(self.dataset, (run or self.run).run_id, _search_args(terms, **filters), limits=limits, cursor=cursor)
        self.assertLessEqual(len(result.output), limits.max_bytes)
        page = json.loads(result.output)
        self.assertEqual(page["response_bytes"], len(result.output))
        return page, result.receipt

    def _texts(self, terms: list[str], **filters: object) -> list[str]:
        page, _ = self._search(terms, **filters)
        self.assertEqual(page["status"], "complete")
        return [row["text"] for row in page["rows"]]

    def test_bilingual_short_and_multiword_literal_terms(self) -> None:
        self.assertEqual(self._texts(["AI"]), ["AI and 信仰 in one line"])
        self.assertEqual(self._texts(["信仰"]), ["AI and 信仰 in one line"])
        self.assertEqual(self._texts(["好"]), ["好"])
        self.assertEqual(self._texts(["浇水"]), ["明天浇水"])
        self.assertEqual(self._texts(["AI", "信仰"]), ["AI and 信仰 in one line"])
        self.assertEqual(self._texts(["AI", "missing"]), [])
        self.assertEqual(self._texts(["first", "water"]), ["Water first"])
        self.assertEqual(self._texts(["water first"]), ["Water first"])
        self.assertEqual(self._texts(["image"]), ["[image]"])
        self.assertEqual(self._texts(["image"], text_origins=["export_text"]), [])
        self.assertEqual(self._texts(["image"], kinds=["image"]), ["[image]"])

    def test_wildcards_and_backslashes_are_literal(self) -> None:
        self.assertEqual(self._texts(["%"]), ["Market: 50% off_seeds\\discount"])
        self.assertEqual(self._texts(["_"]), ["Market: 50% off_seeds\\discount"])
        self.assertEqual(self._texts(["\\"]), ["Market: 50% off_seeds\\discount"])
        self.assertEqual(self._texts(["0% off"]), ["Market: 50% off_seeds\\discount"])
        self.assertEqual(self._texts(["seeds\\discount"]), ["Market: 50% off_seeds\\discount"])
        self.assertEqual(self._texts(["seedsXdiscount"]), ["5000 off seedsXdiscount"])

    def test_case_policy_is_explicit(self) -> None:
        self.assertEqual(len(self._texts(["water"])), 3)
        self.assertEqual(self._texts(["water"], case="sensitive"), ["water later"])
        self.assertEqual(self._texts(["équipe"]), ["équipe meeting"])
        page, receipt = self._search(["water"])
        self.assertEqual(page["scope"]["case"], "ascii-insensitive")
        self.assertIn("ASCII letters", receipt["scope"]["case_policy"])

    def test_pagination_is_deterministic_complete_and_bound(self) -> None:
        full, _ = self._search(["note"])
        self.assertEqual(full["status"], "complete")
        expected = [row["id"] for row in full["rows"]]
        keys = [(row["ts_utc"], row["source_namespace"], row["conversation_id"], row["source_order"], row["id"]) for row in full["rows"]]
        self.assertEqual(keys, sorted(keys))
        collected, cursor, pages = [], None, 0
        while True:
            page, receipt = self._search(["note"], limits=Limits(max_rows=7), cursor=cursor)
            pages += 1
            collected.extend(row["id"] for row in page["rows"])
            self.assertEqual(page["total_matches"], len(expected))
            self.assertEqual([item["id"] for item in receipt["results"]], [row["id"] for row in page["rows"]])
            if not page["has_more"]:
                self.assertEqual((page["status"], page["cursor"], receipt["continuation"]), ("complete", None, None))
                break
            self.assertEqual(page["status"], "incomplete")
            self.assertEqual(page["row_count"], 7)
            cursor = page["cursor"]
        self.assertEqual(collected, expected)
        self.assertEqual(pages, -(-len(expected) // 7))
        first, _ = self._search(["note"], limits=Limits(max_rows=7))
        other_run = self.run_for(self.dataset, "another question")
        for label, call in {
            "other arguments": lambda: self._search(["garden"], cursor=first["cursor"]),
            "other run": lambda: self._search(["note"], run=other_run, cursor=first["cursor"]),
            "other operation": lambda: range_query(self.dataset, self.run.run_id, _range_args(), cursor=first["cursor"]),
        }.items():
            with self.subTest(label):
                with self.assertRaises(InvalidInput) as caught:
                    call()
                self.assertEqual(caught.exception.code, "cursor.binding")
        with self.assertRaises(InvalidInput) as caught:
            self._search(["note"], cursor=first["cursor"][:-1] + ("0" if first["cursor"][-1] != "0" else "1"))
        self.assertEqual(caught.exception.code, "cursor.corrupt")

    def test_byte_limit_marks_pages_incomplete(self) -> None:
        page, receipt = self._search(["note"], limits=Limits(max_bytes=2048))
        self.assertEqual(page["status"], "incomplete")
        self.assertTrue(page["has_more"])
        self.assertLess(page["row_count"], page["total_matches"])
        self.assertEqual(receipt["counts"], {"rows": page["row_count"], "response_bytes": page["response_bytes"]})
        self.assertLessEqual(page["response_bytes"], 2048)
        with self.assertRaises(InvalidInput):
            self._search(["note"], limits=Limits(max_bytes=10))

    def test_oversized_message_is_delivered_in_labelled_segments(self) -> None:
        dataset = self.new_dataset("big", "ds-big")
        huge = "测试🙂 segment " * 1500
        records = [
            message("x-1", "ana", "2026-03-01T09:00:00", "before huge", order="1"),
            message("x-2", "ben", "2026-03-01T09:01:00", huge, order="2"),
            message("x-3", "ana", "2026-03-01T09:02:00", "after huge", order="3"),
        ]
        self.ingest(dataset, {"conv-a": records})
        run = self.run_for(dataset)
        limits = Limits(max_bytes=4096)
        pieces, rows, cursor = [], [], None
        while True:
            result = search(dataset, run.run_id, _search_args(["segment"], conversation_id="conv-a"), limits=limits, cursor=cursor)
            self.assertLessEqual(len(result.output), 4096)
            page = json.loads(result.output)
            for row in page["rows"]:
                rows.append(row)
                if not row["text_complete"]:
                    segment = row["segment"]
                    self.assertEqual(segment["end"] - segment["start"], len(row["text"]))
                    pieces.append(row["text"])
                    if segment["final"]:
                        self.assertEqual(segment["end"], row["text_length"])
            if not page["has_more"]:
                break
            cursor = page["cursor"]
        self.assertEqual("".join(pieces), huge)
        self.assertGreater(len(pieces), 3)
        self.assertEqual({row["id"] for row in rows if row["text_complete"]}, set())
        everything = search(dataset, run.run_id, _search_args(["huge"]), limits=Limits(max_bytes=4096))
        self.assertTrue(json.loads(everything.output)["rows"][0]["text_complete"])

        target = rows[0]["id"]
        cursor, delivered, page_ids = None, {}, []
        while True:
            result = context(dataset, run.run_id, target, limits=limits, cursor=cursor)
            page = json.loads(result.output)
            page_ids.append(result.receipt_id)
            for row in page["rows"]:
                delivered.setdefault(row["id"], []).append(row["text"])
            if page["context_complete"]:
                final = result.receipt
                break
            self.assertEqual(page["status"], "incomplete")
            cursor = page["cursor"]
        self.assertEqual(final["pages"], page_ids)
        self.assertEqual("".join(delivered[target]), huge)
        self.assertEqual([item["id"] for item in final["delivered_all"]], [item["id"] for item in final["window"]["messages"]])
        first_page = load_receipt(run, page_ids[0])[0]
        self.assertFalse(first_page["context_complete"])
        self.assertNotIn("pages", first_page)

    def test_step_and_deadline_budgets_fail_with_error_receipts(self) -> None:
        dataset = self.new_dataset("budget", "ds-budget")
        self.ingest(dataset, {"conv-a": numbered("n", 600)})
        run = self.run_for(dataset)
        arguments = _search_args(["note"])
        with self.assertRaises(BudgetExceeded) as caught:
            search(dataset, run.run_id, arguments, limits=Limits(max_steps=1000))
        self.assertEqual(caught.exception.code, "budget.steps")
        receipt = load_receipt(run, caught.exception.details["receipt_id"])[0]
        self.assertEqual((receipt["status"], receipt["results"], receipt["total_matches"]), ("error", [], None))
        self.assertEqual(receipt["error"]["code"], "budget.steps")
        ticks = itertools.count()
        with mock.patch.object(budget, "clock", lambda: next(ticks) * 10.0):
            with self.assertRaises(BudgetExceeded) as caught:
                search(dataset, run.run_id, arguments)
        self.assertEqual(caught.exception.code, "budget.deadline")
        exit_code, stdout, stderr = self.cli("--dataset", dataset.root, "q", "search", "--run", run.run_id, "note", "--max-steps", "1000")
        self.assertEqual((exit_code, stdout), (3, ""))
        self.assertIn("budget.steps", stderr)
        self.assertNotIn("note", json.loads(stderr)["error"]["message"])

    def test_range_is_half_open_in_an_explicit_zone(self) -> None:
        dataset = self.new_dataset("range", "ds-range")
        stamps = ["2026-03-01T23:59:59", "2026-03-02T00:00:00", "2026-03-02T23:59:59.999999", "2026-03-03T00:00:00"]
        self.ingest(dataset, {"conv-a": [message(f"r{i}", "ana", ts, f"at {ts}", order=f"{i}") for i, ts in enumerate(stamps)]})
        run = self.run_for(dataset)
        result = range_query(dataset, run.run_id, _range_args(**{"from": "2026-03-02", "to": "2026-03-03", "tz": "Asia/Shanghai"}))
        page = json.loads(result.output)
        self.assertEqual([row["ts_original"] for row in page["rows"]], stamps[1:3])
        window = result.receipt["scope"]["time_window"]
        self.assertEqual(window["from"], {"local": "2026-03-02", "utc": "2026-03-01T16:00:00.000000Z", "kind": "date"})
        self.assertEqual(window["to"]["utc"], "2026-03-02T16:00:00.000000Z")
        self.assertIn("excluded", window["date_only_end"])
        unbounded = range_query(dataset, run.run_id, _range_args())
        self.assertEqual(json.loads(unbounded.output)["scope"]["time_window"]["unbounded"], ["from", "to"])
        for arguments, code in (
            (_range_args(**{"from": "2026-03-02"}), "time.zone_required"),
            (_range_args(**{"from": "2026-03-08T02:30", "tz": "America/New_York"}), "time.nonexistent_local"),
            (_range_args(**{"from": "2026-11-01T01:30", "tz": "America/New_York"}), "time.ambiguous_local"),
            (_range_args(**{"from": "2026-03-03", "to": "2026-03-02", "tz": "UTC"}), "time.empty_window"),
            (_range_args(**{"from": "2026-03-02T00:00:00+08:00", "tz": "UTC"}), "time.bound_format"),
            ({**_range_args(), "conversation_id": None}, "range.conversation_required"),
        ):
            with self.subTest(code=code):
                with self.assertRaises(InvalidInput) as caught:
                    range_query(dataset, run.run_id, arguments)
                self.assertEqual(caught.exception.code, code)

    def test_context_has_twenty_neighbours_of_all_kinds_and_edge_flags(self) -> None:
        ids = self._conversation_ids()
        page = json.loads(context(self.dataset, self.run.run_id, ids[30]).output)
        window = page["window"]
        self.assertEqual((window["before_count"], window["after_count"], window["edge_before"], window["edge_after"]), (20, 20, False, False))
        self.assertEqual([row["id"] for row in page["rows"]], ids[10:51])
        self.assertEqual([row["position"] for row in page["rows"]], list(range(-20, 21)))
        self.assertTrue(page["context_complete"])
        near_start = json.loads(context(self.dataset, self.run.run_id, ids[3]).output)["window"]
        self.assertEqual((near_start["before_count"], near_start["after_count"], near_start["edge_before"], near_start["edge_after"]), (3, 20, True, False))
        at_end = json.loads(context(self.dataset, self.run.run_id, ids[-1]).output)["window"]
        self.assertEqual((at_end["before_count"], at_end["after_count"], at_end["edge_before"], at_end["edge_after"]), (20, 0, False, True))
        early = json.loads(context(self.dataset, self.run.run_id, ids[8]).output)
        self.assertEqual({row["kind"] for row in early["rows"]}, {"text", "image", "voice"})
        self.assertEqual({row["conversation_id"] for row in early["rows"]}, {"conv-a"})
        with self.assertRaises(InvalidInput) as caught:
            context(self.dataset, self.run.run_id, "m_" + "0" * 64)
        self.assertEqual(caught.exception.code, "ctx.unknown_message")

    def _conversation_ids(self, run=None) -> list[str]:
        page = json.loads(range_query(self.dataset, (run or self.run).run_id, _range_args(), limits=Limits(max_rows=1000)).output)
        self.assertEqual(page["status"], "complete")
        return [row["id"] for row in page["rows"]]

    def _snapshot_of(self, run) -> tuple:
        ids = self._conversation_ids(run)
        searched = json.loads(search(self.dataset, run.run_id, _search_args(["note"])).output)
        ctx = json.loads(context(self.dataset, run.run_id, ids[30]).output)
        return (
            ids,
            searched["total_matches"],
            [row["id"] for row in searched["rows"]],
            [row["id"] for row in ctx["rows"]],
            [row["origins"] for row in ctx["rows"]],
        )

    def test_pinned_run_is_stable_after_later_imports(self) -> None:
        before = self._snapshot_of(self.run)
        interleaved = [message(f"late-{i}", "ben", f"2026-03-01T08:{29 + i:02d}:30", f"late note {i}", order=f"{29 + i:06d}a") for i in range(4)]
        overlap = self.conversation[25:35]
        self.ingest(self.dataset, {"conv-a": overlap + interleaved}, name="later.json")
        self.assertEqual(self._snapshot_of(self.run), before)
        newer = self.run_for(self.dataset, "newer question")
        after = self._snapshot_of(newer)
        self.assertEqual(len(after[0]), len(before[0]) + 4)
        self.assertEqual(after[1], before[1] + 4)
        self.assertNotEqual(after[3], before[3])
        old_origins = {len(item) for item in before[4]}
        self.assertEqual(old_origins, {1})
        self.assertIn(2, {len(item) for item in after[4]})

    def test_concurrent_imports_do_not_reach_a_pinned_run(self) -> None:
        baseline = self._snapshot_of(self.run)
        errors: list[BaseException] = []
        observed: list[tuple] = []

        def importer_thread(index: int) -> None:
            try:
                batch = [message(f"c{index}-{i}", "chen", f"2026-03-01T08:{30 + i:02d}:15", f"concurrent note {index}-{i}", order=f"{30 + i:06d}c{index}") for i in range(5)]
                export = self.write_json(f"concurrent-{index}.json", {"format": "chat-evidence-synthetic-fixture", "format_version": 1, "synthetic": True, "conversations": [{"conversation_id": "conv-a", "messages": batch}]})
                ingest(self.dataset, export, adapter_name="synthetic-fixture", profile_path=self.profile(f"profile-{index}.json"))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        def reader_thread() -> None:
            try:
                for _ in range(5):
                    observed.append(self._snapshot_of(self.run))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=importer_thread, args=(index,)) for index in range(3)]
        threads.append(threading.Thread(target=reader_thread))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertTrue(observed)
        self.assertTrue(all(item == baseline for item in observed))
        import sqlite3

        connection = sqlite3.connect(self.dataset.db_path)
        revisions = [row[0] for row in connection.execute("SELECT revision FROM import_batches ORDER BY revision")]
        connection.close()
        self.assertEqual(revisions, [1, 2, 3, 4])
        self.assertEqual(self._snapshot_of(self.run), baseline)

    def test_runs_and_datasets_are_isolated(self) -> None:
        other = self.new_dataset("other", "ds-other")
        self.ingest(other, {"conv-a": numbered("o", 3, text="isolated secret {i}")})
        self.assertEqual(self._texts(["isolated"]), [])
        with self.assertRaises(InvalidInput) as caught:
            search(other, self.run.run_id, _search_args(["note"]))
        self.assertEqual(caught.exception.code, "run.unknown")
        shutil.copytree(self.run.directory, other.runs_dir / self.run.run_id)
        with self.assertRaises(InvalidInput) as caught:
            search(other, self.run.run_id, _search_args(["note"]))
        self.assertEqual(caught.exception.code, "run.other_dataset")

    def test_revision_zero_is_the_empty_snapshot(self) -> None:
        empty = self.new_dataset("empty", "ds-empty")
        run = self.run_for(empty)
        self.assertEqual(run.corpus_revision, 0)
        page = json.loads(search(empty, run.run_id, _search_args(["anything"])).output)
        self.assertEqual((page["status"], page["total_matches"], page["rows"]), ("complete", 0, []))
        with self.assertRaises(InvalidInput):
            self.run_for(self.dataset, revision=5)


if __name__ == "__main__":
    unittest.main()
