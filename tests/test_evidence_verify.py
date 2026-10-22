from __future__ import annotations

import copy
import hashlib
import itertools
import json
import shutil
import unittest
from unittest import mock

import _path  # noqa: F401

from evidence_support import FIXTURES, EvidenceCase, reseal

from chat_conductor.evidence import budget
from chat_conductor.evidence.errors import BudgetExceeded, InvalidInput
from chat_conductor.evidence.identity import message_id
from chat_conductor.evidence.importer import ingest
from chat_conductor.evidence.query import Limits, context, search
from chat_conductor.evidence.runs import start_run
from chat_conductor.evidence.sqlquery import SqlLimits, sql_query
from chat_conductor.evidence.verify import check_report, validate_draft_structure, verify_run

QUESTION = "Who keeps the shed key, and how often did Ana mention watering?"
NAMESPACE = "synthetic:demo-account"
QUOTE = "the shed key stays with Ben until 4月10日"


def _search_args(terms: list[str], **filters: object) -> dict:
    arguments = dict.fromkeys(("source_namespace", "conversation_id", "sender_id", "kinds", "text_origins", "from", "to", "tz"))
    arguments.update({"terms": terms, "case": "ascii-insensitive"}, **filters)
    return arguments


class VerifyTests(EvidenceCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.new_dataset(dataset_id="demo-garden")
        profile = FIXTURES / "demo-profile.json"
        ingest(self.dataset, FIXTURES / "demo-export-1.json", adapter_name="synthetic-fixture", profile_path=profile)
        self.run = start_run(self.dataset, QUESTION)
        ingest(self.dataset, FIXTURES / "demo-export-2.json", adapter_name="synthetic-fixture", profile_path=profile)
        found = json.loads(search(self.dataset, self.run.run_id, _search_args(["shed key"])).output)
        self.assertEqual(found["total_matches"], 2)
        self.target = found["rows"][0]
        self.ctx_id = context(self.dataset, self.run.run_id, self.target["id"]).receipt_id
        self.sql_id = self._count_receipt()
        self.absent_id = search(
            self.dataset, self.run.run_id, _search_args(["lawyer"], source_namespace=NAMESPACE, conversation_id="conv-garden")
        ).receipt_id
        self.draft = {
            "schema_version": 1,
            "dataset_id": "demo-garden",
            "run_id": self.run.run_id,
            "corpus_revision": 1,
            "question": QUESTION,
            "claims": [
                self.quote(QUOTE),
                {"id": "c_count_1", "type": "statistic", "receipt_id": self.sql_id, "row": 0, "column": "message_count", "value": 7},
                {"id": "c_absent_1", "type": "not_found", "receipt_ids": [self.absent_id]},
            ],
        }

    def _count_receipt(self, **limits: object) -> str:
        return sql_query(
            self.dataset,
            self.run.run_id,
            (FIXTURES / "demo-count.sql").read_text(encoding="utf-8"),
            json.loads((FIXTURES / "demo-count.params.json").read_text(encoding="utf-8")),
            json.loads((FIXTURES / "demo-count.scope.json").read_text(encoding="utf-8")),
            utc_assignments=["from_utc=2026-03-01", "to_utc=2026-03-04"],
            tz="Asia/Shanghai",
            limits=SqlLimits(**limits),
        ).receipt_id

    def quote(self, fragment: str, *, row: dict | None = None, claim_id: str = "c_quote_1", **overrides: object) -> dict:
        row = row or self.target
        start = row["text"].index(fragment)
        claim = {
            "id": claim_id,
            "type": "quote",
            "message_id": row["id"],
            "sender_id": row["sender_id"],
            "ts_utc": row["ts_utc"],
            "start": start,
            "end": start + len(fragment),
            "text": fragment,
            "context_receipt_id": self.ctx_id,
        }
        claim.update(overrides)
        return claim

    def verify(self, draft: dict | bytes):
        data = draft if isinstance(draft, bytes) else json.dumps(draft, ensure_ascii=False).encode("utf-8")
        path = self.tmp / "draft-under-test.json"
        path.write_bytes(data)
        return verify_run(self.dataset, self.run.run_id, draft_source=path)

    def with_claim(self, index: int, **changes: object) -> dict:
        draft = copy.deepcopy(self.draft)
        draft["claims"][index].update(changes)
        return draft

    def assert_fails(self, draft: dict | bytes, code: str) -> dict:
        outcome = self.verify(draft)
        self.assertEqual(outcome.status, "fail", f"expected {code}")
        self.assertEqual(outcome.exit_code, 2)
        codes = [failure["code"] for failure in outcome.record["failures"]]
        self.assertIn(code, codes)
        self.assertFalse(self.run.report_path.exists())
        return outcome.record

    def tamper(self, receipt_id: str, change, *, digest: bool = True) -> None:
        receipt = self.read_receipt(self.run, receipt_id)
        change(receipt)
        self.write_receipt_raw(self.run, reseal(receipt) if digest else receipt)

    # --- success -------------------------------------------------------------------

    def test_three_claim_report_verifies_and_renders(self) -> None:
        outcome = self.verify(self.draft)
        self.assertEqual((outcome.status, outcome.exit_code), ("pass", 0), outcome.record["failures"])
        report = self.run.report_path.read_text(encoding="utf-8")
        for expected in (
            "unverified input",
            "Corpus revision: 1 (only imports up to and including revision 1 are in scope)",
            "### c_quote_1 · Quote",
            f"Span: Unicode code points [11, 50) of {len(self.target['text'])}",
            "Time: `2026-03-02T00:10:00.000000Z` UTC; original `2026-03-02T08:10:00` in declared zone `Asia/Shanghai`",
            "delivered 20 earlier and 20 later messages of all kinds (radius 20)",
            "Corpus edges: earlier side not reached, later side not reached",
            "Text origin: exported message text",
            "revision 1 `demo-export-1.json#/conversations/0/messages/22`",
            "Value: `7` (integer) at row 0, column `message_count`",
            "local `2026-03-04` in `Asia/Shanghai` = `2026-03-03T16:00:00.000000Z` UTC",
            "Missing-data limits:",
            "not counts of real-world events",
            "No matching records were found for these queries and filters at corpus revision 1.",
            'terms `["lawyer"]`',
            "## Limitations",
        ):
            self.assertIn(expected, report)
        self.assertNotIn("still with me", report)
        record = json.loads(self.run.verification_path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "pass")
        self.assertEqual(record["draft_sha256"], hashlib.sha256(self.run.analysis_path.read_bytes()).hexdigest())
        self.assertEqual(record["report_sha256"], hashlib.sha256(self.run.report_path.read_bytes()).hexdigest())
        for receipt_id in (self.ctx_id, self.sql_id, self.absent_id):
            self.assertEqual(record["receipts"][receipt_id], hashlib.sha256((self.run.receipts_dir / f"{receipt_id}.json").read_bytes()).hexdigest())
        self.assertEqual(check_report(self.dataset, self.run.run_id)["status"], "verified")
        self.assertEqual(sorted(path.name for path in self.run.directory.iterdir()), [
            "analysis.json", "receipts", "report.md", "run.json", "sql-snapshot.sqlite3", "verification.json",
        ])

    def test_transcripts_are_labelled_and_paged_context_counts(self) -> None:
        voice = json.loads(search(self.dataset, self.run.run_id, _search_args(["八点"])).output)["rows"][0]
        self.assertEqual(voice["text_origin"], "export_transcript")
        cursor, receipt_id = None, None
        while True:
            result = context(self.dataset, self.run.run_id, voice["id"], limits=Limits(max_bytes=4096), cursor=cursor)
            receipt_id = result.receipt_id
            page = json.loads(result.output)
            if page["context_complete"]:
                break
            cursor = page["cursor"]
        self.assertGreater(page["page"], 1)
        draft = copy.deepcopy(self.draft)
        draft["claims"].append(self.quote("八点浇水", row=voice, claim_id="c_voice", context_receipt_id=receipt_id))
        outcome = self.verify(draft)
        self.assertEqual(outcome.status, "pass", outcome.record["failures"])
        report = self.run.report_path.read_text(encoding="utf-8")
        self.assertIn("exported transcript (labelled; not transcribed or checked by this tool)", report)
        self.assertIn("Corpus edges: earlier side reached", report)

    # --- quotes --------------------------------------------------------------------

    def test_quote_rejections(self) -> None:
        text = self.target["text"]
        tail = "Let's not"
        byte_start = len(text[: text.index(tail)].encode("utf-8"))
        emoji_start = text.index("again 🙂")
        image = json.loads(search(self.dataset, self.run.run_id, _search_args(["[image]"])).output)["rows"][0]
        later_id = message_id("demo-garden", NAMESPACE, "conv-garden", "g-0048")
        cases = {
            "wrong sender": (dict(sender_id="s_ben"), "quote.sender_mismatch"),
            "wrong date": (dict(ts_utc="2026-03-03T00:10:00.000000Z"), "quote.timestamp_mismatch"),
            "non-canonical time": (dict(ts_utc="2026-03-02T08:10:00+08:00"), "quote.timestamp_format"),
            "fabricated": (dict(text="the shed key stays with Bob until 4月10日"), "quote.text_mismatch"),
            "extra space": (dict(text="the shed key  stays with Ben until 4月10日", end=51), "quote.text_mismatch"),
            "whitespace cleanup": (self._span("4月10日.\r\n  Let's", text="4月10日. Let's"), "quote.text_mismatch"),
            "utf-16 offsets": (dict(start=emoji_start, end=emoji_start + 8, text="again 🙂"), "quote.span_out_of_range"),
            "utf-8 byte offsets": (dict(start=byte_start, end=byte_start + len(tail), text=tail), "quote.offsets_mismatch"),
            "empty span": (dict(start=20, end=20), "quote.span_out_of_range"),
            "placeholder": (dict(message_id=image["id"], sender_id=image["sender_id"], ts_utc=image["ts_utc"], start=0, end=7, text="[image]"), "quote.placeholder"),
            "unknown message": (dict(message_id="m_" + "0" * 64), "quote.unknown_message"),
            "later revision": (dict(message_id=later_id), "quote.unknown_message"),
        }
        for label, (changes, code) in cases.items():
            with self.subTest(label):
                self.assert_fails(self.with_claim(0, **changes), code)
        self.assertEqual(self.verify(self.with_claim(0, **self._span("again 🙂"))).status, "pass")

    def _span(self, actual: str, *, text: str | None = None) -> dict:
        start = self.target["text"].index(actual)
        return {"start": start, "end": start + len(actual), "text": actual if text is None else text}

    def test_context_rejections(self) -> None:
        neighbour = json.loads(context(self.dataset, self.run.run_id, self.target["id"]).output)["rows"][21]
        cases = {
            "radius too small": (context(self.dataset, self.run.run_id, self.target["id"], radius=5).receipt_id, "context.radius_too_small"),
            "first of several pages": (context(self.dataset, self.run.run_id, self.target["id"], limits=Limits(max_bytes=2048)).receipt_id, "context.incomplete"),
            "other target": (context(self.dataset, self.run.run_id, neighbour["id"]).receipt_id, "context.wrong_target"),
            "not a context receipt": (self.sql_id, "context.wrong_operation"),
            "missing": ("q_doesnotexist", "receipt.missing"),
        }
        for label, (receipt_id, code) in cases.items():
            with self.subTest(label):
                self.assert_fails(self.with_claim(0, context_receipt_id=receipt_id), code)

    def test_altered_stale_and_escaping_receipts_are_rejected(self) -> None:
        pristine = (self.run.receipts_dir / f"{self.ctx_id}.json").read_bytes()

        def restore() -> None:
            (self.run.receipts_dir / f"{self.ctx_id}.json").write_bytes(pristine)

        def flip_edge(receipt: dict) -> None:
            receipt["window"]["edge_before"] = True

        def other_revision(receipt: dict) -> None:
            receipt["corpus_revision"] = 2

        def drop_a_neighbour(receipt: dict) -> None:
            receipt["delivered_all"] = receipt["delivered_all"][1:]
            receipt["results"] = receipt["results"][1:]

        for label, change, digest, code in (
            ("edited without digest", flip_edge, False, "receipt.digest_mismatch"),
            ("edited with new digest", flip_edge, True, "receipt.replay_mismatch"),
            ("wrong revision", other_revision, True, "receipt.stale_revision"),
            ("clipped window", drop_a_neighbour, True, "context.incomplete"),
        ):
            with self.subTest(label):
                self.tamper(self.ctx_id, change, digest=digest)
                self.assert_fails(self.draft, code)
                restore()
        other_run = start_run(self.dataset, QUESTION)
        foreign = context(self.dataset, other_run.run_id, self.target["id"]).receipt_id
        shutil.copy(other_run.receipts_dir / f"{foreign}.json", self.run.receipts_dir / f"{foreign}.json")
        self.assert_fails(self.with_claim(0, context_receipt_id=foreign), "receipt.other_run")
        self.assertEqual(validate_draft_structure(self.with_claim(0, context_receipt_id="q_../../run.json"))[0].code, "claim.field_invalid")
        outside = self.tmp / "outside-receipts"
        shutil.copytree(self.run.receipts_dir, outside)
        shutil.rmtree(self.run.receipts_dir)
        try:
            self.run.receipts_dir.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links are unavailable")
        self.assert_fails(self.draft, "path.escape")

    # --- statistics ----------------------------------------------------------------

    def test_statistic_rejections(self) -> None:
        for label, changes, code in (
            ("wrong value", dict(value=8), "statistic.value_mismatch"),
            ("real instead of integer", dict(value=7.0), "statistic.value_mismatch"),
            ("text instead of integer", dict(value="7"), "statistic.value_mismatch"),
            ("boolean instead of integer", dict(value=True), "statistic.value_mismatch"),
            ("unknown column", dict(column="count"), "statistic.unknown_column"),
            ("row out of range", dict(row=1), "statistic.row_out_of_range"),
            ("search receipt", dict(receipt_id=self.absent_id), "statistic.wrong_operation"),
        ):
            with self.subTest(label):
                self.assert_fails(self.with_claim(1, **changes), code)
        grouped = sql_query(
            self.dataset,
            self.run.run_id,
            "SELECT sender_id, COUNT(*) AS n FROM visible_messages GROUP BY sender_id ORDER BY sender_id",
            {},
            json.loads((FIXTURES / "demo-count.scope.json").read_text(encoding="utf-8")),
            limits=SqlLimits(max_rows=1),
        )
        self.assertFalse(grouped.receipt["complete"])
        self.assert_fails(self.with_claim(1, receipt_id=grouped.receipt_id, column="n", value=grouped.receipt["rows"][0][1]), "statistic.incomplete_receipt")
        with self.assertRaises(InvalidInput) as caught:
            sql_query(self.dataset, self.run.run_id, "SELECT COUNT(*) AS message_count FROM sqlite_master", {}, json.loads((FIXTURES / "demo-count.scope.json").read_text(encoding="utf-8")))
        self.assert_fails(self.with_claim(1, receipt_id=caught.exception.details["receipt_id"]), "statistic.incomplete_receipt")

        def inflate(receipt: dict) -> None:
            receipt["rows"] = [[8]]

        def rewrite_sql(receipt: dict) -> None:
            receipt["arguments"]["sql"] = receipt["arguments"]["sql"].replace("s", "s", 1) + " "

        for label, change in (("cells", inflate), ("sql text", rewrite_sql)):
            with self.subTest(label):
                original = (self.run.receipts_dir / f"{self.sql_id}.json").read_bytes()
                self.tamper(self.sql_id, change)
                self.assert_fails(self.with_claim(1, value=8) if label == "cells" else self.draft, "receipt.replay_mismatch")
                (self.run.receipts_dir / f"{self.sql_id}.json").write_bytes(original)

    # --- absence -------------------------------------------------------------------

    def test_not_found_rejections(self) -> None:
        with_matches = search(self.dataset, self.run.run_id, _search_args(["shed key"])).receipt_id
        first_page = search(self.dataset, self.run.run_id, _search_args(["water"]), limits=Limits(max_rows=1))
        last_page = first_page
        while last_page.receipt["has_more"]:
            last_page = search(self.dataset, self.run.run_id, _search_args(["water"]), limits=Limits(max_rows=1), cursor=last_page.receipt["continuation"])
        ticks = itertools.count()
        with mock.patch.object(budget, "PROGRESS_INTERVAL", 10), mock.patch.object(budget, "clock", lambda: next(ticks) * 10.0):
            with self.assertRaises(BudgetExceeded) as caught:
                search(self.dataset, self.run.run_id, _search_args(["zzz"]))
        timed_out = caught.exception.details["receipt_id"]
        for label, receipt_ids, code in (
            ("matches", [with_matches], "not_found.has_matches"),
            ("continuation pending", [first_page.receipt_id], "not_found.incomplete_search"),
            ("later page", [last_page.receipt_id], "not_found.incomplete_search"),
            ("budget error", [timed_out], "not_found.incomplete_search"),
            ("context receipt", [self.ctx_id], "not_found.wrong_operation"),
            ("one bad among good", [self.absent_id, with_matches], "not_found.has_matches"),
        ):
            with self.subTest(label):
                self.assert_fails(self.with_claim(2, receipt_ids=receipt_ids), code)

        def hide_matches(receipt: dict) -> None:
            receipt.update(total_matches=0, results=[], status="complete", has_more=False, continuation=None)

        self.tamper(with_matches, hide_matches)
        self.assert_fails(self.with_claim(2, receipt_ids=[with_matches]), "receipt.replay_mismatch")

    # --- structure and identity ----------------------------------------------------

    def test_structure_and_identity_rejections(self) -> None:
        base = json.dumps(self.draft, ensure_ascii=False)
        raw_cases = {
            "duplicate key": (base.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1).encode(), "draft.duplicate_key"),
            "NaN": (base.replace('"value": 7', '"value": NaN').encode(), "draft.nonfinite_number"),
            "overflowing number": (base.replace('"value": 7', '"value": 1e999').encode(), "draft.nonfinite_number"),
            "byte-order mark": (b"\xef\xbb\xbf" + base.encode(), "draft.bom"),
            "invalid utf-8": (base.encode().replace(b"Who", b"\xffho"), "draft.encoding"),
            "not json": (b"{", "draft.invalid_json"),
        }
        for label, (data, code) in raw_cases.items():
            with self.subTest(label):
                self.assert_fails(data, code)
        duplicate = copy.deepcopy(self.draft)
        duplicate["claims"][1]["id"] = "c_quote_1"
        other = copy.deepcopy(self.draft)
        other["claims"].append({"id": "c_x", "type": "summary", "text": "an interpretation"})
        extra_field = self.with_claim(0, note="free-form interpretation")
        missing = copy.deepcopy(self.draft)
        del missing["claims"][0]["context_receipt_id"]
        cases = {
            "duplicate claim ids": (duplicate, "claim.duplicate_id"),
            "unknown claim type": (other, "claim.unknown_type"),
            "free-form field": (extra_field, "claim.unknown_field"),
            "missing field": (missing, "claim.missing_field"),
            "boolean offset": (self.with_claim(0, start=True), "claim.field_invalid"),
            "real offset": (self.with_claim(0, start=11.0), "claim.field_invalid"),
            "bad claim id": (self.with_claim(0, id="quote-1"), "claim.invalid_id"),
            "uppercase message id": (self.with_claim(0, message_id=self.target["id"].upper().replace("M_", "m_")), "claim.field_invalid"),
            "duplicate receipt ids": (self.with_claim(2, receipt_ids=[self.absent_id, self.absent_id]), "claim.field_invalid"),
            "empty claims": ({**self.draft, "claims": []}, "draft.claims"),
            "unknown top-level field": ({**self.draft, "summary": "x"}, "draft.unknown_field"),
            "schema version": ({**self.draft, "schema_version": 2}, "draft.schema_version"),
            "other dataset": ({**self.draft, "dataset_id": "elsewhere"}, "draft.dataset_id_mismatch"),
            "other run": ({**self.draft, "run_id": "run_other"}, "draft.run_id_mismatch"),
            "other revision": ({**self.draft, "corpus_revision": 2}, "draft.corpus_revision_mismatch"),
            "edited question": ({**self.draft, "question": QUESTION + "?"}, "draft.question_mismatch"),
        }
        for label, (draft, code) in cases.items():
            with self.subTest(label):
                self.assert_fails(draft, code)
        fresh = start_run(self.dataset, QUESTION)
        outcome = verify_run(self.dataset, fresh.run_id)
        self.assertEqual(outcome.record["failures"][0]["code"], "draft.missing")

    # --- publication gate ----------------------------------------------------------

    def test_edited_or_stale_output_is_not_verified(self) -> None:
        self.assertEqual(self.verify(self.draft).status, "pass")
        rendered = self.run.report_path.read_bytes()
        self.run.report_path.write_bytes(rendered + b"\nA hand-written conclusion.\n")
        check = check_report(self.dataset, self.run.run_id)
        self.assertEqual((check["status"], check["problems"]), ("not_verified", ["report_sha256.mismatch"]))
        exit_code, _, _ = self.cli("--dataset", self.dataset.root, "verify", "--run", self.run.run_id, "--check")
        self.assertEqual(exit_code, 2)
        self.assertEqual(verify_run(self.dataset, self.run.run_id).status, "pass")
        self.assertEqual(self.run.report_path.read_bytes(), rendered)
        self.run.analysis_path.write_bytes(self.run.analysis_path.read_bytes() + b" ")
        self.assertIn("draft_sha256.mismatch", check_report(self.dataset, self.run.run_id)["problems"])
        self.assertEqual(self.verify(self.draft).status, "pass")
        self.verify(self.with_claim(0, text="fabricated"))
        self.assertFalse(self.run.report_path.exists(), "a failed re-verification must not leave the old report looking verified")
        self.assertEqual(check_report(self.dataset, self.run.run_id)["status"], "not_verified")
        self.assertEqual(self.verify(self.draft).status, "pass")
        self.run.report_path.write_bytes(b"operator notes, not tool output")
        self.verify(self.with_claim(0, text="fabricated"))
        self.assertEqual(self.run.report_path.read_bytes(), b"operator notes, not tool output")

    def test_cli_exit_codes(self) -> None:
        draft_path = self.write_json("cli-draft.json", self.draft)
        exit_code, stdout, _ = self.cli("--dataset", self.dataset.root, "verify", "--run", self.run.run_id, "--draft", draft_path)
        self.assertEqual((exit_code, json.loads(stdout)["status"]), (0, "pass"))
        exit_code, stdout, _ = self.cli("--dataset", self.dataset.root, "verify", "--run", self.run.run_id, "--check")
        self.assertEqual((exit_code, json.loads(stdout)["status"]), (0, "verified"))
        bad_path = self.write_json("cli-bad.json", self.with_claim(1, value=6))
        exit_code, stdout, _ = self.cli("--dataset", self.dataset.root, "verify", "--run", self.run.run_id, "--draft", bad_path)
        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(stdout)["failures"], [{"claim_id": "c_count_1", "code": "statistic.value_mismatch", "field": "value"}])
        exit_code, _, stderr = self.cli("--dataset", self.tmp / "no-such-dataset", "verify", "--run", self.run.run_id)
        self.assertEqual(exit_code, 3)
        self.assertIn("dataset.not_initialized", stderr)

    def test_packaged_draft_schema_is_the_approved_contract(self) -> None:
        from importlib import resources

        data = resources.files("chat_conductor.evidence").joinpath("analysis.schema.json").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), "41cc0d5dfda6ba9dcbb7b4ff1a43a732a04fe903f319eb38f10f650d669dc8f6")
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed; the packaged schema hash still pins the contract")
        validator = jsonschema.Draft202012Validator(json.loads(data))
        self.assertEqual(list(validator.iter_errors(self.draft)), [])
        self.assertEqual(validate_draft_structure(self.draft), [])
        for invalid in (
            self.with_claim(0, note="x"),
            {**self.draft, "claims": []},
            self.with_claim(2, receipt_ids=[]),
            self.with_claim(0, message_id="m_123"),
            {**self.draft, "extra": 1},
        ):
            self.assertTrue(list(validator.iter_errors(invalid)))
            self.assertTrue(validate_draft_structure(invalid))


if __name__ == "__main__":
    unittest.main()
