"""Strict verification of a structured report draft, then deterministic rendering.

Pipeline: strict JSON (duplicate keys and non-finite numbers rejected), a closed
structure, run/dataset/revision/question identity, then per-claim checks in
which every referenced receipt is replayed against the canonical pinned
evidence. SQL receipts are replayed over a snapshot the verifier builds itself
from canonical evidence; the run's own ``sql-snapshot.sqlite3`` is never read.
Only after every check passes is Markdown rendered (atomically) and
``verification.json`` written with the draft, report and receipt hashes.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import TOOL_NAME, TOOL_VERSION
from .budget import Budget
from .errors import EXIT_INVALID, EXIT_OK, BudgetExceeded, ConfigError, InvalidInput
from .fsutil import replace_file
from .identity import payload_of_row
from .jsonutil import canonical_bytes, loads_strict, sha256_bytes, sha256_json, sha256_text
from .paths import CLAIM_ID_RE, MESSAGE_ID_RE, RECEIPT_ID_RE, RUN_ID_RE
from .query import compute_window, count_matches, merge_deliveries, search_where
from .receipts import load_receipt
from .render import LIMITATIONS, render_report
from .runs import Run, load_run
from .sqlsandbox import (
    SqlOutcome,
    build_snapshot,
    execute_user_sql,
    inspect_sql,
    remove_snapshot,
    temporary_snapshot_path,
    utc_parameters,
    validate_parameters,
    validate_scope,
)
from .store import MESSAGE_SELECT, Dataset, message_dict
from .timeutil import is_canonical_utc, utc_now

VERIFICATION_FORMAT = "chat-evidence-verification"
VERIFICATION_VERSION = 1
DRAFT_SCHEMA_VERSION = 1
REQUIRED_CONTEXT_RADIUS = 20
REPLAY_MIN_DEADLINE_MS = 30_000
_TOP_KEYS = frozenset({"schema_version", "dataset_id", "run_id", "corpus_revision", "question", "claims"})
_CLAIM_KEYS = {
    "quote": frozenset({"id", "type", "message_id", "sender_id", "ts_utc", "start", "end", "text", "context_receipt_id"}),
    "statistic": frozenset({"id", "type", "receipt_id", "row", "column", "value"}),
    "not_found": frozenset({"id", "type", "receipt_ids"}),
}
_CHECKS = {
    "quote": [
        "message_in_pinned_corpus", "payload_hash", "sender", "ts_utc", "not_placeholder",
        "codepoint_span", "exact_substring", "context_receipt_final", "context_window_replayed",
        "context_segments_replayed",
    ],
    "statistic": ["sql_receipt_complete", "sql_replayed_on_fresh_snapshot", "cell_type_and_value"],
    "not_found": ["search_receipts_complete_first_page", "zero_matches", "search_replayed"],
}


class Failure(Exception):
    """One failed check; reported without echoing source text."""

    def __init__(self, code: str, message: str, *, claim_id: str | None = None, field: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.claim_id = claim_id
        self.field = field

    def to_dict(self) -> dict:
        return {"code": self.code, "claim_id": self.claim_id, "field": self.field, "message": self.message}


@dataclass(frozen=True)
class VerificationOutcome:
    status: str
    exit_code: int
    record: dict
    report_path: Path | None


def verify_run(dataset: Dataset, run_id: str, *, draft_source: Path | None = None) -> VerificationOutcome:
    connection = dataset.connect_ro()
    try:
        run = load_run(dataset, connection, run_id)
        if draft_source is not None:
            try:
                data = Path(draft_source).read_bytes()
            except OSError as exc:
                raise ConfigError("draft.unreadable", "cannot read the draft file") from exc
            replace_file(run.analysis_path, data)
        return _Verifier(dataset, connection, run).execute()
    finally:
        connection.close()


def check_report(dataset: Dataset, run_id: str) -> dict:
    """Is the current report.md still the verified one? Compare every recorded hash."""
    connection = dataset.connect_ro()
    try:
        run = load_run(dataset, connection, run_id)
    finally:
        connection.close()
    problems: list[str] = []
    record = _read_record(run)
    if record is None:
        problems.append("verification.missing")
        record = {}
    elif record.get("status") != "pass":
        problems.append("verification.not_passed")
    for path, key in ((run.report_path, "report_sha256"), (run.analysis_path, "draft_sha256"), (run.run_path, "run_sha256")):
        if _file_sha256(path) is None or _file_sha256(path) != record.get(key):
            problems.append(f"{key}.mismatch")
    receipts = record.get("receipts")
    for receipt_id, digest in (receipts.items() if isinstance(receipts, dict) else ()):
        if not isinstance(receipt_id, str) or not RECEIPT_ID_RE.fullmatch(receipt_id):
            problems.append("receipts.invalid")
        elif _file_sha256(run.receipts_dir / f"{receipt_id}.json") != digest:
            problems.append(f"receipt.{receipt_id}.mismatch")
    return {
        "status": "verified" if not problems else "not_verified",
        "run_id": run.run_id,
        "problems": problems,
        "report": str(run.report_path),
    }


def validate_draft_structure(draft: object) -> list[Failure]:
    """Closed-structure checks equivalent to analysis.schema.json, with strict types."""
    failures: list[Failure] = []

    def fail(code: str, field: str, message: str, claim_id: str | None = None) -> None:
        failures.append(Failure(code, message, claim_id=claim_id, field=field))

    if not isinstance(draft, dict):
        return [Failure("draft.not_object", "the draft must be a JSON object")]
    for key in sorted(draft.keys() - _TOP_KEYS):
        fail("draft.unknown_field", key, "unknown top-level field")
    for key in sorted(_TOP_KEYS - draft.keys()):
        fail("draft.missing_field", key, "missing top-level field")
    if failures:
        return failures
    if not _is_int(draft["schema_version"]) or draft["schema_version"] != DRAFT_SCHEMA_VERSION:
        fail("draft.schema_version", "schema_version", "schema_version must be the integer 1")
    if not _nonempty(draft["dataset_id"]):
        fail("draft.dataset_id", "dataset_id", "dataset_id must be nonempty text")
    if not isinstance(draft["run_id"], str) or not RUN_ID_RE.fullmatch(draft["run_id"]):
        fail("draft.run_id", "run_id", "run_id must match run_[A-Za-z0-9_-]+")
    if not _is_int(draft["corpus_revision"]) or draft["corpus_revision"] < 0:
        fail("draft.corpus_revision", "corpus_revision", "corpus_revision must be a nonnegative integer")
    if not _nonempty(draft["question"]):
        fail("draft.question", "question", "question must be nonempty text")
    claims = draft["claims"]
    if not isinstance(claims, list) or not claims:
        fail("draft.claims", "claims", "claims must be a nonempty array")
        return failures
    seen: set[str] = set()
    for index, claim in enumerate(claims):
        where = f"claims[{index}]"
        if not isinstance(claim, dict):
            fail("claim.not_object", where, "each claim must be an object")
            continue
        claim_id = claim.get("id") if isinstance(claim.get("id"), str) else None
        kind = claim.get("type")
        if not isinstance(kind, str) or kind not in _CLAIM_KEYS:
            fail("claim.unknown_type", f"{where}.type", "claim type must be quote, statistic or not_found", claim_id)
            continue
        keys = _CLAIM_KEYS[kind]
        for key in sorted(claim.keys() - keys):
            fail("claim.unknown_field", f"{where}.{key}", "unknown claim field", claim_id)
        for key in sorted(keys - claim.keys()):
            fail("claim.missing_field", f"{where}.{key}", "missing claim field", claim_id)
        if claim.keys() != keys:
            continue
        if not isinstance(claim["id"], str) or not CLAIM_ID_RE.fullmatch(claim["id"]):
            fail("claim.invalid_id", f"{where}.id", "claim ids must match c_[A-Za-z0-9_-]+")
        elif claim["id"] in seen:
            fail("claim.duplicate_id", f"{where}.id", "claim ids must be unique", claim_id)
        else:
            seen.add(claim["id"])
        for field, message in _claim_field_problems(kind, claim):
            fail("claim.field_invalid", f"{where}.{field}", message, claim_id)
    return failures


def _claim_field_problems(kind: str, claim: dict) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    if kind == "quote":
        if not isinstance(claim["message_id"], str) or not MESSAGE_ID_RE.fullmatch(claim["message_id"]):
            problems.append(("message_id", "message_id must match m_[0-9a-f]{64}"))
        for name in ("sender_id", "ts_utc", "text"):
            if not _nonempty(claim[name]):
                problems.append((name, f"{name} must be nonempty text"))
        if not _is_int(claim["start"]) or claim["start"] < 0:
            problems.append(("start", "start must be a nonnegative integer"))
        if not _is_int(claim["end"]) or claim["end"] < 1:
            problems.append(("end", "end must be a positive integer"))
        if not _receipt_id(claim["context_receipt_id"]):
            problems.append(("context_receipt_id", "receipt ids must match q_[A-Za-z0-9_-]+"))
    elif kind == "statistic":
        if not _receipt_id(claim["receipt_id"]):
            problems.append(("receipt_id", "receipt ids must match q_[A-Za-z0-9_-]+"))
        if not _is_int(claim["row"]) or claim["row"] < 0:
            problems.append(("row", "row must be a nonnegative integer"))
        if not _nonempty(claim["column"]):
            problems.append(("column", "column must be nonempty text"))
        if not (claim["value"] is None or isinstance(claim["value"], (str, int, float, bool))):
            problems.append(("value", "value must be a JSON scalar"))
    else:
        receipt_ids = claim["receipt_ids"]
        if (
            not isinstance(receipt_ids, list)
            or not receipt_ids
            or not all(_receipt_id(item) for item in receipt_ids)
            or len(set(receipt_ids)) != len(receipt_ids)
        ):
            problems.append(("receipt_ids", "receipt_ids must be a nonempty array of unique receipt ids"))
    return problems


def strict_equal(left: object, right: object) -> bool:
    """Equality that distinguishes JSON types: 1, 1.0 and true all differ."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return struct.pack("<d", left) == struct.pack("<d", right)
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(strict_equal(left[key], right[key]) for key in left)
    return left == right


class _Verifier:
    def __init__(self, dataset: Dataset, connection: sqlite3.Connection, run: Run) -> None:
        self.dataset = dataset
        self.connection = connection
        self.run = run
        self.receipt_hashes: dict[str, str] = {}
        self._receipts: dict[str, dict] = {}
        self._sql_replays: dict[str, SqlOutcome] = {}
        self._snapshot: Path | None = None

    def execute(self) -> VerificationOutcome:
        previous = _read_record(self.run)
        draft_sha256: str | None = None
        try:
            try:
                data = self.run.analysis_path.read_bytes()
            except FileNotFoundError:
                return self._fail([Failure("draft.missing", "the run has no analysis.json draft")], None, previous)
            draft_sha256 = sha256_bytes(data)
            try:
                draft = loads_strict(data, what="draft")
            except InvalidInput as exc:
                return self._fail([Failure(exc.code, exc.message)], draft_sha256, previous)
            failures = validate_draft_structure(draft) or self._identity(draft)
            if failures:
                return self._fail(failures, draft_sha256, previous)
            verified: list[dict] = []
            for claim in draft["claims"]:
                try:
                    verified.append(self._claim(claim))
                except Failure as failure:
                    failures.append(failure)
            if failures:
                return self._fail(failures, draft_sha256, previous)
            report = render_report(self.run, draft, verified).encode("utf-8")
            replace_file(self.run.report_path, report)
            checks = [{"claim_id": claim["id"], "type": claim["type"], "checks": _CHECKS[claim["type"]]} for claim in verified]
            record = self._record("pass", draft_sha256, report_sha256=sha256_bytes(report), checks=checks)
            replace_file(self.run.verification_path, canonical_bytes(record) + b"\n")
            return VerificationOutcome("pass", EXIT_OK, record, self.run.report_path)
        except (ConfigError, sqlite3.Error, OSError) as exc:
            _remove_stale_report(self.run, previous)
            code = exc.code if isinstance(exc, ConfigError) else f"runtime.{type(exc).__name__}"
            record = self._record("error", draft_sha256, error={"code": code})
            replace_file(self.run.verification_path, canonical_bytes(record) + b"\n")
            raise
        finally:
            if self._snapshot is not None:
                remove_snapshot(self._snapshot)

    def _fail(self, failures: list[Failure], draft_sha256: str | None, previous: dict | None) -> VerificationOutcome:
        _remove_stale_report(self.run, previous)
        record = self._record("fail", draft_sha256, failures=failures)
        replace_file(self.run.verification_path, canonical_bytes(record) + b"\n")
        return VerificationOutcome("fail", EXIT_INVALID, record, None)

    def _record(
        self,
        status: str,
        draft_sha256: str | None,
        *,
        report_sha256: str | None = None,
        failures: Sequence[Failure] = (),
        checks: Sequence[dict] = (),
        error: dict | None = None,
    ) -> dict:
        return {
            "verification_format": VERIFICATION_FORMAT,
            "verification_version": VERIFICATION_VERSION,
            "status": status,
            "run_id": self.run.run_id,
            "dataset_id": self.run.dataset_id,
            "corpus_revision": self.run.corpus_revision,
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "verified_at": utc_now(),
            "run_sha256": _file_sha256(self.run.run_path),
            "draft_sha256": draft_sha256,
            "report_sha256": report_sha256,
            "receipts": dict(sorted(self.receipt_hashes.items())),
            "checks": list(checks),
            "failures": [failure.to_dict() for failure in failures],
            "error": error,
            "limitations": list(LIMITATIONS),
        }

    def _identity(self, draft: dict) -> list[Failure]:
        expected = {
            "dataset_id": self.run.dataset_id,
            "run_id": self.run.run_id,
            "corpus_revision": self.run.corpus_revision,
            "question": self.run.question,
        }
        return [
            Failure(f"draft.{name}_mismatch", f"{name} differs from the run being verified", field=name)
            for name, value in expected.items()
            if not strict_equal(draft[name], value)
        ]

    def _claim(self, claim: dict) -> dict:
        try:
            if claim["type"] == "quote":
                return self._quote(claim)
            if claim["type"] == "statistic":
                return self._statistic(claim)
            return self._not_found(claim)
        except (KeyError, TypeError, AttributeError, IndexError, ValueError):
            raise Failure(
                "receipt.malformed",
                "a referenced receipt is missing fields or has fields of the wrong type",
                claim_id=claim["id"],
            ) from None

    def _receipt(self, receipt_id: str, claim_id: str, field: str) -> dict:
        if receipt_id not in self._receipts:
            try:
                receipt, data = load_receipt(self.run, receipt_id)
            except InvalidInput as exc:
                raise Failure(exc.code, exc.message, claim_id=claim_id, field=field) from None
            self._receipts[receipt_id] = receipt
            self.receipt_hashes[receipt_id] = sha256_bytes(data)
        return self._receipts[receipt_id]

    # --- quote ---------------------------------------------------------------------

    def _quote(self, claim: dict) -> dict:
        claim_id = claim["id"]

        def fail(code: str, message: str, field: str) -> Failure:
            return Failure(code, message, claim_id=claim_id, field=field)

        row = self.connection.execute(
            f"SELECT {MESSAGE_SELECT} FROM messages WHERE id = ? AND first_seen_revision <= ?",
            (claim["message_id"], self.run.corpus_revision),
        ).fetchone()
        if row is None:
            raise fail("quote.unknown_message", "the message is not in the run's pinned corpus", "message_id")
        message = message_dict(row)
        if payload_of_row(message) != message["payload_sha256"]:
            raise ConfigError("dataset.payload_mismatch", "a stored message no longer matches its payload hash")
        if claim["sender_id"] != message["sender_id"]:
            raise fail("quote.sender_mismatch", "the sender differs from the cited message", "sender_id")
        if not is_canonical_utc(claim["ts_utc"]):
            raise fail("quote.timestamp_format", "ts_utc must be canonical YYYY-MM-DDTHH:MM:SS.ffffffZ", "ts_utc")
        if claim["ts_utc"] != message["ts_utc"]:
            raise fail("quote.timestamp_mismatch", "the timestamp differs from the cited message", "ts_utc")
        if message["text_origin"] == "placeholder":
            raise fail("quote.placeholder", "a media placeholder is not quotable speech", "message_id")
        text = message["text"]
        start, end = claim["start"], claim["end"]
        if not 0 <= start < end <= len(text):
            raise fail("quote.span_out_of_range", "the span must satisfy 0 <= start < end <= code point length", "end")
        if text[start:end] != claim["text"]:
            if claim["text"] in text:
                raise fail("quote.offsets_mismatch", "the text occurs in the message at other code point offsets", "start")
            raise fail("quote.text_mismatch", "the quoted text is not the exact substring at these offsets", "text")
        window = self._context(claim["context_receipt_id"], claim["message_id"], claim_id)
        origins = self.connection.execute(
            """
            SELECT import_revision, raw_locator FROM message_origins
            WHERE message_id = ? AND import_revision <= ? ORDER BY import_revision, raw_locator
            """,
            (message["id"], self.run.corpus_revision),
        ).fetchall()
        return {
            **claim,
            "message": message,
            "window": window,
            "origins": [{"revision": revision, "raw_locator": locator} for revision, locator in origins],
        }

    def _context(self, receipt_id: str, message_id: str, claim_id: str) -> dict:
        def fail(code: str, message: str) -> Failure:
            return Failure(code, message, claim_id=claim_id, field="context_receipt_id")

        receipt = self._receipt(receipt_id, claim_id, "context_receipt_id")
        if receipt["operation"] != "ctx":
            raise fail("context.wrong_operation", "the referenced receipt is not a context receipt")
        if (
            receipt["status"] != "complete"
            or receipt["context_complete"] is not True
            or receipt["has_more"] is not False
            or receipt["error"] is not None
        ):
            raise fail("context.incomplete", "the context receipt is not a final, complete delivery")
        arguments = receipt["arguments"]
        if arguments["message_id"] != message_id:
            raise fail("context.wrong_target", "the context receipt is for a different message")
        radius = arguments["radius"]
        if type(radius) is not int or radius < REQUIRED_CONTEXT_RADIUS:
            raise fail("context.radius_too_small", f"quotes need context radius {REQUIRED_CONTEXT_RADIUS} or more")
        if receipt["arguments_sha256"] != sha256_json(arguments):
            raise fail("receipt.replay_mismatch", "the context receipt's arguments were altered")
        window = compute_window(self.connection, self.run.corpus_revision, message_id, radius)
        summary = window.summary()
        if not strict_equal(receipt["window"], summary) or receipt["window_sha256"] != sha256_json(summary):
            raise fail("receipt.replay_mismatch", "the context window differs from the pinned corpus")
        pages = receipt["pages"]
        if not isinstance(pages, list) or not pages or pages[-1] != receipt_id or len(set(pages)) != len(pages):
            raise fail("context.chain_mismatch", "the context pages do not form one delivery chain")
        deliveries: list[dict] = []
        for index, page_id in enumerate(pages):
            page = self._receipt(page_id, claim_id, "context_receipt_id")
            if index == 0:
                continues = page["cursor_in"] is None
            else:
                continues = page["cursor_in"] == self._receipts[pages[index - 1]]["continuation"]
            if (
                page["operation"] != "ctx"
                or page["arguments_sha256"] != receipt["arguments_sha256"]
                or page["window_sha256"] != receipt["window_sha256"]
                or page["chain"] != pages[:index]
                or not continues
                or (index < len(pages) - 1 and not page["has_more"])
            ):
                raise fail("context.chain_mismatch", "the context pages do not form one delivery chain")
            deliveries.extend(page["results"])
        expected_payloads = {item["id"]: item["payload_sha256"] for item in summary["messages"]}
        if any(expected_payloads.get(item["id"]) != item["payload_sha256"] for item in deliveries):
            raise fail("receipt.replay_mismatch", "a context page delivered a message outside the window")
        try:
            merged = merge_deliveries(summary, deliveries)
        except ConfigError:
            raise fail("context.incomplete", "the pages did not deliver every required message completely") from None
        if not strict_equal(merged, receipt["delivered_all"]):
            raise fail("receipt.replay_mismatch", "the recorded context delivery differs from its pages")
        texts = {message["id"]: message["text"] for message in window.messages}
        for entry in merged:
            for start, end, digest in entry["spans"]:
                if sha256_text(texts[entry["id"]][start:end]) != digest:
                    raise fail("receipt.replay_mismatch", "a delivered segment differs from the canonical text")
        return summary

    # --- statistic -----------------------------------------------------------------

    def _statistic(self, claim: dict) -> dict:
        claim_id = claim["id"]
        receipt = self._receipt(claim["receipt_id"], claim_id, "receipt_id")

        def fail(code: str, message: str, field: str = "receipt_id") -> Failure:
            return Failure(code, message, claim_id=claim_id, field=field)

        if receipt["operation"] != "sql":
            raise fail("statistic.wrong_operation", "the referenced receipt is not a SQL receipt")
        if (
            receipt["status"] != "complete"
            or receipt["complete"] is not True
            or receipt["has_more"] is not False
            or receipt["error"] is not None
        ):
            raise fail("statistic.incomplete_receipt", "statistics need a complete SQL result without truncation, error or timeout")
        outcome = self._replay_sql(receipt, claim_id)
        if (
            outcome.error is not None
            or outcome.has_more
            or outcome.columns != receipt["columns"]
            or not strict_equal(outcome.rows, receipt["rows"])
        ):
            raise fail("receipt.replay_mismatch", "replaying the SQL over a fresh snapshot gave a different result")
        columns = receipt["columns"]
        if columns.count(claim["column"]) != 1:
            raise fail("statistic.unknown_column", "the column is not a unique result column", "column")
        if claim["row"] >= len(receipt["rows"]):
            raise fail("statistic.row_out_of_range", "the row index is outside the result", "row")
        cell = receipt["rows"][claim["row"]][columns.index(claim["column"])]
        if isinstance(cell, dict):
            raise fail("statistic.unsupported_cell", "blob and non-finite cells cannot be statistic values", "value")
        if not strict_equal(cell, claim["value"]):
            raise fail("statistic.value_mismatch", "the claimed value differs in type or value from the result cell", "value")
        return {**claim, "receipt": receipt}

    def _replay_sql(self, receipt: dict, claim_id: str) -> SqlOutcome:
        receipt_id = receipt["receipt_id"]
        if receipt_id in self._sql_replays:
            return self._sql_replays[receipt_id]

        def fail(code: str, message: str) -> Failure:
            return Failure(code, message, claim_id=claim_id, field="receipt_id")

        arguments = receipt["arguments"]
        sql = arguments["sql"]
        if receipt["sql_sha256"] != sha256_text(sql) or receipt["arguments_sha256"] != sha256_json(arguments):
            raise fail("receipt.replay_mismatch", "the SQL receipt's arguments were altered")
        try:
            shape = inspect_sql(sql)
            parameters = validate_parameters(arguments["parameters"], shape)
            validate_scope(arguments["scope"])
            for item in arguments["utc_parameters"]:
                converted, _ = utc_parameters([f"{item['name']}={item['local']}"], item["zone"])
                if converted[item["name"]] != item["utc"] or parameters.get(item["name"]) != item["utc"]:
                    raise fail("receipt.replay_mismatch", "a recorded local-to-UTC parameter conversion is wrong")
        except InvalidInput as exc:
            raise fail(exc.code, exc.message) from None
        limits = receipt["limits"]
        if self._snapshot is None:
            path = temporary_snapshot_path(self.run, "verify-snapshot")
            build_snapshot(self.dataset, self.run, path)
            self._snapshot = path
        outcome = execute_user_sql(
            self._snapshot,
            sql,
            parameters,
            shape,
            max_rows=limits["max_rows"],
            max_steps=limits["max_steps"],
            deadline_ms=max(limits["deadline_ms"], REPLAY_MIN_DEADLINE_MS),
        )
        if outcome.error is not None and outcome.error["code"] == "budget.deadline":
            raise BudgetExceeded("verify.replay_deadline", "replaying a SQL receipt exceeded its deadline")
        self._sql_replays[receipt_id] = outcome
        return outcome

    # --- not found -----------------------------------------------------------------

    def _not_found(self, claim: dict) -> dict:
        claim_id = claim["id"]

        def fail(code: str, message: str) -> Failure:
            return Failure(code, message, claim_id=claim_id, field="receipt_ids")

        receipts = []
        for receipt_id in claim["receipt_ids"]:
            receipt = self._receipt(receipt_id, claim_id, "receipt_ids")
            if receipt["operation"] != "search":
                raise fail("not_found.wrong_operation", "absence findings need search receipts")
            if (
                receipt["status"] != "complete"
                or receipt["has_more"] is not False
                or receipt["continuation"] is not None
                or receipt["cursor_in"] is not None
                or receipt["error"] is not None
            ):
                raise fail("not_found.incomplete_search", "absence needs complete first-page searches without continuation, error or timeout")
            if receipt["total_matches"] != 0 or receipt["results"]:
                raise fail("not_found.has_matches", "a referenced search found matching records")
            try:
                where, params, scope = search_where(receipt["arguments"], self.run.corpus_revision)
            except InvalidInput as exc:
                raise fail(exc.code, exc.message) from None
            if receipt["arguments_sha256"] != sha256_json(receipt["arguments"]) or not strict_equal(scope, receipt["scope"]):
                raise fail("receipt.replay_mismatch", "the search receipt's arguments or scope were altered")
            limits = receipt["limits"]
            budget = Budget(deadline_ms=max(limits["deadline_ms"], REPLAY_MIN_DEADLINE_MS), max_steps=limits["max_steps"])
            try:
                matches = count_matches(self.connection, where, params, budget)
            except sqlite3.OperationalError:
                if budget.reason is None:
                    raise
                raise BudgetExceeded("verify.replay_budget", "replaying a search exceeded its budget") from None
            if matches != 0:
                raise fail("receipt.replay_mismatch", "replaying the search found matching records")
            receipts.append(receipt)
        return {**claim, "receipts": receipts}


def _read_record(run: Run) -> dict | None:
    try:
        record = loads_strict(run.verification_path.read_bytes(), what="verification")
    except (FileNotFoundError, InvalidInput):
        return None
    return record if isinstance(record, dict) else None


def _remove_stale_report(run: Run, previous: dict | None) -> None:
    """Remove a report.md that an earlier pass rendered, so it cannot pass for verified."""
    if previous and previous.get("status") == "pass" and _file_sha256(run.report_path) == previous.get("report_sha256"):
        run.report_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str | None:
    try:
        return sha256_bytes(path.read_bytes())
    except (FileNotFoundError, NotADirectoryError):
        return None


def _is_int(value: object) -> bool:
    return type(value) is int


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _receipt_id(value: object) -> bool:
    return isinstance(value, str) and bool(RECEIPT_ID_RE.fullmatch(value))
