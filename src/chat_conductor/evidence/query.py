"""Bounded, pinned, read-only queries: ``q search``, ``q range`` and ``q ctx``.

Every operation reads only messages with ``first_seen_revision`` at or below
the run's ``corpus_revision`` (and origins with ``import_revision`` at or below
it) through a read-only connection with step and deadline budgets. Responses
are capped by rows and UTF-8 JSON bytes. A message that does not fit is
delivered as explicitly labelled text segments across pages, never as a
silently truncated snippet. Every call writes a receipt; a budget failure
writes an error receipt and is never reported as an empty result.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

from ..text import escape_like
from .budget import Budget
from .errors import BudgetExceeded, ConfigError, InvalidInput
from .jsonutil import dumps_canonical, sha256_json, sha256_text
from .paths import MESSAGE_ID_RE
from .receipts import decode_cursor, encode_cursor, load_receipt, new_receipt_id, receipt_header, write_receipt
from .records import KINDS, TEXT_ORIGINS
from .runs import Run, load_run
from .store import MESSAGE_SELECT, Dataset, message_dict
from .timeutil import convert_window, utc_now

DEFAULT_MAX_ROWS = 200
MAX_ROWS_CEILING = 1000
DEFAULT_MAX_BYTES = 64 * 1024
MIN_MAX_BYTES = 1024
MAX_BYTES_CEILING = 1024 * 1024
DEFAULT_DEADLINE_MS = 5000
MAX_DEADLINE_MS = 600_000
DEFAULT_MAX_STEPS = 200_000_000
CONTEXT_RADIUS = 20
MAX_CONTEXT_RADIUS = 200
CASE_POLICIES = ("ascii-insensitive", "sensitive")
GLOBAL_ORDERING = "ts_utc, source_namespace, conversation_id, source_order, id"
CONVERSATION_ORDERING = "ts_utc, source_order, id within one source namespace and conversation"
_ORDER_BY = "ORDER BY ts_utc, source_namespace, conversation_id, source_order, id"
_KEY_TUPLE = "(ts_utc, source_namespace, conversation_id, source_order, id)"
_CASE_DESCRIPTIONS = {
    "ascii-insensitive": (
        "ASCII letters A-Z match case-insensitively (SQLite LIKE); every other character, "
        "including CJK and accented letters, matches exactly by code point"
    ),
    "sensitive": "every character matches exactly by code point",
}
_EDGE_NOTE = "an edge flag marks the boundary of the imported corpus at this revision, not of the real conversation"
# Seven digits always cover the final response size (at most 1 MiB).
_SIZE_PLACEHOLDER = 9_999_999


@dataclass(frozen=True)
class Limits:
    max_rows: int = DEFAULT_MAX_ROWS
    max_bytes: int = DEFAULT_MAX_BYTES
    deadline_ms: int = DEFAULT_DEADLINE_MS
    max_steps: int = DEFAULT_MAX_STEPS

    def checked(self) -> Limits:
        bounds = {
            "max_rows": (1, MAX_ROWS_CEILING),
            "max_bytes": (MIN_MAX_BYTES, MAX_BYTES_CEILING),
            "deadline_ms": (1, MAX_DEADLINE_MS),
            "max_steps": (1000, 10**12),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise InvalidInput("limits.out_of_range", f"{name} must be an integer from {low} to {high}")
        return self

    def to_dict(self) -> dict:
        return {
            "max_rows": self.max_rows,
            "max_bytes": self.max_bytes,
            "deadline_ms": self.deadline_ms,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class QueryResult:
    receipt_id: str
    status: str
    output: bytes
    receipt: dict


@dataclass(frozen=True)
class Window:
    target: dict
    messages: list[dict]
    target_index: int
    edge_before: bool
    edge_after: bool
    radius: int

    def summary(self) -> dict:
        return {
            "target_id": self.target["id"],
            "source_namespace": self.target["source_namespace"],
            "conversation_id": self.target["conversation_id"],
            "radius": self.radius,
            "target_index": self.target_index,
            "before_count": self.target_index,
            "after_count": len(self.messages) - self.target_index - 1,
            "edge_before": self.edge_before,
            "edge_after": self.edge_after,
            "equal_time_order_uncertain": any(
                first["ts_utc"] == second["ts_utc"]
                and "timestamp_only" in (first["order_basis"], second["order_basis"])
                for first, second in zip(self.messages, self.messages[1:])
            ),
            "messages": [
                {
                    "id": message["id"],
                    "payload_sha256": message["payload_sha256"],
                    "text_length": len(message["text"]),
                    "text_sha256": sha256_text(message["text"]),
                }
                for message in self.messages
            ],
        }


@dataclass
class _Item:
    message: dict
    extra: dict | None = None


@dataclass
class _Page:
    rows: list[dict]
    deliveries: list[dict]
    has_more: bool
    cursor: str | None
    output: bytes


# --- argument normalization shared with the verifier ---------------------------------


def search_where(arguments: dict, revision: int) -> tuple[str, list, dict]:
    """Validate recorded search arguments; return WHERE clause, parameters and scope."""
    terms = arguments.get("terms")
    if not isinstance(terms, list) or not terms:
        raise InvalidInput("search.terms_missing", "give at least one literal search term")
    for term in terms:
        if not isinstance(term, str) or not term or "\x00" in term:
            raise InvalidInput("search.term_invalid", "search terms must be nonempty text")
        _require_unicode(term, "search.term_invalid")
    case = arguments.get("case")
    if case not in CASE_POLICIES:
        raise InvalidInput("search.case_policy", "case must be 'ascii-insensitive' or 'sensitive'")
    clauses, params, filters, window = _common_filters(arguments, revision)
    for term in terms:
        if case == "sensitive":
            clauses.append("instr(text, ?) > 0")
            params.append(term)
        else:
            clauses.append("text LIKE ? ESCAPE '\\'")
            params.append(f"%{escape_like(term)}%")
    scope = {
        "match": "literal substring of message text; every term must match (AND); "
        "no stemming, synonyms, normalization or query syntax",
        "case": case,
        "case_policy": _CASE_DESCRIPTIONS[case],
        "filters": filters,
        "time_window": window,
        "includes": "all kinds and text origins (exported text, exported transcripts, placeholders) unless filtered",
    }
    return " AND ".join(clauses), params, scope


def range_where(arguments: dict, revision: int) -> tuple[str, list, dict]:
    for name in ("source_namespace", "conversation_id"):
        if not isinstance(arguments.get(name), str) or not arguments[name]:
            raise InvalidInput("range.conversation_required", "q range needs --source-namespace and --conversation")
    clauses, params, filters, window = _common_filters(arguments, revision)
    scope = {
        "match": "every message in one conversation within the filters, in chronological order",
        "filters": filters,
        "time_window": window,
        "includes": "all kinds and text origins unless filtered",
    }
    return " AND ".join(clauses), params, scope


def _common_filters(arguments: dict, revision: int) -> tuple[list[str], list, dict, dict]:
    clauses = ["first_seen_revision <= ?"]
    params: list = [revision]
    filters: dict = {}
    for name in ("source_namespace", "conversation_id", "sender_id"):
        value = arguments.get(name)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise InvalidInput("query.filter_invalid", f"{name} must be nonempty text")
            _require_unicode(value, "query.filter_invalid")
            clauses.append(f"{name} = ?")
            params.append(value)
        filters[name] = value
    for name, allowed in (("kinds", KINDS), ("text_origins", TEXT_ORIGINS)):
        values = arguments.get(name)
        if values is None:
            filters[name] = "all"
            continue
        if not isinstance(values, list) or not values or any(value not in allowed for value in values):
            raise InvalidInput("query.filter_invalid", f"{name} accepts: {', '.join(allowed)}")
        normalized = sorted(set(values))
        column = "kind" if name == "kinds" else "text_origin"
        clauses.append(f"{column} IN ({', '.join('?' for _ in normalized)})")
        params.extend(normalized)
        filters[name] = normalized
    window = convert_window(arguments.get("from"), arguments.get("to"), arguments.get("tz"))
    if window["from"] is not None:
        clauses.append("ts_utc >= ?")
        params.append(window["from"]["utc"])
    if window["to"] is not None:
        clauses.append("ts_utc < ?")
        params.append(window["to"]["utc"])
    return clauses, params, filters, window


def _require_unicode(value: str, code: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidInput(code, "arguments must be valid Unicode text") from None


# --- operations ------------------------------------------------------------------------


def search(dataset: Dataset, run_id: str, arguments: dict, *, limits: Limits = Limits(), cursor: str | None = None) -> QueryResult:
    limits = limits.checked()
    connection = dataset.connect_ro()
    try:
        run = load_run(dataset, connection, run_id)
        where, params, scope = search_where(arguments, run.corpus_revision)
        return _stream(connection, run, "search", arguments, where, params, scope, limits, cursor, GLOBAL_ORDERING)
    finally:
        connection.close()


def range_query(dataset: Dataset, run_id: str, arguments: dict, *, limits: Limits = Limits(), cursor: str | None = None) -> QueryResult:
    limits = limits.checked()
    connection = dataset.connect_ro()
    try:
        run = load_run(dataset, connection, run_id)
        where, params, scope = range_where(arguments, run.corpus_revision)
        return _stream(connection, run, "range", arguments, where, params, scope, limits, cursor, CONVERSATION_ORDERING)
    finally:
        connection.close()


def count_matches(connection: sqlite3.Connection, where: str, params: list, budget: Budget) -> int:
    budget.install(connection)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM messages WHERE {where}", params).fetchone()[0])
    finally:
        connection.set_progress_handler(None, 0)


def _stream(
    connection: sqlite3.Connection,
    run: Run,
    operation: str,
    arguments: dict,
    where: str,
    params: list,
    scope: dict,
    limits: Limits,
    cursor: str | None,
    ordering: str,
) -> QueryResult:
    arguments_sha256 = sha256_json(arguments)
    binding = {"run": run.run_id, "op": operation, "args": arguments_sha256, "rev": run.corpus_revision, "order": ordering}
    position = decode_cursor(cursor, binding) if cursor is not None else None
    key, first_seg = _cursor_position(position)
    receipt_id = new_receipt_id()
    base = {
        **receipt_header(run, receipt_id, operation),
        "arguments": arguments,
        "arguments_sha256": arguments_sha256,
        "scope": scope,
        "ordering": ordering,
        "limits": limits.to_dict(),
        "cursor_in": cursor,
        "executed_at": utc_now(),
    }
    budget = Budget(deadline_ms=limits.deadline_ms, max_steps=limits.max_steps)
    budget.install(connection)
    try:
        total = int(connection.execute(f"SELECT COUNT(*) FROM messages WHERE {where}", params).fetchone()[0])
        page_where, page_params = where, list(params)
        if key is not None:
            page_where += f" AND {_KEY_TUPLE} {'>=' if first_seg else '>'} (?, ?, ?, ?, ?)"
            page_params.extend(key)
        rows = connection.execute(
            f"SELECT {MESSAGE_SELECT} FROM messages WHERE {page_where} {_ORDER_BY} LIMIT ?",
            (*page_params, limits.max_rows + 1),
        ).fetchall()
    except sqlite3.OperationalError:
        if budget.reason is None:
            raise
        _budget_failure(run, base, budget, total_field=True)
    finally:
        connection.set_progress_handler(None, 0)
    messages = [message_dict(row) for row in rows]
    if first_seg and (not messages or _key(messages[0]) != key or first_seg >= len(messages[0]["text"])):
        raise InvalidInput("cursor.stale", "the cursor no longer points at a partially delivered message")
    items = [_Item(message) for message in messages]

    def make_cursor(index: int, seg: int) -> str:
        anchor = items[index] if seg else items[index - 1]
        return encode_cursor({**binding, "key": _key(anchor.message), "seg": seg})

    envelope = {
        "receipt_id": receipt_id,
        "operation": operation,
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "corpus_revision": run.corpus_revision,
        "ordering": ordering,
        "scope": scope,
        "total_matches": total,
    }
    page = _build_page(
        items,
        first_seg=first_seg,
        envelope=envelope,
        dynamic=lambda has_more: {"status": "incomplete" if has_more else "complete"},
        limits=limits,
        make_cursor=make_cursor,
    )
    receipt = {
        **base,
        "status": "incomplete" if page.has_more else "complete",
        "has_more": page.has_more,
        "continuation": page.cursor,
        "total_matches": total,
        "results": page.deliveries,
        "counts": {"rows": len(page.rows), "response_bytes": len(page.output)},
        "error": None,
        "elapsed_ms": budget.elapsed_ms(),
    }
    write_receipt(run, receipt)
    return QueryResult(receipt_id=receipt_id, status=receipt["status"], output=page.output, receipt=receipt)


def _cursor_position(position: dict | None) -> tuple[list[str] | None, int]:
    if position is None:
        return None, 0
    key, seg = position.get("key"), position.get("seg")
    if (
        not isinstance(key, list)
        or len(key) != 5
        or not all(isinstance(part, str) for part in key)
        or type(seg) is not int
        or seg < 0
    ):
        raise InvalidInput("cursor.malformed", "the continuation cursor is malformed")
    return key, seg


def _key(message: dict) -> list[str]:
    return [message["ts_utc"], message["source_namespace"], message["conversation_id"], message["source_order"], message["id"]]


def context(
    dataset: Dataset,
    run_id: str,
    message_id: str,
    *,
    radius: int = CONTEXT_RADIUS,
    limits: Limits = Limits(),
    cursor: str | None = None,
) -> QueryResult:
    limits = limits.checked()
    if not isinstance(message_id, str) or not MESSAGE_ID_RE.fullmatch(message_id):
        raise InvalidInput("ctx.invalid_message_id", "message ids look like m_<64 lowercase hex digits>")
    if type(radius) is not int or not 0 <= radius <= MAX_CONTEXT_RADIUS:
        raise InvalidInput("ctx.radius", f"radius must be an integer from 0 to {MAX_CONTEXT_RADIUS}")
    arguments = {"message_id": message_id, "radius": radius}
    arguments_sha256 = sha256_json(arguments)
    connection = dataset.connect_ro()
    try:
        run = load_run(dataset, connection, run_id)
        receipt_id = new_receipt_id()
        base = {
            **receipt_header(run, receipt_id, "ctx"),
            "arguments": arguments,
            "arguments_sha256": arguments_sha256,
            "ordering": CONVERSATION_ORDERING,
            "limits": limits.to_dict(),
            "cursor_in": cursor,
            "executed_at": utc_now(),
        }
        budget = Budget(deadline_ms=limits.deadline_ms, max_steps=limits.max_steps)
        budget.install(connection)
        try:
            window = compute_window(connection, run.corpus_revision, message_id, radius)
            origins = _origins(connection, run.corpus_revision, [message["id"] for message in window.messages])
        except sqlite3.OperationalError:
            if budget.reason is None:
                raise
            _budget_failure(run, base, budget, total_field=False)
        finally:
            connection.set_progress_handler(None, 0)
    finally:
        connection.close()

    summary = window.summary()
    window_sha256 = sha256_json(summary)
    binding = {
        "run": run.run_id,
        "op": "ctx",
        "args": arguments_sha256,
        "rev": run.corpus_revision,
        "order": CONVERSATION_ORDERING,
        "window": window_sha256,
    }
    chain: list[str] = []
    earlier: list[dict] = []
    start_index, first_seg = 0, 0
    if cursor is not None:
        position = decode_cursor(cursor, binding)
        previous, _ = load_receipt(run, position.get("prev"))
        if (
            previous.get("operation") != "ctx"
            or previous.get("arguments_sha256") != arguments_sha256
            or previous.get("continuation") != cursor
        ):
            raise InvalidInput("cursor.chain", "the cursor does not continue a context page of this run")
        chain = [*previous["chain"], previous["receipt_id"]]
        for page_id in chain:
            earlier.extend(load_receipt(run, page_id)[0]["results"])
        start_index, first_seg = position.get("index"), position.get("seg")
        if (
            type(start_index) is not int
            or type(first_seg) is not int
            or not 0 <= start_index < len(window.messages)
            or not 0 <= first_seg < len(window.messages[start_index]["text"])
        ):
            raise InvalidInput("cursor.malformed", "the continuation cursor is malformed")

    items = [
        _Item(message, {"position": index - window.target_index, "origins": origins[message["id"]]})
        for index, message in enumerate(window.messages)
    ][start_index:]

    def make_cursor(index: int, seg: int) -> str:
        return encode_cursor({**binding, "prev": receipt_id, "index": start_index + index, "seg": seg})

    public_window = {key: value for key, value in summary.items() if key != "messages"}
    public_window["message_count"] = len(window.messages)
    public_window["edge_note"] = _EDGE_NOTE
    envelope = {
        "receipt_id": receipt_id,
        "operation": "ctx",
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "corpus_revision": run.corpus_revision,
        "ordering": CONVERSATION_ORDERING,
        "window": public_window,
        "page": len(chain) + 1,
    }
    page = _build_page(
        items,
        first_seg=first_seg,
        envelope=envelope,
        dynamic=lambda has_more: {
            "status": "incomplete" if has_more else "complete",
            "context_complete": not has_more,
        },
        limits=limits,
        make_cursor=make_cursor,
    )
    receipt = {
        **base,
        "window": summary,
        "window_sha256": window_sha256,
        "chain": chain,
        "status": "incomplete" if page.has_more else "complete",
        "has_more": page.has_more,
        "continuation": page.cursor,
        "context_complete": not page.has_more,
        "results": page.deliveries,
        "counts": {"rows": len(page.rows), "response_bytes": len(page.output)},
        "error": None,
        "elapsed_ms": budget.elapsed_ms(),
    }
    if not page.has_more:
        receipt["pages"] = [*chain, receipt_id]
        receipt["delivered_all"] = merge_deliveries(summary, [*earlier, *page.deliveries])
    write_receipt(run, receipt)
    return QueryResult(receipt_id=receipt_id, status=receipt["status"], output=page.output, receipt=receipt)


def compute_window(connection: sqlite3.Connection, revision: int, message_id: str, radius: int) -> Window:
    """Target plus up to ``radius`` neighbours each side in its conversation, all kinds."""
    row = connection.execute(
        f"SELECT {MESSAGE_SELECT} FROM messages WHERE id = ? AND first_seen_revision <= ?",
        (message_id, revision),
    ).fetchone()
    if row is None:
        raise InvalidInput("ctx.unknown_message", "the message is not in this run's pinned corpus")
    target = message_dict(row)
    scope = "source_namespace = ? AND conversation_id = ? AND first_seen_revision <= ?"
    params = (target["source_namespace"], target["conversation_id"], revision, target["ts_utc"], target["source_order"], target["id"])
    before = connection.execute(
        f"SELECT {MESSAGE_SELECT} FROM messages WHERE {scope} AND (ts_utc, source_order, id) < (?, ?, ?) "
        "ORDER BY ts_utc DESC, source_order DESC, id DESC LIMIT ?",
        (*params, radius + 1),
    ).fetchall()
    after = connection.execute(
        f"SELECT {MESSAGE_SELECT} FROM messages WHERE {scope} AND (ts_utc, source_order, id) > (?, ?, ?) "
        "ORDER BY ts_utc, source_order, id LIMIT ?",
        (*params, radius + 1),
    ).fetchall()
    earlier = [message_dict(item) for item in reversed(before[:radius])]
    later = [message_dict(item) for item in after[:radius]]
    return Window(
        target=target,
        messages=[*earlier, target, *later],
        target_index=len(earlier),
        edge_before=len(before) <= radius,
        edge_after=len(after) <= radius,
        radius=radius,
    )


def merge_deliveries(summary: dict, deliveries: list[dict]) -> list[dict]:
    """Combine delivered spans per window message and require full, exact coverage."""
    spans: dict[str, list[list]] = {}
    for item in deliveries:
        spans.setdefault(item["id"], []).append([item["start"], item["end"], item["sha256"]])
    merged = []
    for message in summary["messages"]:
        covered = sorted(spans.get(message["id"], []))
        position = 0
        for start, end, _ in covered:
            if start != position or end <= start:
                raise ConfigError("ctx.coverage", "context pages did not tile a message's text exactly")
            position = end
        if position != message["text_length"]:
            raise ConfigError("ctx.coverage", "context pages did not deliver every required message completely")
        merged.append({
            "id": message["id"],
            "payload_sha256": message["payload_sha256"],
            "text_length": message["text_length"],
            "spans": covered,
        })
    return merged


def _origins(connection: sqlite3.Connection, revision: int, message_ids: list[str]) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {message_id: [] for message_id in message_ids}
    for offset in range(0, len(message_ids), 400):
        chunk = message_ids[offset:offset + 400]
        rows = connection.execute(
            f"""
            SELECT message_id, import_revision, raw_locator FROM message_origins
            WHERE import_revision <= ? AND message_id IN ({', '.join('?' for _ in chunk)})
            ORDER BY message_id, import_revision, raw_locator
            """,
            (revision, *chunk),
        ).fetchall()
        for message_id, import_revision, raw_locator in rows:
            found[message_id].append({"revision": import_revision, "raw_locator": raw_locator})
    return found


def _budget_failure(run: Run, base: dict, budget: Budget, *, total_field: bool) -> None:
    receipt = {
        **base,
        "status": "error",
        "has_more": False,
        "continuation": None,
        "results": [],
        "counts": {"rows": 0, "response_bytes": 0},
        "error": {"code": f"budget.{budget.reason}", "steps": budget.steps},
        "elapsed_ms": budget.elapsed_ms(),
    }
    if total_field:
        receipt["total_matches"] = None
    else:
        receipt["context_complete"] = False
    write_receipt(run, receipt)
    raise BudgetExceeded(
        f"budget.{budget.reason}",
        "the query exceeded its execution budget; no result, including an empty one, is implied",
        details={"receipt_id": base["receipt_id"]},
    )


# --- response pages --------------------------------------------------------------------


def _row(message: dict, start: int, end: int, extra: dict | None) -> dict:
    text = message["text"]
    piece = text[start:end]
    whole = start == 0 and end == len(text)
    row = {
        "id": message["id"],
        "source_namespace": message["source_namespace"],
        "conversation_id": message["conversation_id"],
        "native_message_id": message["native_message_id"],
        "sender_id": message["sender_id"],
        "ts_utc": message["ts_utc"],
        "ts_original": message["ts_original"],
        "source_timezone": message["source_timezone"],
        "source_order": message["source_order"],
        "order_basis": message["order_basis"],
        "kind": message["kind"],
        "text_origin": message["text_origin"],
        "payload_sha256": message["payload_sha256"],
        "text_length": len(text),
        "text": piece,
        "text_complete": whole,
        "segment": None if whole else {
            "start": start,
            "end": end,
            "final": end == len(text),
            "sha256": sha256_text(piece),
        },
    }
    if extra:
        row.update(extra)
    return row


def _delivery(message: dict, start: int, end: int) -> dict:
    return {
        "id": message["id"],
        "payload_sha256": message["payload_sha256"],
        "text_length": len(message["text"]),
        "start": start,
        "end": end,
        "sha256": sha256_text(message["text"][start:end]),
    }


def _build_page(
    items: list[_Item],
    *,
    first_seg: int,
    envelope: dict,
    dynamic: Callable[[bool], dict],
    limits: Limits,
    make_cursor: Callable[[int, int], str],
) -> _Page:
    """Fill one response within the row and byte limits, segmenting an oversized text."""
    encoded_rows: list[bytes] = []
    rows: list[dict] = []
    deliveries: list[dict] = []
    index, seg = 0, first_seg

    def size(candidate: list[bytes], has_more: bool, cursor: str | None) -> int:
        shell = {
            **envelope,
            **dynamic(has_more),
            "has_more": has_more,
            "cursor": cursor,
            "row_count": len(candidate),
            "response_bytes": _SIZE_PLACEHOLDER,
            "rows": [],
        }
        return len(dumps_canonical(shell).encode("utf-8")) + sum(map(len, candidate)) + max(0, len(candidate) - 1)

    def accept(row: dict, encoded: bytes, message: dict, start: int, end: int) -> None:
        encoded_rows.append(encoded)
        rows.append(row)
        deliveries.append(_delivery(message, start, end))

    while index < len(items) and len(rows) < limits.max_rows:
        item = items[index]
        length = len(item.message["text"])
        row = _row(item.message, seg, length, item.extra)
        encoded = dumps_canonical(row).encode("utf-8")
        more_after = index + 1 < len(items)
        if size([*encoded_rows, encoded], more_after, make_cursor(index + 1, 0) if more_after else None) <= limits.max_bytes:
            accept(row, encoded, item.message, seg, length)
            index, seg = index + 1, 0
            continue
        if rows:
            break

        def fits(end: int) -> bool:
            candidate = dumps_canonical(_row(item.message, seg, end, item.extra)).encode("utf-8")
            return size([candidate], True, make_cursor(index, end)) <= limits.max_bytes

        low, high = seg + 1, length - 1
        if high < low or not fits(low):
            raise InvalidInput("limits.max_bytes_too_small", "max-bytes is too small to deliver one text segment")
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle):
                low = middle
            else:
                high = middle - 1
        row = _row(item.message, seg, low, item.extra)
        accept(row, dumps_canonical(row).encode("utf-8"), item.message, seg, low)
        seg = low
        break

    has_more = index < len(items)
    cursor = make_cursor(index, seg) if has_more else None
    final = {
        **envelope,
        **dynamic(has_more),
        "has_more": has_more,
        "cursor": cursor,
        "row_count": len(rows),
        "response_bytes": 0,
        "rows": rows,
    }
    output = settle_size(final)
    if len(output) > limits.max_bytes:
        raise ConfigError("limits.internal", "a response exceeded its byte budget")
    return _Page(rows=rows, deliveries=deliveries, has_more=has_more, cursor=cursor, output=output)


def settle_size(envelope: dict) -> bytes:
    """Serialize with ``response_bytes`` equal to the serialized length itself."""
    for _ in range(5):
        data = dumps_canonical(envelope).encode("utf-8")
        if envelope["response_bytes"] == len(data):
            return data
        envelope["response_bytes"] = len(data)
    raise ConfigError("limits.internal", "response size did not settle")
