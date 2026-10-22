"""``q sql``: one reviewed read-only statement over a freshly built pinned snapshot.

The snapshot is rebuilt for every call (no caching before its cost is
measured) and kept afterwards as the run's disposable ``sql-snapshot.sqlite3``.
Results are typed cells. A result is ``complete`` only if no row or byte cap
cut it; incomplete, failed, denied and timed-out results are recorded but are
ineligible for statistical claims.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import BudgetExceeded, InvalidInput
from .jsonutil import sha256_json, sha256_text
from .query import QueryResult, settle_size
from .receipts import new_receipt_id, receipt_header, write_receipt
from .runs import Run, load_run
from .sqlsandbox import (
    DEFAULT_SNAPSHOT_MAX_BYTES,
    DEFAULT_SNAPSHOT_MAX_SECONDS,
    build_snapshot,
    execute_user_sql,
    inspect_sql,
    remove_snapshot,
    temporary_snapshot_path,
    utc_parameters,
    validate_parameters,
    validate_scope,
)
from .store import Dataset
from .timeutil import utc_now


@dataclass(frozen=True)
class SqlLimits:
    max_rows: int = 200
    max_bytes: int = 64 * 1024
    deadline_ms: int = 5000
    max_steps: int = 50_000_000
    snapshot_max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES
    snapshot_max_seconds: float = DEFAULT_SNAPSHOT_MAX_SECONDS

    def checked(self) -> SqlLimits:
        integer_bounds = {
            "max_rows": (1, 1000),
            "max_bytes": (1024, 1024 * 1024),
            "deadline_ms": (1, 600_000),
            "max_steps": (1000, 10**12),
            "snapshot_max_bytes": (16 * 1024, 64 * 1024**3),
        }
        for name, (low, high) in integer_bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise InvalidInput("limits.out_of_range", f"{name} must be an integer from {low} to {high}")
        if not isinstance(self.snapshot_max_seconds, (int, float)) or not 0 < self.snapshot_max_seconds <= 3600:
            raise InvalidInput("limits.out_of_range", "snapshot_max_seconds must be above 0 and at most 3600")
        return self

    def to_dict(self) -> dict:
        return {
            "max_rows": self.max_rows,
            "max_bytes": self.max_bytes,
            "deadline_ms": self.deadline_ms,
            "max_steps": self.max_steps,
            "snapshot_max_bytes": self.snapshot_max_bytes,
            "snapshot_max_seconds": self.snapshot_max_seconds,
        }


def sql_query(
    dataset: Dataset,
    run_id: str,
    sql: str,
    parameters: object,
    scope: object,
    *,
    utc_assignments: list[str] | tuple[str, ...] = (),
    tz: str | None = None,
    limits: SqlLimits = SqlLimits(),
) -> QueryResult:
    limits = limits.checked()
    shape = inspect_sql(sql)
    if not isinstance(parameters, dict):
        raise InvalidInput("sql.parameters", "parameters must be a JSON object of named values")
    utc_values, utc_records = utc_parameters(list(utc_assignments), tz)
    if utc_values.keys() & parameters.keys():
        raise InvalidInput("sql.parameters", "a --utc-param name repeats a parameter-file name")
    bound = validate_parameters({**parameters, **utc_values}, shape)
    scope = validate_scope(scope)
    connection = dataset.connect_ro()
    try:
        run = load_run(dataset, connection, run_id)
    finally:
        connection.close()

    receipt_id = new_receipt_id()
    arguments = {"sql": sql, "parameters": bound, "utc_parameters": utc_records, "scope": scope}
    base = {
        **receipt_header(run, receipt_id, "sql"),
        "arguments": arguments,
        "arguments_sha256": sha256_json(arguments),
        "sql_sha256": sha256_text(sql),
        "top_level_order_by": shape.top_level_order_by,
        "limits": limits.to_dict(),
        "executed_at": utc_now(),
    }
    temporary = temporary_snapshot_path(run, "sql-snapshot")
    try:
        try:
            snapshot = build_snapshot(
                dataset,
                run,
                temporary,
                max_bytes=limits.snapshot_max_bytes,
                max_seconds=limits.snapshot_max_seconds,
            )
        except BudgetExceeded as exc:
            _record_failure(run, base, None, {"code": exc.code, "message": exc.message}, steps=0, elapsed_ms=0)
            raise BudgetExceeded(exc.code, exc.message, details={"receipt_id": receipt_id}) from None
        outcome = execute_user_sql(
            temporary,
            sql,
            bound,
            shape,
            max_rows=limits.max_rows,
            max_steps=limits.max_steps,
            deadline_ms=limits.deadline_ms,
        )
        os.replace(temporary, run.snapshot_path)
    finally:
        remove_snapshot(temporary)

    if outcome.error is not None:
        _record_failure(run, base, snapshot.to_dict(), outcome.error, steps=outcome.steps, elapsed_ms=outcome.elapsed_ms)
        error_class = BudgetExceeded if outcome.error["code"].startswith("budget.") else InvalidInput
        details = {"receipt_id": receipt_id}
        if outcome.error.get("denied"):
            details["denied"] = outcome.error["denied"]
        # SQLite's own message can quote SQL fragments; it stays in the private receipt.
        message = (
            "SQLite rejected the statement; its message is in the private receipt"
            if outcome.error["code"] == "sql.error"
            else outcome.error["message"]
        )
        raise error_class(outcome.error["code"], message, details=details)

    envelope = {
        "receipt_id": receipt_id,
        "operation": "sql",
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "corpus_revision": run.corpus_revision,
        "columns": outcome.columns,
        "scope": scope,
        "snapshot": {"rows": snapshot.rows, "bytes": snapshot.bytes},
    }
    kept, output = _fit_rows(envelope, outcome.rows, outcome.has_more, limits.max_bytes)
    has_more = outcome.has_more or kept < len(outcome.rows)
    truncation = None if not has_more else ("max_bytes" if kept < len(outcome.rows) else "max_rows")
    receipt = {
        **base,
        "snapshot": snapshot.to_dict(),
        "status": "incomplete" if has_more else "complete",
        "complete": not has_more,
        "has_more": has_more,
        "truncation": truncation,
        "columns": outcome.columns,
        "rows": outcome.rows[:kept],
        "counts": {"rows": kept, "response_bytes": len(output)},
        "error": None,
        "steps": outcome.steps,
        "elapsed_ms": outcome.elapsed_ms,
    }
    write_receipt(run, receipt)
    return QueryResult(receipt_id=receipt_id, status=receipt["status"], output=output, receipt=receipt)


def _fit_rows(envelope: dict, rows: list[list], has_more: bool, max_bytes: int) -> tuple[int, bytes]:
    """Largest prefix of ``rows`` whose response fits ``max_bytes``."""

    def render(count: int) -> bytes:
        more = has_more or count < len(rows)
        truncation = None if not more else ("max_bytes" if count < len(rows) else "max_rows")
        return settle_size({
            **envelope,
            "status": "incomplete" if more else "complete",
            "complete": not more,
            "has_more": more,
            "truncation": truncation,
            "row_count": count,
            "rows": rows[:count],
            "response_bytes": 0,
        })

    output = render(len(rows))
    if len(output) <= max_bytes:
        return len(rows), output
    low, high = 0, len(rows) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if len(render(middle)) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    output = render(low)
    if len(output) > max_bytes:
        raise InvalidInput("limits.max_bytes_too_small", "max-bytes is too small for the result envelope")
    return low, output


def _record_failure(run: Run, base: dict, snapshot: dict | None, error: dict, *, steps: int, elapsed_ms: int) -> None:
    write_receipt(run, {
        **base,
        "snapshot": snapshot,
        "status": "error",
        "complete": False,
        "has_more": False,
        "truncation": None,
        "columns": [],
        "rows": [],
        "counts": {"rows": 0, "response_bytes": 0},
        "error": error,
        "steps": steps,
        "elapsed_ms": elapsed_ms,
    })
