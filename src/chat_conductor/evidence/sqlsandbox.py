"""Read-only user SQL over a physically isolated, pinned ``visible_messages`` snapshot.

The connection that prepares user SQL never opens or attaches the canonical
database. A trusted reader copies only the run's pinned messages into a new
private snapshot file whose sole queryable table is ``visible_messages``; that
writer is closed and the file is reopened with ``mode=ro``, ``query_only``,
hard SQLite limits, an authorizer and step/deadline budgets before any user
SQL is prepared. A view over the canonical table plus an authorizer would not
suffice: a CTE named like the view impersonates the view's origin in the
authorizer callback and can read the base table.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import budget as budget_module
from .budget import Budget
from .errors import BudgetExceeded, ConfigError, InvalidInput
from .jsonutil import sha256_bytes
from .runs import Run
from .store import MESSAGE_COLUMNS, MESSAGE_SELECT, Dataset
from .timeutil import load_zone, local_bound

SNAPSHOT_DERIVATION = "visible-messages-v1"
DEFAULT_SNAPSHOT_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_SNAPSHOT_MAX_SECONDS = 30.0
SQL_TEXT_LIMIT = 100_000
SNAPSHOT_DDL = (
    """
    CREATE TABLE visible_messages (
        id TEXT PRIMARY KEY NOT NULL,
        source_namespace TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        native_message_id TEXT NOT NULL,
        sender_id TEXT NOT NULL,
        ts_utc TEXT NOT NULL,
        ts_original TEXT NOT NULL,
        source_timezone TEXT NOT NULL,
        source_order TEXT NOT NULL,
        order_basis TEXT NOT NULL,
        kind TEXT NOT NULL,
        text_origin TEXT NOT NULL,
        text TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        first_seen_revision INTEGER NOT NULL
    ) WITHOUT ROWID
    """,
    "CREATE INDEX visible_by_conversation_time ON visible_messages "
    "(source_namespace, conversation_id, ts_utc, source_order, id)",
    "CREATE INDEX visible_by_sender_time ON visible_messages (source_namespace, sender_id, ts_utc, id)",
)
# Reviewed deterministic functions. Excluded on purpose: date/time functions
# ('now', 'localtime'), random*, *blob, char, printf/format, group_concat and
# string_agg (row-order dependent), json*, load_extension, sqlite_* and
# changes()-style connection state.
ALLOWED_FUNCTIONS = frozenset({
    "abs", "avg", "coalesce", "count", "cume_dist", "dense_rank", "first_value", "glob", "ifnull",
    "iif", "instr", "lag", "last_value", "lead", "length", "like", "lower", "ltrim", "max", "min",
    "nth_value", "ntile", "nullif", "percent_rank", "rank", "replace", "round", "row_number",
    "rtrim", "substr", "substring", "sum", "total", "trim", "typeof", "unicode", "upper",
})
_HARD_LIMITS = (
    (sqlite3.SQLITE_LIMIT_ATTACHED, 0),
    (sqlite3.SQLITE_LIMIT_LENGTH, 16 * 1024 * 1024),
    (sqlite3.SQLITE_LIMIT_SQL_LENGTH, SQL_TEXT_LIMIT),
    (sqlite3.SQLITE_LIMIT_COLUMN, 100),
    (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100),
    (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 20),
    (sqlite3.SQLITE_LIMIT_FUNCTION_ARG, 16),
    (sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH, 10_000),
    (sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 100),
    (sqlite3.SQLITE_LIMIT_TRIGGER_DEPTH, 1),
)
_ACTION_NAMES = {
    value: name
    for name, value in vars(sqlite3).items()
    if name.startswith("SQLITE_")
    and not name.startswith("SQLITE_LIMIT_")
    and name not in {"SQLITE_OK", "SQLITE_DENY", "SQLITE_IGNORE"}
    and isinstance(value, int)
}
_PARAMETER_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_SCOPE_REQUIRED = ("definition", "unit", "filters", "missing_data")
_SCOPE_OPTIONAL = ("denominator", "period", "timezone")


@dataclass(frozen=True)
class SnapshotInfo:
    derivation: str
    dataset_id: str
    corpus_revision: int
    rows: int
    bytes: int
    build_ms: int

    def to_dict(self) -> dict:
        return {
            "derivation": self.derivation,
            "dataset_id": self.dataset_id,
            "corpus_revision": self.corpus_revision,
            "rows": self.rows,
            "bytes": self.bytes,
            "build_ms": self.build_ms,
        }


@dataclass(frozen=True)
class SqlShape:
    parameters: frozenset[str]
    top_level_order_by: bool
    cte_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SqlOutcome:
    columns: list[str]
    rows: list[list]
    has_more: bool
    error: dict | None
    steps: int
    elapsed_ms: int


# --- snapshot ------------------------------------------------------------------------


def build_snapshot(
    dataset: Dataset,
    run: Run,
    destination: Path,
    *,
    max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES,
    max_seconds: float = DEFAULT_SNAPSHOT_MAX_SECONDS,
) -> SnapshotInfo:
    """Copy the run's pinned messages, and nothing else, into a new SQLite file."""
    started = budget_module.clock()
    deadline = started + max_seconds

    def past_deadline() -> int:
        return 1 if budget_module.clock() > deadline else 0

    writer = sqlite3.connect(destination, isolation_level=None)
    reader: sqlite3.Connection | None = None
    rows = 0
    try:
        writer.execute("PRAGMA journal_mode = OFF")
        writer.execute("PRAGMA synchronous = OFF")
        writer.execute("PRAGMA temp_store = MEMORY")
        page_size = writer.execute("PRAGMA page_size").fetchone()[0]
        # The empty schema needs four pages; the post-build size check covers the rest.
        writer.execute(f"PRAGMA max_page_count = {max(4, max_bytes // page_size)}")
        for statement in SNAPSHOT_DDL:
            writer.execute(statement)
        reader = dataset.connect_ro()
        if dataset.info(reader).dataset_id != run.dataset_id:
            raise ConfigError("sql.snapshot_dataset", "the run belongs to a different dataset")
        reader.set_progress_handler(past_deadline, 10_000)
        writer.set_progress_handler(past_deadline, 10_000)
        reader.execute("BEGIN")
        cursor = reader.execute(
            f"SELECT {MESSAGE_SELECT} FROM messages WHERE first_seen_revision <= ? "
            "ORDER BY source_namespace, conversation_id, ts_utc, source_order, id",
            (run.corpus_revision,),
        )
        insert = (
            f"INSERT INTO visible_messages ({', '.join(MESSAGE_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in MESSAGE_COLUMNS)})"
        )
        writer.execute("BEGIN")
        while batch := cursor.fetchmany(1000):
            writer.executemany(insert, batch)
            rows += len(batch)
            if past_deadline():
                raise BudgetExceeded("sql.snapshot_time", "building the SQL snapshot exceeded its time budget")
        writer.execute("COMMIT")
        reader.execute("COMMIT")
        if writer.execute("SELECT COUNT(*) FROM visible_messages").fetchone()[0] != rows:
            raise ConfigError("sql.snapshot_count", "the SQL snapshot row count does not match the copy")
    except sqlite3.OperationalError as exc:
        _remove_database(destination, writer, reader)
        code = getattr(exc, "sqlite_errorcode", None)
        if code == sqlite3.SQLITE_FULL:
            raise BudgetExceeded("sql.snapshot_size", "the pinned corpus exceeds the SQL snapshot size budget; nothing was sampled") from None
        if code == sqlite3.SQLITE_INTERRUPT:
            raise BudgetExceeded("sql.snapshot_time", "building the SQL snapshot exceeded its time budget") from None
        raise
    except BaseException:
        _remove_database(destination, writer, reader)
        raise
    writer.close()
    if reader is not None:
        reader.close()
    size = destination.stat().st_size
    if size > max_bytes:
        destination.unlink(missing_ok=True)
        raise BudgetExceeded("sql.snapshot_size", "the pinned corpus exceeds the SQL snapshot size budget; nothing was sampled")
    return SnapshotInfo(
        derivation=SNAPSHOT_DERIVATION,
        dataset_id=run.dataset_id,
        corpus_revision=run.corpus_revision,
        rows=rows,
        bytes=size,
        build_ms=int((budget_module.clock() - started) * 1000),
    )


def temporary_snapshot_path(run: Run, label: str) -> Path:
    return run.directory / f".{label}-{secrets.token_hex(8)}.sqlite3"


def _remove_database(path: Path, *connections: sqlite3.Connection | None) -> None:
    for connection in connections:
        if connection is not None:
            connection.close()
    path.unlink(missing_ok=True)


def open_user_sql_connection(snapshot: Path, cte_names: frozenset[str] = frozenset()) -> tuple[sqlite3.Connection, list[dict]]:
    """Reopen a finished snapshot read-only, locked down before any user SQL."""
    connection = sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True, isolation_level=None)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA trusted_schema = OFF")
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        if [row[0] for row in tables] != ["visible_messages"]:
            raise ConfigError("sql.snapshot_schema", "the SQL snapshot does not contain exactly one table")
        if hasattr(connection, "enable_load_extension"):
            connection.enable_load_extension(False)
        for limit, value in _HARD_LIMITS:
            connection.setlimit(limit, value)
        denials: list[dict] = []
        connection.set_authorizer(_authorizer(denials, cte_names))
    except BaseException:
        connection.close()
        raise
    return connection, denials


def _authorizer(denials: list[dict], cte_names: frozenset[str]):
    countable = {"visible_messages", *(name for name in cte_names if not name.startswith(("sqlite_", "pragma_")))}

    def authorize(action: int, first: str | None, second: str | None, database: str | None, source: str | None) -> int:
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        # No other schema can hold visible_messages: ATTACH is limited to 0 and CREATE is denied.
        if action == sqlite3.SQLITE_READ and first == "visible_messages" and database in ("main", None):
            return sqlite3.SQLITE_OK
        # A column-less read such as COUNT(*) reports an empty column, no database, and
        # for a CTE the CTE's own name; allow that only for this statement's CTE names.
        if action == sqlite3.SQLITE_READ and second == "" and database is None and (first or "").lower() in countable:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and isinstance(second, str) and second.lower() in ALLOWED_FUNCTIONS:
            return sqlite3.SQLITE_OK
        target = second if action == sqlite3.SQLITE_FUNCTION else first
        denials.append({"action": _ACTION_NAMES.get(action, str(action)), "object": target})
        return sqlite3.SQLITE_DENY

    return authorize


# --- statement checks ------------------------------------------------------------------


def inspect_sql(sql: object) -> SqlShape:
    """Lexical pre-checks: one SELECT/WITH/VALUES statement, named parameters only.

    These give clear errors early; the read-only connection, authorizer and
    budgets remain the enforcement boundary.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise InvalidInput("sql.empty", "the SQL file is empty")
    if len(sql) > SQL_TEXT_LIMIT or "\x00" in sql:
        raise InvalidInput("sql.too_long", f"SQL must be under {SQL_TEXT_LIMIT} characters without NUL")
    tokens = [token for token in _tokens(sql) if token[0] not in ("space", "comment")]
    first_kind, first_text = tokens[0] if tokens else ("", "")
    if first_kind != "word" or first_text.upper() not in ("SELECT", "WITH", "VALUES"):
        raise InvalidInput("sql.not_query", "only one SELECT, WITH or VALUES query is accepted")
    semicolons = [index for index, token in enumerate(tokens) if token == ("punct", ";")]
    if semicolons and semicolons != [len(tokens) - 1]:
        raise InvalidInput("sql.multiple_statements", "exactly one SQL statement is accepted")
    parameters: set[str] = set()
    depth = 0
    order_by = False
    cte_names = _cte_names(tokens)
    for index, (kind, text) in enumerate(tokens):
        if kind == "param":
            if text[0] != ":" or not _PARAMETER_NAME_RE.fullmatch(text[1:]):
                raise InvalidInput("sql.parameter_style", "use named :parameters bound from the parameter file")
            parameters.add(text[1:])
        elif (kind, text) == ("punct", "("):
            depth += 1
        elif (kind, text) == ("punct", ")"):
            depth -= 1
        elif (
            kind == "word"
            and depth == 0
            and text.upper() == "BY"
            and index > 0
            and tokens[index - 1][0] == "word"
            and tokens[index - 1][1].upper() == "ORDER"
        ):
            order_by = True
    return SqlShape(parameters=frozenset(parameters), top_level_order_by=order_by, cte_names=cte_names)


def _cte_names(tokens: list[tuple[str, str]]) -> frozenset[str]:
    """Names declared as ``name [(columns)] AS [[NOT] MATERIALIZED] (`` (a superset of CTE names)."""
    names: set[str] = set()
    for index, (kind, text) in enumerate(tokens):
        if kind not in ("word", "identifier"):
            continue
        position = index + 1
        if position < len(tokens) and tokens[position] == ("punct", "("):
            depth = 0
            while position < len(tokens):
                if tokens[position] == ("punct", "("):
                    depth += 1
                elif tokens[position] == ("punct", ")"):
                    depth -= 1
                    if depth == 0:
                        position += 1
                        break
                position += 1
        if position >= len(tokens) or tokens[position][0] != "word" or tokens[position][1].upper() != "AS":
            continue
        position += 1
        while position < len(tokens) and tokens[position][0] == "word" and tokens[position][1].upper() in ("NOT", "MATERIALIZED"):
            position += 1
        if position < len(tokens) and tokens[position] == ("punct", "("):
            names.add(_unquote(text).lower())
    return frozenset(names)


def _unquote(identifier: str) -> str:
    if identifier[:1] in ('"', "`") and len(identifier) >= 2:
        return identifier[1:-1].replace(identifier[0] * 2, identifier[0])
    if identifier[:1] == "[" and len(identifier) >= 2:
        return identifier[1:-1]
    return identifier


def _tokens(sql: str):
    index, length = 0, len(sql)
    while index < length:
        char = sql[index]
        if char.isspace():
            end = index + 1
            while end < length and sql[end].isspace():
                end += 1
            yield ("space", sql[index:end])
        elif sql.startswith("--", index):
            end = sql.find("\n", index)
            end = length if end < 0 else end
            yield ("comment", sql[index:end])
        elif sql.startswith("/*", index):
            close = sql.find("*/", index + 2)
            if close < 0:
                raise InvalidInput("sql.unterminated", "unterminated comment")
            end = close + 2
            yield ("comment", sql[index:end])
        elif char in "'\"`":
            end = index + 1
            while True:
                close = sql.find(char, end)
                if close < 0:
                    raise InvalidInput("sql.unterminated", "unterminated quoted text")
                if close + 1 < length and sql[close + 1] == char:
                    end = close + 2
                    continue
                end = close + 1
                break
            yield ("string" if char == "'" else "identifier", sql[index:end])
        elif char == "[":
            close = sql.find("]", index + 1)
            if close < 0:
                raise InvalidInput("sql.unterminated", "unterminated bracketed identifier")
            end = close + 1
            yield ("identifier", sql[index:end])
        elif char in ":@$?":
            end = index + 1
            while end < length and (sql[end].isalnum() or sql[end] == "_"):
                end += 1
            yield ("param", sql[index:end])
        elif char.isalpha() or char == "_" or ord(char) > 127:
            end = index + 1
            while end < length and (sql[end].isalnum() or sql[end] in "_$" or ord(sql[end]) > 127):
                end += 1
            yield ("word", sql[index:end])
        elif char.isdigit():
            end = index + 1
            while end < length and (sql[end].isalnum() or sql[end] == "."):
                end += 1
            yield ("number", sql[index:end])
        else:
            end = index + 1
            yield ("punct", char)
        index = end


def validate_parameters(raw: object, shape: SqlShape) -> dict:
    if not isinstance(raw, dict):
        raise InvalidInput("sql.parameters", "parameters must be a JSON object of named values")
    for name, value in raw.items():
        if not _PARAMETER_NAME_RE.fullmatch(name):
            raise InvalidInput("sql.parameters", "parameter names use letters, digits and '_'")
        if isinstance(value, bool) or not (value is None or isinstance(value, (str, int, float))):
            raise InvalidInput("sql.parameters", "parameter values must be strings, numbers or null")
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidInput("sql.parameters", "parameter values must be finite")
        if isinstance(value, int) and not -(2**63) <= value < 2**63:
            raise InvalidInput("sql.parameters", "integer parameters must fit in 64 bits")
    missing = sorted(shape.parameters - raw.keys())
    unused = sorted(raw.keys() - shape.parameters)
    if missing or unused:
        raise InvalidInput(
            "sql.parameters",
            "the parameter set must match the SQL's named parameters exactly",
            details={"missing": missing, "unused": unused},
        )
    return dict(sorted(raw.items()))


def validate_scope(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise InvalidInput("sql.scope", "the scope must be a JSON object")
    unknown = sorted(raw.keys() - set(_SCOPE_REQUIRED) - set(_SCOPE_OPTIONAL))
    missing = [name for name in _SCOPE_REQUIRED if name not in raw]
    if unknown or missing:
        raise InvalidInput("sql.scope", "the scope has missing or unknown fields", details={"missing": missing, "unknown": unknown})
    for name, value in raw.items():
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise InvalidInput("sql.scope", "scope fields must be nonempty text under 2000 characters", details={"field": name})
    if "timezone" in raw:
        load_zone(raw["timezone"])
    return {name: raw[name] for name in (*_SCOPE_REQUIRED, *_SCOPE_OPTIONAL) if name in raw}


def utc_parameters(assignments: list[str], zone_name: str | None) -> tuple[dict, list[dict]]:
    """Convert ``name=LOCAL`` wall times in an explicit IANA zone to canonical UTC once."""
    if not assignments:
        return {}, []
    if zone_name is None:
        raise InvalidInput("time.zone_required", "--tz with an explicit IANA zone is required with --utc-param")
    zone = load_zone(zone_name)
    values: dict = {}
    records: list[dict] = []
    for assignment in assignments:
        name, sep, local = assignment.partition("=")
        if not sep or not _PARAMETER_NAME_RE.fullmatch(name) or name in values:
            raise InvalidInput("sql.utc_parameter", "use --utc-param NAME=YYYY-MM-DD[THH:MM[:SS]] once per name")
        bound = local_bound(local, zone, label="utc-param")
        values[name] = bound.utc
        records.append({"name": name, "local": local, "zone": zone_name, "utc": bound.utc, "kind": bound.kind})
    return values, records


# --- execution -----------------------------------------------------------------------


def execute_user_sql(
    snapshot: Path,
    sql: str,
    parameters: dict,
    shape: SqlShape,
    *,
    max_rows: int,
    max_steps: int,
    deadline_ms: int,
) -> SqlOutcome:
    connection, denials = open_user_sql_connection(snapshot, shape.cte_names)
    budget = Budget(deadline_ms=deadline_ms, max_steps=max_steps)
    budget.install(connection)
    columns: list[str] = []
    try:
        cursor = connection.execute(sql, parameters)
        columns = [item[0] for item in cursor.description or ()]
        raw_rows = cursor.fetchmany(max_rows + 1)
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return SqlOutcome(columns, [], False, _sql_error(exc, budget, denials), budget.steps, budget.elapsed_ms())
    finally:
        connection.close()
    error = None
    if len(set(columns)) != len(columns):
        error = {"code": "sql.duplicate_columns", "message": "result column names must be unique"}
    elif len(raw_rows) > 1 and not shape.top_level_order_by:
        error = {"code": "sql.unordered_multirow", "message": "a multirow result needs a top-level ORDER BY"}
    if error is not None:
        return SqlOutcome(columns, [], False, error, budget.steps, budget.elapsed_ms())
    rows = [[encode_cell(value) for value in row] for row in raw_rows[:max_rows]]
    return SqlOutcome(columns, rows, len(raw_rows) > max_rows, None, budget.steps, budget.elapsed_ms())


def encode_cell(value: object) -> object:
    """JSON form of one SQLite value; JSON keeps integer and real distinct."""
    if value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"type": "real", "repr": repr(value)}
    if isinstance(value, bytes):
        return {"type": "blob", "length": len(value), "sha256": sha256_bytes(value)}
    raise ConfigError("sql.cell_type", "unexpected SQLite value type")


def _sql_error(exc: Exception, budget: Budget, denials: list[dict]) -> dict:
    if budget.reason is not None:
        return {"code": f"budget.{budget.reason}", "message": "the statement exceeded its execution budget"}
    message = str(exc)
    if denials or "not authorized" in message:
        return {
            "code": "sql.not_authorized",
            "message": "the statement used a table, function or operation outside the allowlist",
            "denied": denials[:1],
        }
    if isinstance(exc, (sqlite3.ProgrammingError, sqlite3.Warning)) and "one statement" in message:
        return {"code": "sql.multiple_statements", "message": "exactly one SQL statement is accepted"}
    if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_TOOBIG or "too big" in message:
        return {"code": "sql.too_big", "message": "a value exceeded the SQL length limit"}
    return {"code": "sql.error", "message": message[:300]}


def remove_snapshot(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
