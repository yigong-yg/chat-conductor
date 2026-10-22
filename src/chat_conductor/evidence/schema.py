"""The packaged version 1 dataset schema and checks that refuse anything else."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from .errors import ConfigError
from .paths import DATASET_ID_RE

SCHEMA_VERSION = 1
# Stored in the SQLite header ("CEV1") to mark a Chat Evidence dataset database.
APPLICATION_ID = 0x43455631
DOMAIN_TABLES = frozenset({"dataset", "import_batches", "messages", "message_origins"})
TURN_INDEX_TABLES = frozenset({"turns", "turn_fts", "meta"})


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    schema_version: int
    created_at: str


def schema_sql() -> str:
    return resources.files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")


def iter_statements(script: str) -> Iterator[str]:
    """Split a script into complete statements (trigger bodies included)."""
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            yield buffer.strip()
            buffer = ""
    if any(line.strip() and not line.strip().startswith("--") for line in buffer.splitlines()):
        raise ConfigError("schema.incomplete_statement", "the packaged schema ends with an incomplete statement")


def create_schema(connection: sqlite3.Connection, *, dataset_id: str, created_at: str) -> None:
    """Create all objects and the dataset row; the caller holds BEGIN IMMEDIATE."""
    for statement in iter_statements(schema_sql()):
        connection.execute(statement)
    connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.execute(
        "INSERT INTO dataset (singleton, dataset_id, schema_version, created_at) VALUES (1, ?, ?, ?)",
        (dataset_id, SCHEMA_VERSION, created_at),
    )


def _schema_objects(connection: sqlite3.Connection) -> tuple[tuple, ...]:
    return tuple(sorted(
        connection.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall(),
        key=lambda row: (row[0], row[1]),
    ))


@lru_cache(maxsize=1)
def expected_schema_objects() -> tuple[tuple, ...]:
    connection = sqlite3.connect(":memory:")
    try:
        for statement in iter_statements(schema_sql()):
            connection.execute(statement)
        return _schema_objects(connection)
    finally:
        connection.close()


def check_schema(connection: sqlite3.Connection) -> DatasetInfo:
    """Accept only an unmodified version 1 evidence schema with its identity row."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if tables & TURN_INDEX_TABLES:
        raise ConfigError(
            "dataset.turn_index",
            "this database is a chat-conductor turn index; evidence uses a separate dataset database",
        )
    if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
        raise ConfigError("dataset.not_evidence", "this database is not a Chat Evidence dataset")
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        raise ConfigError(
            "dataset.unsupported_version",
            "unsupported evidence schema version",
            details={"schema_version": version, "supported": SCHEMA_VERSION},
        )
    if _schema_objects(connection) != expected_schema_objects():
        raise ConfigError("dataset.schema_mismatch", "the dataset schema differs from the supported version 1 schema")
    rows = connection.execute("SELECT singleton, dataset_id, schema_version, created_at FROM dataset").fetchall()
    if len(rows) != 1 or rows[0][2] != SCHEMA_VERSION or not DATASET_ID_RE.fullmatch(rows[0][1]):
        raise ConfigError("dataset.identity_invalid", "the dataset identity row is missing or invalid")
    return DatasetInfo(dataset_id=rows[0][1], schema_version=rows[0][2], created_at=rows[0][3])
