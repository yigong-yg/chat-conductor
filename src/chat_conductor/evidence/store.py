"""Dataset directory layout and connections to the canonical evidence database.

Each dataset is one explicitly selected directory with its own database; this
module never opens, initializes or unions the turn index used by the existing
CLI. Query paths use read-only (``mode=ro`` + ``query_only``) connections.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError
from .fsutil import PRIVATE_FILE_MODE, ensure_private_dir, restrict_permissions
from .paths import DATASET_ID_RE, resolve_dataset_dir
from .schema import DatasetInfo, check_schema, create_schema
from .timeutil import utc_now

DATABASE_FILENAME = "dataset.sqlite3"
BUSY_TIMEOUT_SECONDS = 30.0
MESSAGE_COLUMNS = (
    "id",
    "source_namespace",
    "conversation_id",
    "native_message_id",
    "sender_id",
    "ts_utc",
    "ts_original",
    "source_timezone",
    "source_order",
    "order_basis",
    "kind",
    "text_origin",
    "text",
    "payload_sha256",
    "first_seen_revision",
)
MESSAGE_SELECT = ", ".join(MESSAGE_COLUMNS)


@dataclass(frozen=True)
class Dataset:
    root: Path
    expected_id: str | None = None

    @property
    def db_path(self) -> Path:
        return self.root / DATABASE_FILENAME

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def import_reports_dir(self) -> Path:
        return self.root / "import-reports"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def require_initialized(self) -> None:
        if not self.db_path.is_file():
            raise ConfigError("dataset.not_initialized", "no evidence dataset in this directory; run init first")

    def connect_rw(self) -> sqlite3.Connection:
        self.require_initialized()
        connection = sqlite3.connect(self.db_path, isolation_level=None, timeout=BUSY_TIMEOUT_SECONDS)
        _configure(connection)
        return connection

    def connect_ro(self) -> sqlite3.Connection:
        self.require_initialized()
        try:
            connection = sqlite3.connect(
                self.db_path.as_uri() + "?mode=ro",
                uri=True,
                isolation_level=None,
                timeout=BUSY_TIMEOUT_SECONDS,
            )
            _configure(connection)
            connection.execute("PRAGMA query_only = ON")
        except sqlite3.OperationalError as exc:
            raise ConfigError("dataset.open_failed", "cannot open the dataset database read-only") from exc
        _check_like_semantics(connection)
        return connection

    def info(self, connection: sqlite3.Connection) -> DatasetInfo:
        info = check_schema(connection)
        if self.expected_id is not None and info.dataset_id != self.expected_id:
            raise ConfigError("dataset.id_mismatch", "the selected directory holds a different dataset id")
        return info


def locate_dataset(dataset: str | None, dataset_id: str | None) -> Dataset:
    root, expected_id = resolve_dataset_dir(dataset, dataset_id)
    return Dataset(root=root, expected_id=expected_id)


def init_dataset(dataset: Dataset, *, requested_id: str | None = None) -> DatasetInfo:
    root = dataset.root
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ConfigError("dataset.not_empty", "init requires a new or empty directory")
    if dataset.expected_id and requested_id and requested_id != dataset.expected_id:
        raise ConfigError("dataset.id_mismatch", "--id differs from the selected --dataset-id")
    dataset_id = dataset.expected_id or requested_id or f"ds_{secrets.token_hex(8)}"
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise ConfigError(
            "dataset.invalid_id",
            "dataset ids use 1-64 letters, digits, '_' or '-' and start with a letter or digit",
        )
    ensure_private_dir(root)
    db_path = dataset.db_path
    connection = sqlite3.connect(db_path, isolation_level=None, timeout=BUSY_TIMEOUT_SECONDS)
    try:
        if connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
            raise ConfigError("dataset.wal_unavailable", "SQLite WAL journal mode is unavailable here")
        _configure(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            create_schema(connection, dataset_id=dataset_id, created_at=utc_now())
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        info = check_schema(connection)
    except BaseException:
        connection.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
        raise
    connection.close()
    restrict_permissions(db_path, PRIVATE_FILE_MODE)
    for directory in (dataset.raw_dir, dataset.profiles_dir, dataset.import_reports_dir, dataset.runs_dir):
        ensure_private_dir(directory)
    return info


def latest_revision(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT COALESCE(MAX(revision), 0) FROM import_batches").fetchone()[0])


def message_dict(row: tuple) -> dict:
    return dict(zip(MESSAGE_COLUMNS, row))


def _configure(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA recursive_triggers = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    # Keep sort/temp data in memory: SQLite temp files would land outside the dataset.
    connection.execute("PRAGMA temp_store = MEMORY")


def _check_like_semantics(connection: sqlite3.Connection) -> None:
    # Search documents LIKE as ASCII-only case folding; refuse a build that differs (e.g. ICU).
    if tuple(connection.execute("SELECT 'A' LIKE 'a', 'Ä' LIKE 'ä'").fetchone()) != (1, 0):
        raise ConfigError("sqlite.like_semantics", "this SQLite build does not fold case for ASCII only in LIKE")
