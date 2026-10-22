"""Analysis runs pinned to one dataset and one committed corpus revision.

``run.json`` is written once. Every query, receipt and verification of the
run uses its ``corpus_revision`` boundary, so later or concurrent imports
cannot change what the run sees. The ledger digest of import batches up to
that revision detects a dataset whose history differs from the run's.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import TOOL_NAME, TOOL_VERSION
from .errors import ConfigError, InvalidInput
from .fsutil import ensure_private_dir, write_new_file
from .jsonutil import canonical_bytes, loads_strict, sha256_json
from .paths import RUN_ID_RE, contained_path
from .store import Dataset, latest_revision
from .timeutil import utc_now

RUN_FORMAT = "chat-evidence-run"
RUN_VERSION = 1
_RUN_KEYS = frozenset({
    "run_format",
    "run_version",
    "run_id",
    "dataset_id",
    "corpus_revision",
    "question",
    "created_at",
    "tool",
    "tool_version",
    "ledger_sha256",
})


@dataclass(frozen=True)
class Run:
    run_id: str
    dataset_id: str
    corpus_revision: int
    question: str
    created_at: str
    tool_version: str
    ledger_sha256: str
    directory: Path

    @property
    def receipts_dir(self) -> Path:
        return self.directory / "receipts"

    @property
    def run_path(self) -> Path:
        return self.directory / "run.json"

    @property
    def analysis_path(self) -> Path:
        return self.directory / "analysis.json"

    @property
    def verification_path(self) -> Path:
        return self.directory / "verification.json"

    @property
    def report_path(self) -> Path:
        return self.directory / "report.md"

    @property
    def snapshot_path(self) -> Path:
        return self.directory / "sql-snapshot.sqlite3"

    def to_dict(self) -> dict:
        return {
            "run_format": RUN_FORMAT,
            "run_version": RUN_VERSION,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "corpus_revision": self.corpus_revision,
            "question": self.question,
            "created_at": self.created_at,
            "tool": TOOL_NAME,
            "tool_version": self.tool_version,
            "ledger_sha256": self.ledger_sha256,
        }


def ledger_sha256(connection: sqlite3.Connection, revision: int) -> str:
    rows = connection.execute(
        """
        SELECT revision, import_id, export_sha256, adapter, adapter_version, profile_sha256,
               raw_manifest_path, imported_at, record_count, appended_count
        FROM import_batches WHERE revision <= ? ORDER BY revision
        """,
        (revision,),
    ).fetchall()
    return sha256_json([list(row) for row in rows])


def start_run(dataset: Dataset, question: str, *, revision: int | None = None) -> Run:
    if not isinstance(question, str) or not question.strip():
        raise InvalidInput("run.question_empty", "a run needs a nonempty question")
    try:
        question.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidInput("run.question_encoding", "the question is not valid Unicode text") from None
    connection = dataset.connect_ro()
    try:
        info = dataset.info(connection)
        latest = latest_revision(connection)
        pinned = latest if revision is None else revision
        if pinned < 0 or pinned > latest:
            raise InvalidInput("run.revision_unavailable", "the requested revision is not committed", details={"latest": latest})
        ledger = ledger_sha256(connection, pinned)
    finally:
        connection.close()
    created_at = utc_now()
    stamp = created_at[:19].replace("-", "").replace(":", "")
    run_id = f"run_{stamp}Z_{secrets.token_hex(4)}"
    ensure_private_dir(dataset.runs_dir)
    directory = dataset.runs_dir / run_id
    directory.mkdir(mode=0o700)
    ensure_private_dir(directory / "receipts")
    run = Run(
        run_id=run_id,
        dataset_id=info.dataset_id,
        corpus_revision=pinned,
        question=question,
        created_at=created_at,
        tool_version=TOOL_VERSION,
        ledger_sha256=ledger,
        directory=directory,
    )
    write_new_file(run.run_path, canonical_bytes(run.to_dict()) + b"\n")
    return run


def load_run(dataset: Dataset, connection: sqlite3.Connection, run_id: str) -> Run:
    """Load a run and confirm it belongs to this dataset's current import history."""
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise InvalidInput("run.invalid_id", "run ids look like run_<letters, digits, '_' or '-'>")
    info = dataset.info(connection)
    directory = contained_path(dataset.runs_dir, run_id)
    try:
        data = (directory / "run.json").read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        raise InvalidInput("run.unknown", "no such run in this dataset") from None
    raw = loads_strict(data, what="run")
    if not isinstance(raw, dict) or raw.keys() != _RUN_KEYS:
        raise ConfigError("run.malformed", "run.json has missing or unknown fields")
    revision = raw["corpus_revision"]
    if (
        raw["run_format"] != RUN_FORMAT
        or raw["run_version"] != RUN_VERSION
        or raw["run_id"] != run_id
        or type(revision) is not int
        or revision < 0
        or not isinstance(raw["question"], str)
    ):
        raise ConfigError("run.malformed", "run.json is not a valid run manifest")
    if raw["dataset_id"] != info.dataset_id:
        raise InvalidInput("run.other_dataset", "the run belongs to a different dataset")
    if revision > latest_revision(connection) or ledger_sha256(connection, revision) != raw["ledger_sha256"]:
        raise ConfigError("run.ledger_mismatch", "the dataset's import history differs from the one this run pinned")
    return Run(
        run_id=run_id,
        dataset_id=raw["dataset_id"],
        corpus_revision=revision,
        question=raw["question"],
        created_at=raw["created_at"],
        tool_version=raw["tool_version"],
        ledger_sha256=raw["ledger_sha256"],
        directory=directory,
    )
