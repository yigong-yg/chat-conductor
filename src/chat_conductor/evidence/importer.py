"""Transactional import: one preserved export becomes one new corpus revision.

The export is preserved first, then parsed from the preserved copy. Every
record is validated; any invalid record, changed payload for a known native
identity, or conflicting duplicate inside the batch quarantines the whole
batch, so no message, origin or revision from it is committed. Accepted
batches are written under ``BEGIN IMMEDIATE`` and their exact totals are
checked before ``COMMIT``. Reimporting the same export/adapter/version/profile
tuple is a no-op that allocates no revision.
"""

from __future__ import annotations

import secrets
import sqlite3
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .errors import ConfigError, InvalidInput
from .fixture import SyntheticFixtureAdapter
from .fsutil import ensure_private_dir, write_new_file
from .jsonutil import canonical_bytes
from .profile import Profile, load_profile
from .rawstore import preserve_export
from .records import RecordProblem, SourceRecord, validate_record
from .store import Dataset, latest_revision
from .timeutil import utc_now

IMPORT_REPORT_FORMAT = "chat-evidence-import-report"
IMPORT_REPORT_VERSION = 1
MAX_REPORTED_DIAGNOSTICS = 1000
ADAPTERS = {adapter.name: adapter for adapter in (SyntheticFixtureAdapter(),)}


class ImportRejected(InvalidInput):
    """The incoming batch was quarantined; nothing from it was committed."""

    def __init__(self, summary: dict) -> None:
        super().__init__("ingest.rejected", "the incoming batch was rejected; nothing was committed", details=summary)


def get_adapter(name: str) -> SyntheticFixtureAdapter:
    adapter = ADAPTERS.get(name)
    if adapter is None:
        raise ConfigError(
            "ingest.adapter_unavailable",
            "no approved adapter with this name; a real export adapter requires an inspected, "
            "redacted export contract before it can be implemented",
            details={"available": sorted(ADAPTERS)},
        )
    return adapter


def ingest(dataset: Dataset, export_path: Path, *, adapter_name: str, profile_path: Path) -> dict:
    adapter = get_adapter(adapter_name)
    profile = load_profile(profile_path)
    if (profile.adapter, profile.adapter_version) != (adapter.name, adapter.version):
        raise ConfigError("profile.adapter_mismatch", "the profile is bound to a different adapter or adapter version")
    connection = dataset.connect_rw()
    try:
        info = dataset.info(connection)
        preserved = preserve_export(dataset, export_path)
        _store_profile(dataset, profile)
        base = {
            "dataset_id": info.dataset_id,
            "adapter": adapter.name,
            "adapter_version": adapter.version,
            "profile_sha256": profile.sha256,
            "export_sha256": preserved.export_sha256,
            "raw_manifest_path": preserved.manifest_relpath,
        }
        existing = _existing_revision(connection, base)
        if existing is not None:
            return _finish(dataset, base, "noop", revision=existing)

        parsed = adapter.parse(preserved, profile)
        problems = list(parsed.problems)
        valid: list[SourceRecord] = []
        for record in parsed.records:
            found = validate_record(record, preserved.paths)
            problems.extend(found)
            if not found:
                valid.append(record)
        problems.extend(_duplicate_locators(parsed.records))
        groups, conflicts = _group_by_identity(info.dataset_id, valid)
        counts = {
            "records": len(parsed.records),
            "unique_messages": len(groups),
            "overlap_in_batch": len(valid) - len(groups),
            "invalid": len(problems),
            "conflicting": len(conflicts),
        }
        if problems or conflicts:
            _reject(dataset, base, counts, problems + conflicts)
        return _commit(connection, dataset, base, info.dataset_id, valid, groups, counts)
    finally:
        connection.close()


def _commit(
    connection: sqlite3.Connection,
    dataset: Dataset,
    base: dict,
    dataset_id: str,
    records: list[SourceRecord],
    groups: dict[str, tuple[SourceRecord, str]],
    counts: dict,
) -> dict:
    connection.execute("BEGIN IMMEDIATE")
    try:
        existing = _existing_revision(connection, base)
        if existing is not None:
            connection.execute("ROLLBACK")
            return _finish(dataset, base, "noop", revision=existing)
        _require_complete_latest(connection)
        new_ids, overlap_existing, conflicts = _classify(connection, groups)
        if conflicts:
            connection.execute("ROLLBACK")
            _reject(dataset, base, {**counts, "conflicting": len(conflicts)}, conflicts)
        revision = latest_revision(connection) + 1
        import_id = f"imp_{secrets.token_hex(8)}"
        imported_at = utc_now()
        connection.execute(
            """
            INSERT INTO import_batches (
                revision, import_id, export_sha256, adapter, adapter_version, profile_sha256,
                raw_manifest_path, imported_at, record_count, appended_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision,
                import_id,
                base["export_sha256"],
                base["adapter"],
                base["adapter_version"],
                base["profile_sha256"],
                base["raw_manifest_path"],
                imported_at,
                len(records),
                len(new_ids),
            ),
        )
        _insert_messages(connection, revision, [(message_id, *groups[message_id]) for message_id in new_ids])
        _insert_origins(connection, revision, dataset_id, records)
        _check_totals(connection, revision, appended=len(new_ids), record_count=len(records))
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    counts = {**counts, "new": len(new_ids), "overlap_existing": overlap_existing}
    return _finish(
        dataset,
        {**base, "import_id": import_id, "imported_at": imported_at},
        "committed",
        revision=revision,
        counts=counts,
    )


def _classify(
    connection: sqlite3.Connection,
    groups: dict[str, tuple[SourceRecord, str]],
) -> tuple[list[str], int, list[RecordProblem]]:
    new_ids: list[str] = []
    overlap = 0
    conflicts: list[RecordProblem] = []
    for message_id, (record, payload) in groups.items():
        row = connection.execute(
            "SELECT source_namespace, conversation_id, native_message_id, payload_sha256 FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            clash = connection.execute(
                "SELECT 1 FROM messages WHERE source_namespace = ? AND conversation_id = ? AND native_message_id = ?",
                record.native_identity(),
            ).fetchone()
            if clash is not None:
                raise ConfigError("ingest.identity_corrupt", "a stored native identity is bound to a different message id")
            new_ids.append(message_id)
        elif tuple(row[:3]) != record.native_identity():
            raise ConfigError("ingest.id_collision", "message id digest collision between different native identities")
        elif row[3] != payload:
            conflicts.append(RecordProblem("conflict.changed_payload", record.raw_locator, None, record.native_message_id))
        else:
            overlap += 1
    return new_ids, overlap, conflicts


def _insert_messages(
    connection: sqlite3.Connection,
    revision: int,
    rows: list[tuple[str, SourceRecord, str]],
) -> None:
    rows.sort(key=lambda item: (
        item[1].source_namespace, item[1].conversation_id, item[1].ts_utc, item[1].source_order, item[0],
    ))
    connection.executemany(
        """
        INSERT INTO messages (
            id, source_namespace, conversation_id, native_message_id, sender_id, ts_utc,
            ts_original, source_timezone, source_order, order_basis, kind, text_origin,
            text, payload_sha256, first_seen_revision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                message_id,
                record.source_namespace,
                record.conversation_id,
                record.native_message_id,
                record.sender_id,
                record.ts_utc,
                record.ts_original,
                record.source_timezone,
                record.source_order,
                record.order_basis,
                record.kind,
                record.text_origin,
                record.text,
                payload,
                revision,
            )
            for message_id, record, payload in rows
        ),
    )


def _insert_origins(connection: sqlite3.Connection, revision: int, dataset_id: str, records: list[SourceRecord]) -> None:
    connection.executemany(
        "INSERT INTO message_origins (import_revision, raw_locator, message_id) VALUES (?, ?, ?)",
        ((revision, record.raw_locator, record.message_id(dataset_id)) for record in records),
    )


def _check_totals(connection: sqlite3.Connection, revision: int, *, appended: int, record_count: int) -> None:
    messages = connection.execute("SELECT COUNT(*) FROM messages WHERE first_seen_revision = ?", (revision,)).fetchone()[0]
    origins = connection.execute("SELECT COUNT(*) FROM message_origins WHERE import_revision = ?", (revision,)).fetchone()[0]
    if messages != appended or origins != record_count:
        raise ConfigError("ingest.count_mismatch", "batch totals differ from the declared counts; rolled back")


def _require_complete_latest(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT revision, record_count, appended_count FROM import_batches ORDER BY revision DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return
    revision, record_count, appended_count = row
    messages = connection.execute("SELECT COUNT(*) FROM messages WHERE first_seen_revision = ?", (revision,)).fetchone()[0]
    origins = connection.execute("SELECT COUNT(*) FROM message_origins WHERE import_revision = ?", (revision,)).fetchone()[0]
    if messages != appended_count or origins != record_count:
        raise ConfigError("dataset.incomplete_batch", "the latest import batch is incomplete; restore the dataset from a backup")


def _group_by_identity(
    dataset_id: str,
    records: list[SourceRecord],
) -> tuple[dict[str, tuple[SourceRecord, str]], list[RecordProblem]]:
    groups: dict[str, tuple[SourceRecord, str]] = {}
    conflicts: list[RecordProblem] = []
    for record in records:
        message_id = record.message_id(dataset_id)
        payload = record.payload_sha256()
        first = groups.get(message_id)
        if first is None:
            groups[message_id] = (record, payload)
        elif first[0].native_identity() != record.native_identity():
            raise ConfigError("ingest.id_collision", "message id digest collision between different native identities")
        elif first[1] != payload:
            conflicts.append(RecordProblem(
                "conflict.duplicate_changed_payload", record.raw_locator, None, record.native_message_id,
            ))
    return groups, conflicts


def _duplicate_locators(records: tuple[SourceRecord, ...]) -> list[RecordProblem]:
    seen: set[str] = set()
    problems: list[RecordProblem] = []
    for record in records:
        if record.raw_locator in seen:
            problems.append(RecordProblem("record.duplicate_locator", record.raw_locator))
        seen.add(record.raw_locator)
    return problems


def _existing_revision(connection: sqlite3.Connection, base: dict) -> int | None:
    row = connection.execute(
        """
        SELECT revision FROM import_batches
        WHERE export_sha256 = ? AND adapter = ? AND adapter_version = ? AND profile_sha256 = ?
        """,
        (base["export_sha256"], base["adapter"], base["adapter_version"], base["profile_sha256"]),
    ).fetchone()
    return None if row is None else int(row[0])


def _store_profile(dataset: Dataset, profile: Profile) -> None:
    ensure_private_dir(dataset.profiles_dir)
    path = dataset.profiles_dir / f"{profile.sha256}.json"
    try:
        write_new_file(path, profile.canonical)
    except FileExistsError:
        if path.read_bytes() != profile.canonical:
            raise ConfigError("profile.stored_mismatch", "a stored profile copy no longer matches its digest") from None


def _reject(dataset: Dataset, base: dict, counts: dict, diagnostics: list[RecordProblem]) -> None:
    summary = _finish(dataset, base, "rejected", revision=None, counts=counts, diagnostics=diagnostics)
    raise ImportRejected(summary)


def _finish(
    dataset: Dataset,
    base: dict,
    status: str,
    *,
    revision: int | None,
    counts: dict | None = None,
    diagnostics: Sequence[RecordProblem] = (),
) -> dict:
    created_at = utc_now()
    report = {
        "report_format": IMPORT_REPORT_FORMAT,
        "report_version": IMPORT_REPORT_VERSION,
        "status": status,
        "revision": revision,
        "created_at": created_at,
        **base,
        "counts": counts or {},
        "problem_codes": dict(sorted(Counter(item.code for item in diagnostics).items())),
        "diagnostics_total": len(diagnostics),
        "diagnostics": [item.to_dict() for item in diagnostics[:MAX_REPORTED_DIAGNOSTICS]],
    }
    ensure_private_dir(dataset.import_reports_dir)
    stamp = created_at.replace("-", "").replace(":", "").replace(".", "")
    path = dataset.import_reports_dir / f"{stamp}-{status}-{secrets.token_hex(4)}.json"
    write_new_file(path, canonical_bytes(report) + b"\n")
    summary = {key: value for key, value in report.items() if key != "diagnostics"}
    summary["report"] = str(path)
    return summary
