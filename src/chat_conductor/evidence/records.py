"""Typed, source-neutral records that adapters hand to the importer.

An adapter turns one preserved export into ``SourceRecord`` values plus
``RecordProblem`` diagnostics. Any problem rejects the whole batch: records
are never dropped silently. The importer validates every record again here,
independently of the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from . import identity
from .errors import EvidenceError
from .timeutil import is_canonical_utc, load_zone

KINDS = ("text", "voice", "image", "video", "file", "transfer", "call", "recall", "system", "other")
TEXT_ORIGINS = ("export_text", "export_transcript", "placeholder")
ORDER_BASES = ("native", "timestamp_only")
TRANSCRIPT_KINDS = frozenset({"voice", "video", "call"})
_NONEMPTY_FIELDS = (
    "source_namespace",
    "conversation_id",
    "native_message_id",
    "sender_id",
    "ts_original",
    "source_timezone",
    "text",
    "raw_locator",
)


@dataclass(frozen=True)
class SourceRecord:
    """One observed source occurrence.

    ``text`` is the decoded source text exactly as exported, or an adapter
    placeholder when ``text_origin == "placeholder"``; it is never normalized.
    ``raw_locator`` is ``<preserved file path>#<stable position>``, e.g. a JSON
    pointer or a line number inside the preserved raw copy.
    """

    source_namespace: str
    conversation_id: str
    native_message_id: str
    sender_id: str
    ts_utc: str
    ts_original: str
    source_timezone: str
    source_order: str
    order_basis: str
    kind: str
    text_origin: str
    text: str
    raw_locator: str

    def native_identity(self) -> tuple[str, str, str]:
        return (self.source_namespace, self.conversation_id, self.native_message_id)

    def message_id(self, dataset_id: str) -> str:
        return identity.message_id(dataset_id, *self.native_identity())

    def payload_sha256(self) -> str:
        return identity.payload_sha256(
            sender_id=self.sender_id,
            ts_original=self.ts_original,
            ts_utc=self.ts_utc,
            source_timezone=self.source_timezone,
            source_order=self.source_order,
            order_basis=self.order_basis,
            kind=self.kind,
            text_origin=self.text_origin,
            text=self.text,
        )


@dataclass(frozen=True)
class RecordProblem:
    """A private diagnostic; written only to the dataset's import reports."""

    code: str
    locator: str
    field: str | None = None
    native_message_id: str | None = None

    def to_dict(self) -> dict:
        payload: dict = {"code": self.code, "locator": self.locator}
        if self.field is not None:
            payload["field"] = self.field
        if self.native_message_id is not None:
            payload["native_message_id"] = self.native_message_id
        return payload


@dataclass(frozen=True)
class ParsedBatch:
    records: tuple[SourceRecord, ...]
    problems: tuple[RecordProblem, ...]


def validate_record(record: SourceRecord, manifest_paths: frozenset[str]) -> list[RecordProblem]:
    locator = record.raw_locator if isinstance(record.raw_locator, str) else "?"
    native = record.native_message_id if isinstance(record.native_message_id, str) else None
    problems: list[RecordProblem] = []

    def problem(code: str, field: str) -> None:
        problems.append(RecordProblem(code, locator, field, native))

    for item in fields(record):
        value = getattr(record, item.name)
        if not isinstance(value, str):
            problem("record.not_string", item.name)
            continue
        if "\x00" in value:
            problem("record.nul_character", item.name)
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            problem("record.lone_surrogate", item.name)
    if problems:
        return problems

    for name in _NONEMPTY_FIELDS:
        if not getattr(record, name):
            problem("record.empty", name)
    platform, _, account = record.source_namespace.partition(":")
    if not platform or not account:
        problem("record.namespace_format", "source_namespace")
    if not is_canonical_utc(record.ts_utc):
        problem("record.ts_utc_not_canonical", "ts_utc")
    try:
        load_zone(record.source_timezone)
    except EvidenceError:
        problem("record.zone_invalid", "source_timezone")
    if record.order_basis not in ORDER_BASES:
        problem("record.order_basis_unknown", "order_basis")
    elif record.order_basis == "native" and not record.source_order:
        problem("record.native_order_missing", "source_order")
    elif record.order_basis == "timestamp_only" and record.source_order:
        problem("record.order_without_native_basis", "source_order")
    if record.kind not in KINDS:
        problem("record.kind_unknown", "kind")
    if record.text_origin not in TEXT_ORIGINS:
        problem("record.text_origin_unknown", "text_origin")
    if record.kind == "text" and record.text_origin != "export_text":
        problem("record.text_kind_requires_export_text", "text_origin")
    if record.text_origin == "export_transcript" and record.kind not in TRANSCRIPT_KINDS:
        problem("record.transcript_kind", "kind")
    file_part, hash_sign, position = record.raw_locator.partition("#")
    if not hash_sign or not position or file_part not in manifest_paths:
        problem("record.locator_unresolved", "raw_locator")
    return problems
