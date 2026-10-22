"""``synthetic-fixture`` v1: an explicitly synthetic test format, not an export adapter.

No real export adapter exists. One requires an inspected, redacted export
contract (field definitions, exporter version, native account/conversation/
message identity, sender mapping, timestamp precision and zone, ordering,
media representation, overlap/edit/recall behaviour). This adapter accepts only
documents that declare themselves synthetic, so it cannot ingest a real export
by accident::

    {"format": "chat-evidence-synthetic-fixture", "format_version": 1, "synthetic": true,
     "conversations": [{"conversation_id": "conv-a", "messages": [
        {"native_id": "n1", "sender": "ana", "ts": "2026-03-01T09:15:00",
         "order": "000001", "kind": "text", "text": "..."},
        {"native_id": "n2", "sender": "ben", "ts": "2026-03-01T09:16:00+08:00",
         "order": "000002", "kind": "image", "placeholder": "[image]"},
        {"native_id": "n3", "sender": "ben", "ts": "2026-03-01T09:17:00",
         "order": "000003", "kind": "voice", "transcript": "..."}]}]}

``ts`` is ISO 8601 with seconds; without an offset it is a wall time in the
profile's declared zone, and with one it must match that zone. ``order`` is
present exactly when the profile's ``order_policy`` is ``native``. Exactly one
of ``text`` / ``transcript`` / ``placeholder`` is given. Raw locators are JSON
pointers into the preserved file.
"""

from __future__ import annotations

from .errors import InvalidInput
from .jsonutil import loads_strict
from .profile import Profile
from .rawstore import PreservedExport
from .records import KINDS, ParsedBatch, RecordProblem, SourceRecord
from .timeutil import resolve_source_timestamp

ADAPTER_NAME = "synthetic-fixture"
ADAPTER_VERSION = "1"
FIXTURE_FORMAT = "chat-evidence-synthetic-fixture"
FIXTURE_FORMAT_VERSION = 1
_TOP_KEYS = frozenset({"format", "format_version", "synthetic", "conversations"})
_CONVERSATION_KEYS = frozenset({"conversation_id", "messages"})
_MESSAGE_KEYS = frozenset({"native_id", "sender", "ts", "order", "kind", "text", "transcript", "placeholder"})
_TEXT_FIELDS = {"text": "export_text", "transcript": "export_transcript", "placeholder": "placeholder"}


class SyntheticFixtureAdapter:
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def parse(self, export: PreservedExport, profile: Profile) -> ParsedBatch:
        records: list[SourceRecord] = []
        problems: list[RecordProblem] = []
        for entry in export.files:
            file_locator = f"{entry.path}#"
            if not entry.path.endswith(".json"):
                problems.append(RecordProblem("fixture.unsupported_file", file_locator))
                continue
            try:
                document = loads_strict(export.read_bytes(entry.path), what="fixture")
            except InvalidInput as exc:
                problems.append(RecordProblem(exc.code, file_locator))
                continue
            _parse_document(entry.path, document, profile, records, problems)
        return ParsedBatch(records=tuple(records), problems=tuple(problems))


def _parse_document(
    relpath: str,
    document: object,
    profile: Profile,
    records: list[SourceRecord],
    problems: list[RecordProblem],
) -> None:
    root = f"{relpath}#"
    if (
        not isinstance(document, dict)
        or document.get("format") != FIXTURE_FORMAT
        or type(document.get("format_version")) is not int
        or document.get("format_version") != FIXTURE_FORMAT_VERSION
        or document.get("synthetic") is not True
    ):
        problems.append(RecordProblem("fixture.not_synthetic_fixture", root))
        return
    for key in sorted(document.keys() - _TOP_KEYS):
        problems.append(RecordProblem("fixture.unknown_field", root, key))
    conversations = document.get("conversations")
    if not isinstance(conversations, list):
        problems.append(RecordProblem("fixture.conversations_invalid", root, "conversations"))
        return
    for conversation_index, conversation in enumerate(conversations):
        pointer = f"{relpath}#/conversations/{conversation_index}"
        if not isinstance(conversation, dict) or conversation.keys() != _CONVERSATION_KEYS:
            problems.append(RecordProblem("fixture.conversation_fields", pointer))
            continue
        conversation_id = conversation["conversation_id"]
        messages = conversation["messages"]
        if not isinstance(conversation_id, str) or not conversation_id or not isinstance(messages, list):
            problems.append(RecordProblem("fixture.conversation_invalid", pointer))
            continue
        for message_index, message in enumerate(messages):
            record = _parse_message(message, f"{pointer}/messages/{message_index}", conversation_id, profile, problems)
            if record is not None:
                records.append(record)


def _parse_message(
    message: object,
    locator: str,
    conversation_id: str,
    profile: Profile,
    problems: list[RecordProblem],
) -> SourceRecord | None:
    if not isinstance(message, dict):
        problems.append(RecordProblem("fixture.message_not_object", locator))
        return None
    native_id = message.get("native_id")
    found: list[RecordProblem] = []

    def problem(code: str, field: str) -> None:
        found.append(RecordProblem(code, locator, field, native_id if isinstance(native_id, str) else None))

    for key in sorted(message.keys() - _MESSAGE_KEYS):
        problem("fixture.unknown_field", key)
    if not isinstance(native_id, str) or not native_id:
        problem("fixture.native_id_missing", "native_id")
    sender = message.get("sender")
    sender_id = profile.senders.get(sender) if isinstance(sender, str) else None
    if sender_id is None:
        problem("fixture.unknown_sender", "sender")
    ts = message.get("ts")
    ts_utc = ""
    if not isinstance(ts, str):
        problem("time.missing", "ts")
    else:
        try:
            ts_utc = resolve_source_timestamp(ts, profile.source_timezone)
        except InvalidInput as exc:
            problem(exc.code, "ts")
    order = message.get("order")
    if profile.order_policy == "native":
        if not isinstance(order, str) or not order:
            problem("fixture.native_order_missing", "order")
    elif "order" in message:
        problem("fixture.order_without_native_policy", "order")
    kind = message.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        problem("fixture.kind_unknown", "kind")
    present = [key for key in _TEXT_FIELDS if key in message]
    text = message[present[0]] if len(present) == 1 else None
    if len(present) != 1:
        problem("fixture.text_field_count", "text")
    elif not isinstance(text, str) or not text:
        problem("fixture.text_empty", present[0])
    if found:
        problems.extend(found)
        return None
    return SourceRecord(
        source_namespace=profile.source_namespace,
        conversation_id=conversation_id,
        native_message_id=native_id,
        sender_id=sender_id,
        ts_utc=ts_utc,
        ts_original=ts,
        source_timezone=profile.source_timezone,
        source_order=order if profile.order_policy == "native" else "",
        order_basis=profile.order_policy,
        kind=kind,
        text_origin=_TEXT_FIELDS[present[0]],
        text=text,
        raw_locator=locator,
    )
