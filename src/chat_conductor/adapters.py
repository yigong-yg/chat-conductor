from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .export import conversations_json_from_export, load_conversations
from .models import RawMessage, Segment, SpeakerSpan, Turn
from .text import normalize_text, sha256_hex, stable_json
from .turns import iter_messages, iter_turns

WECHAT_SEGMENT_GAP_SECONDS = 300


@dataclass(frozen=True)
class SourceIndexBatch:
    source: str
    conversations: int
    messages: int
    segments: tuple[Segment, ...]


class SourceAdapter(Protocol):
    name: str

    def detect(self, source: Path) -> bool:
        ...

    def load(self, source: Path) -> SourceIndexBatch:
        ...


class ClaudeAdapter:
    name = "claude"

    def detect(self, source: Path) -> bool:
        source = source.expanduser()
        if source.is_file() and source.name == "conversations.json":
            return True
        if source.is_dir():
            return (source / "conversations.json").is_file() or any(source.glob("**/conversations.json"))
        if source.is_file() and source.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(source) as archive:
                    return any(Path(name).name == "conversations.json" for name in archive.namelist())
            except zipfile.BadZipFile:
                return False
        return False

    def load(self, source: Path) -> SourceIndexBatch:
        with conversations_json_from_export(source) as conversations_path:
            conversations = load_conversations(conversations_path)

        segments: list[Segment] = []
        message_count = 0
        for conversation in conversations:
            message_count += len(list(iter_messages(conversation)))
            segments.extend(_segment_from_turn(turn) for turn in iter_turns(conversation))
        return SourceIndexBatch(
            source=self.name,
            conversations=len(conversations),
            messages=message_count,
            segments=tuple(segments),
        )


class WeChatAdapter:
    name = "wechat"

    def detect(self, source: Path) -> bool:
        return any(_looks_like_wechat_jsonl(path) for path in _candidate_jsonl_files(source.expanduser()))

    def load(self, source: Path) -> SourceIndexBatch:
        text_files = tuple(path for path in _candidate_jsonl_files(source.expanduser()) if _looks_like_wechat_jsonl(path))
        if not text_files:
            raise FileNotFoundError(f"could not locate WeChat JSONL text files under {source}")

        all_segments: list[Segment] = []
        message_count = 0
        for text_file in text_files:
            messages, conv_title = _load_wechat_messages(text_file)
            message_count += len(messages)
            all_segments.extend(_assemble_wechat_segments(messages, conv_title=conv_title))
        return SourceIndexBatch(
            source=self.name,
            conversations=len(text_files),
            messages=message_count,
            segments=tuple(all_segments),
        )


ADAPTERS: tuple[SourceAdapter, ...] = (ClaudeAdapter(), WeChatAdapter())


def resolve_adapter(source: Path, source_name: str | None = None) -> SourceAdapter:
    if source_name:
        for adapter in ADAPTERS:
            if adapter.name == source_name:
                return adapter
        known = ", ".join(adapter.name for adapter in ADAPTERS)
        raise ValueError(f"unknown source {source_name!r}; expected one of: {known}")

    matches = [adapter for adapter in ADAPTERS if adapter.detect(source)]
    if not matches:
        known = ", ".join(adapter.name for adapter in ADAPTERS)
        raise FileNotFoundError(f"could not detect chat source format for {source}; known sources: {known}")
    if len(matches) > 1:
        names = ", ".join(adapter.name for adapter in matches)
        raise ValueError(f"multiple source adapters match {source}: {names}; pass --source explicitly")
    return matches[0]


def _segment_from_turn(turn: Turn) -> Segment:
    spans = (
        SpeakerSpan("human", "Human", turn.human_text),
        SpeakerSpan("assistant", "Assistant", turn.assistant_text),
    )
    participants = tuple(span.speaker_id for span in spans if span.text)
    return Segment(
        segment_id=turn.turn_id,
        source="claude",
        source_id=turn.conv_uuid,
        conv_id=turn.conv_uuid,
        conv_title=turn.conv_title,
        participants=participants,
        ts_start=turn.ts,
        ts_end=turn.ts,
        ordinal=turn.ordinal,
        full_text=turn.full_text,
        speaker_spans=spans,
        media_refs=(),
        source_msg_ids_json=turn.source_msg_ids_json,
        source_key=turn.source_key,
        content_hash=turn.content_hash,
    )


def _candidate_jsonl_files(source: Path) -> tuple[Path, ...]:
    if source.is_file() and source.suffix.lower() == ".jsonl":
        return (source,)
    if not source.is_dir():
        return ()
    text_dir = source / "texts"
    if text_dir.is_dir():
        return tuple(sorted(text_dir.glob("*.jsonl")))
    return tuple(sorted(source.glob("*.jsonl")))


def _looks_like_wechat_jsonl(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return False
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            seen_message = False
            seen_header = False
            for _, line in zip(range(50), handle):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    return False
                record_type = raw.get("_type")
                if record_type == "header":
                    seen_header = True
                if record_type == "message" and "platformMessageId" in raw and "timestamp" in raw:
                    seen_message = True
                if seen_header and seen_message:
                    return True
            return seen_message
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _load_wechat_messages(path: Path) -> tuple[list[RawMessage], str]:
    conv_title = path.stem
    messages: list[RawMessage] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            if raw.get("_type") == "header":
                conv_title = _wechat_title(raw, default=conv_title)
                continue
            if raw.get("_type") != "message":
                continue
            timestamp = raw.get("timestamp")
            sender = raw.get("sender")
            if not isinstance(timestamp, int) or not isinstance(sender, str) or not sender:
                continue
            native_id = raw.get("platformMessageId")
            if not isinstance(native_id, str) or not native_id:
                native_id = f"{path.stem}:{line_number}"
            content = raw.get("content")
            text = normalize_text(content) if isinstance(content, str) else ""
            source_id = path.stem
            messages.append(
                RawMessage(
                    source="wechat",
                    conv_id=source_id,
                    native_id=native_id,
                    speaker_id=sender,
                    speaker_name=str(raw.get("accountName") or sender),
                    ts=_iso_from_epoch(timestamp),
                    kind=_wechat_kind(raw.get("type")),
                    text=text,
                    reply_to=raw.get("replyToMessageId") if isinstance(raw.get("replyToMessageId"), str) else None,
                    native_meta={"type": raw.get("type"), "line_number": line_number},
                )
            )
    messages.sort(key=lambda message: (message.ts, int((message.native_meta or {}).get("line_number", 0))))
    return messages, conv_title


def _assemble_wechat_segments(messages: list[RawMessage], *, conv_title: str) -> tuple[Segment, ...]:
    by_conv: dict[str, list[RawMessage]] = defaultdict(list)
    for message in messages:
        by_conv[message.conv_id].append(message)

    segments: list[Segment] = []
    for conv_id, conv_messages in sorted(by_conv.items()):
        current: list[RawMessage] = []
        ordinal = 0
        previous_ts: datetime | None = None
        for message in conv_messages:
            current_ts = _parse_iso(message.ts)
            if current and previous_ts and (current_ts - previous_ts).total_seconds() > WECHAT_SEGMENT_GAP_SECONDS:
                segments.append(_wechat_segment(current, conv_title=conv_title, ordinal=ordinal))
                ordinal += 1
                current = []
            current.append(message)
            previous_ts = current_ts
        if current:
            segments.append(_wechat_segment(current, conv_title=conv_title, ordinal=ordinal))
    return tuple(segments)


def _wechat_segment(messages: list[RawMessage], *, conv_title: str, ordinal: int) -> Segment:
    spans: list[SpeakerSpan] = []
    media_refs: list[str] = []
    for message in messages:
        text = message.text
        if message.media_ref:
            media_refs.append(message.media_ref)
        if spans and spans[-1].speaker_id == message.speaker_id:
            previous = spans[-1]
            joined = normalize_text("\n\n".join(part for part in (previous.text, text) if part))
            spans[-1] = SpeakerSpan(previous.speaker_id, previous.speaker_name, joined)
        else:
            spans.append(SpeakerSpan(message.speaker_id, message.speaker_name, text))

    full_text = normalize_text("\n\n".join(span.text for span in spans if span.text))
    if not full_text:
        full_text = normalize_text("\n\n".join(f"[{message.kind}]" for message in messages))
    identities = [[message.conv_id, message.native_id] for message in messages]
    source_msg_ids_json = stable_json(identities)
    source_key = sha256_hex(source_msg_ids_json)
    participants = tuple(dict.fromkeys(span.speaker_id for span in spans if span.text))
    return Segment(
        segment_id="turn_" + source_key[:24],
        source="wechat",
        source_id=messages[0].conv_id,
        conv_id=messages[0].conv_id,
        conv_title=conv_title,
        participants=participants,
        ts_start=messages[0].ts,
        ts_end=messages[-1].ts,
        ordinal=ordinal,
        full_text=full_text,
        speaker_spans=tuple(spans),
        media_refs=tuple(media_refs),
        source_msg_ids_json=source_msg_ids_json,
        source_key=source_key,
        content_hash=sha256_hex(full_text),
    )


def _wechat_title(raw: dict, *, default: str) -> str:
    meta = raw.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("name"), str) and meta["name"].strip():
        return meta["name"].strip()
    return default


def _wechat_kind(value: object) -> str:
    if value == 80:
        return "system"
    if value == 7:
        return "rich"
    if value == 4:
        return "file"
    if value == 25:
        return "reply"
    if value == 99:
        return "transfer"
    return "text"


def _iso_from_epoch(value: int) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

