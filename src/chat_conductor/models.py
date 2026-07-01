from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Message:
    conv_uuid: str
    msg_uuid: str
    sender: str
    text: str
    created_at: str


@dataclass(frozen=True)
class RawMessage:
    source: str
    conv_id: str
    native_id: str
    speaker_id: str
    speaker_name: str
    ts: str
    kind: str
    text: str
    media_ref: str | None = None
    reply_to: str | None = None
    native_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class SpeakerSpan:
    speaker_id: str
    speaker_name: str
    text: str


@dataclass(frozen=True)
class Segment:
    segment_id: str
    source: str
    source_id: str
    conv_id: str
    conv_title: str
    participants: tuple[str, ...]
    ts_start: str
    ts_end: str
    ordinal: int
    full_text: str
    speaker_spans: tuple[SpeakerSpan, ...]
    media_refs: tuple[str, ...]
    source_msg_ids_json: str
    source_key: str
    content_hash: str
    native_meta: dict[str, Any] | None = None

    @property
    def turn_id(self) -> str:
        return self.segment_id

    @property
    def conv_uuid(self) -> str:
        return self.conv_id

    @property
    def ts(self) -> str:
        return self.ts_start

    def text_for_speaker(self, speaker_id: str) -> str:
        return "\n\n".join(
            span.text for span in self.speaker_spans if span.speaker_id == speaker_id and span.text
        )


@dataclass(frozen=True)
class Turn:
    turn_id: str
    source_key: str
    conv_uuid: str
    conv_title: str
    ts: str
    ordinal: int
    human_text: str
    assistant_text: str
    full_text: str
    source_msg_ids_json: str
    content_hash: str


@dataclass(frozen=True)
class IndexStats:
    conversations: int = 0
    messages: int = 0
    turns_seen: int = 0
    appended: int = 0
    skipped: int = 0
    superseded: int = 0


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    conv_uuid: str
    conv_title: str
    date: str
    role: str
    turn_id: str
    ordinal: int
    source: str = "claude"
    source_id: str = ""


@dataclass(frozen=True)
class StoredTurn:
    turn_id: str
    conv_uuid: str
    conv_title: str
    ts: str
    ordinal: int
    human_text: str
    assistant_text: str
    full_text: str
    source: str = "claude"
    source_id: str = ""
    speaker_spans_json: str = ""
