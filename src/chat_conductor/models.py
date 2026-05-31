from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    conv_uuid: str
    msg_uuid: str
    sender: str
    text: str
    created_at: str


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
