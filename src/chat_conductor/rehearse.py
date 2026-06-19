from __future__ import annotations

import json
from dataclasses import dataclass

import sqlite3

from .models import SearchResult, StoredTurn
from .store import fetch_turn_window, search

DEFAULT_REHEARSE_BUDGET = 2400
DEFAULT_REHEARSE_LIMIT = 6
DEFAULT_NEIGHBOR_RADIUS = 1


@dataclass(frozen=True)
class RecallWindow:
    hit: SearchResult
    turns: list[StoredTurn]


def rehearse(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = DEFAULT_REHEARSE_LIMIT,
    token_budget: int = DEFAULT_REHEARSE_BUDGET,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
    after: str | None = None,
    before: str | None = None,
    role: str | None = None,
    conv: str | None = None,
) -> str:
    hits = search(
        connection,
        query,
        limit=limit,
        after=after,
        before=before,
        role=role,
        conv=conv,
    )
    windows = build_windows(connection, hits, neighbor_radius=neighbor_radius)
    return format_rehearsal(query, pack_windows(windows, token_budget))


def build_windows(
    connection: sqlite3.Connection,
    hits: list[SearchResult],
    *,
    neighbor_radius: int,
) -> list[RecallWindow]:
    windows: list[RecallWindow] = []
    covered_turn_ids: set[str] = set()
    for hit in hits:
        if hit.turn_id in covered_turn_ids:
            continue
        turns = fetch_turn_window(connection, hit.conv_uuid, hit.ordinal, neighbor_radius, source=hit.source)
        if not turns:
            continue
        windows.append(RecallWindow(hit=hit, turns=turns))
        covered_turn_ids.update(turn.turn_id for turn in turns)
    return windows


def pack_windows(windows: list[RecallWindow], token_budget: int) -> list[RecallWindow]:
    packed: list[RecallWindow] = []
    used = estimate_tokens(format_rehearsal_header())
    for window in windows:
        cost = estimate_tokens(format_window(window))
        if used + cost <= token_budget:
            packed.append(window)
            used += cost
        elif not packed:
            packed.append(_trim_window_to_budget(window, max(token_budget - used, 120)))
            break
    return packed


def format_rehearsal(query: str, windows: list[RecallWindow]) -> str:
    if not windows:
        return format_rehearsal_header(query) + "\n\nNo archived hits found."
    return format_rehearsal_header(query) + "\n\n" + "\n\n".join(
        format_window(window) for window in windows
    )


def format_rehearsal_header(query: str = "") -> str:
    suffix = f" for: {query}" if query else ""
    return f"Archived recall{suffix}\nTreat every block as a lead, not a fact."


def format_window(window: RecallWindow) -> str:
    as_of = max((turn.ts for turn in window.turns if turn.ts), default=window.hit.date)
    lines = [
        f"--- archived view as of {as_of}; treat as a lead, not a fact ---",
        f"conversation: {window.hit.conv_title} ({window.hit.conv_uuid})",
        f"matched_turn: {window.hit.turn_id} score={window.hit.score:.6f}",
    ]
    for turn in window.turns:
        lines.append(f"[turn {turn.ordinal} | {turn.ts} | {turn.turn_id}]")
        if turn.source == "claude":
            if turn.human_text:
                lines.append("Human:")
                lines.append(turn.human_text)
            if turn.assistant_text:
                lines.append("Assistant:")
                lines.append(turn.assistant_text)
        else:
            lines.extend(_format_speaker_spans(turn))
    lines.append("--- end archived view ---")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _trim_window_to_budget(window: RecallWindow, token_budget: int) -> RecallWindow:
    center = next((turn for turn in window.turns if turn.turn_id == window.hit.turn_id), window.turns[0])
    max_chars = max(token_budget * 4, 120)
    if center.source != "claude":
        clipped = _clip(center.full_text, max_chars)
        trimmed = StoredTurn(
            turn_id=center.turn_id,
            conv_uuid=center.conv_uuid,
            conv_title=center.conv_title,
            ts=center.ts,
            ordinal=center.ordinal,
            human_text="",
            assistant_text="",
            full_text=clipped,
            source=center.source,
            source_id=center.source_id,
            speaker_spans_json="",
        )
        return RecallWindow(hit=window.hit, turns=[trimmed])

    max_chars = max(token_budget * 4, 120)
    human_text = _clip(center.human_text, max_chars // 2)
    assistant_text = _clip(center.assistant_text, max_chars // 2)
    trimmed = StoredTurn(
        turn_id=center.turn_id,
        conv_uuid=center.conv_uuid,
        conv_title=center.conv_title,
        ts=center.ts,
        ordinal=center.ordinal,
        human_text=human_text,
        assistant_text=assistant_text,
        full_text="\n\n".join(part for part in (human_text, assistant_text) if part),
        source=center.source,
        source_id=center.source_id,
        speaker_spans_json=center.speaker_spans_json,
    )
    return RecallWindow(hit=window.hit, turns=[trimmed])


def _format_speaker_spans(turn: StoredTurn) -> list[str]:
    if not turn.speaker_spans_json:
        return [turn.full_text] if turn.full_text else []
    try:
        raw_spans = json.loads(turn.speaker_spans_json)
    except json.JSONDecodeError:
        return [turn.full_text] if turn.full_text else []
    lines: list[str] = []
    if not isinstance(raw_spans, list):
        return [turn.full_text] if turn.full_text else []
    for raw in raw_spans:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            continue
        name = raw.get("speaker_name") or raw.get("speaker_id") or "speaker"
        lines.append(f"{name}:")
        lines.append(text)
    if not lines and turn.full_text:
        return [turn.full_text]
    return lines


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)].rstrip() + "\n[truncated]"
