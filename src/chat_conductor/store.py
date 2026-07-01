from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

from .models import IndexStats, SearchResult, Segment, StoredTurn
from .text import build_fts_match, escape_like, split_query_terms, stable_json

SCHEMA_VERSION = "2"


def default_index_path() -> Path:
    override = os.environ.get("CHAT_CONDUCTOR_INDEX")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "chat-conductor" / "index.sqlite3"
    return Path.home() / ".chat-conductor" / "index.sqlite3"


def connect(index_path: Path) -> sqlite3.Connection:
    index_path = index_path.expanduser()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    if not _fts5_trigram_available(connection):
        version = connection.execute("select sqlite_version()").fetchone()[0]
        raise RuntimeError(f"SQLite FTS5 trigram tokenizer is unavailable (sqlite {version})")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY,
            turn_id TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'claude',
            source_id TEXT NOT NULL DEFAULT '',
            source_key TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            source_msg_ids TEXT NOT NULL,
            conv_uuid TEXT NOT NULL,
            conv_title TEXT NOT NULL,
            ts TEXT NOT NULL,
            ts_start TEXT NOT NULL DEFAULT '',
            ts_end TEXT NOT NULL DEFAULT '',
            ordinal INTEGER NOT NULL,
            human_text TEXT NOT NULL,
            assistant_text TEXT NOT NULL,
            full_text TEXT NOT NULL,
            speaker_spans TEXT NOT NULL DEFAULT '[]',
            participants TEXT NOT NULL DEFAULT '[]',
            media_refs TEXT NOT NULL DEFAULT '[]',
            active INTEGER NOT NULL DEFAULT 1,
            superseded_by INTEGER REFERENCES turns(id),
            indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_turns_source_active ON turns(source, source_key, active);
        CREATE INDEX IF NOT EXISTS idx_turns_content_hash ON turns(content_hash);
        CREATE INDEX IF NOT EXISTS idx_turns_conv_uuid ON turns(conv_uuid);
        CREATE INDEX IF NOT EXISTS idx_turns_source_conv ON turns(source, conv_uuid, ordinal);
        CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);

        CREATE VIRTUAL TABLE IF NOT EXISTS turn_fts
        USING fts5(full_text, human_text, assistant_text, tokenize='trigram');
        """
    )
    _ensure_column(connection, "turns", "source", "TEXT NOT NULL DEFAULT 'claude'")
    _ensure_column(connection, "turns", "source_id", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(connection, "turns", "ts_start", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(connection, "turns", "ts_end", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(connection, "turns", "speaker_spans", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(connection, "turns", "participants", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(connection, "turns", "media_refs", "TEXT NOT NULL DEFAULT '[]'")
    connection.execute("UPDATE turns SET source_id = conv_uuid WHERE source_id = ''")
    connection.execute("UPDATE turns SET ts_start = ts WHERE ts_start = ''")
    connection.execute("UPDATE turns SET ts_end = ts WHERE ts_end = ''")
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    connection.commit()


def upsert_segment(connection: sqlite3.Connection, segment: Segment) -> str:
    active = connection.execute(
        """
        SELECT id, content_hash
        FROM turns
        WHERE source = ? AND source_key = ? AND active = 1
        ORDER BY id DESC
        LIMIT 1
        """,
        (segment.source, segment.source_key),
    ).fetchone()
    if active and active["content_hash"] == segment.content_hash:
        return "skipped"

    stale_ids = set()
    if active:
        stale_ids.add(int(active["id"]))
    stale_ids.update(_overlapping_active_turn_ids(connection, segment))

    human_text = segment.text_for_speaker("human")
    assistant_text = segment.text_for_speaker("assistant")
    cursor = connection.execute(
        """
        INSERT INTO turns (
            turn_id, source, source_id, source_key, content_hash, source_msg_ids,
            conv_uuid, conv_title, ts, ts_start, ts_end, ordinal, human_text,
            assistant_text, full_text, speaker_spans, participants, media_refs, active
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (
            segment.segment_id,
            segment.source,
            segment.source_id,
            segment.source_key,
            segment.content_hash,
            segment.source_msg_ids_json,
            segment.conv_id,
            segment.conv_title,
            segment.ts_start,
            segment.ts_start,
            segment.ts_end,
            segment.ordinal,
            human_text,
            assistant_text,
            segment.full_text,
            _speaker_spans_json(segment),
            stable_json(list(segment.participants)),
            stable_json(list(segment.media_refs)),
        ),
    )
    row_id = cursor.lastrowid
    connection.execute(
        """
        INSERT INTO turn_fts(rowid, full_text, human_text, assistant_text)
        VALUES (?, ?, ?, ?)
        """,
        (row_id, segment.full_text, human_text, assistant_text),
    )

    if stale_ids:
        _deactivate_turns(connection, sorted(stale_ids), row_id)
        return "superseded"
    return "appended"


def search(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    after: str | None = None,
    before: str | None = None,
    role: str | None = None,
    conv: str | None = None,
    source: str | None = None,
) -> list[SearchResult]:
    if role not in {None, "human", "assistant"}:
        raise ValueError("role must be human or assistant")

    if role == "human":
        text_column, fts_column, provenance_role = "t.human_text", "human_text", "human"
    elif role == "assistant":
        text_column, fts_column, provenance_role = "t.assistant_text", "assistant_text", "assistant"
    else:
        text_column, fts_column, provenance_role = "t.full_text", None, "turn"

    fts_terms, like_terms = split_query_terms(query)
    if not fts_terms and not like_terms:
        raise ValueError("query must not be empty")

    clauses: list[str] = []
    params: list[object] = []

    if fts_terms:
        clauses.append("turn_fts MATCH ?")
        params.append(build_fts_match(fts_terms, fts_column))
        from_clause = "turn_fts JOIN turns t ON t.id = turn_fts.rowid"
        score_expr = "-bm25(turn_fts)"
        order_clause = "bm25(turn_fts), t.ts DESC"
    else:
        from_clause = "turns t"
        score_expr = "0.0"
        order_clause = "t.ts DESC"

    clauses.append("t.active = 1")

    for term in like_terms:
        clauses.append(f"{text_column} LIKE ? ESCAPE '\\'")
        params.append(f"%{escape_like(term)}%")

    if source:
        clauses.append("t.source = ?")
        params.append(source)
    if after:
        clauses.append("t.ts >= ?")
        params.append(after)
    if before:
        clauses.append("t.ts <= ?")
        params.append(before)
    if conv:
        clauses.append("(t.conv_uuid = ? OR t.conv_title LIKE ? ESCAPE '\\')")
        params.extend([conv, f"%{escape_like(conv)}%"])
    params.append(limit)

    rows = connection.execute(
        f"""
        SELECT
            {text_column} AS text,
            {score_expr} AS score,
            t.source,
            t.source_id,
            t.conv_uuid,
            t.conv_title,
            t.ts AS date,
            t.turn_id,
            t.ordinal
        FROM {from_clause}
        WHERE {" AND ".join(clauses)}
        ORDER BY {order_clause}
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        SearchResult(
            text=row["text"],
            score=float(row["score"]),
            conv_uuid=row["conv_uuid"],
            conv_title=row["conv_title"],
            date=row["date"],
            role=provenance_role,
            turn_id=row["turn_id"],
            ordinal=int(row["ordinal"]),
            source=row["source"],
            source_id=row["source_id"],
        )
        for row in rows
    ]


def fetch_turn_window(
    connection: sqlite3.Connection,
    conv_uuid: str,
    center_ordinal: int,
    radius: int,
    *,
    source: str | None = None,
) -> list[StoredTurn]:
    clauses = [
        "active = 1",
        "conv_uuid = ?",
        "ordinal BETWEEN ? AND ?",
    ]
    params: list[object] = [conv_uuid, center_ordinal - radius, center_ordinal + radius]
    if source:
        clauses.append("source = ?")
        params.append(source)
    rows = connection.execute(
        f"""
        SELECT
            turn_id,
            source,
            source_id,
            conv_uuid,
            conv_title,
            ts,
            ordinal,
            human_text,
            assistant_text,
            full_text,
            speaker_spans
        FROM turns
        WHERE {" AND ".join(clauses)}
        ORDER BY ordinal
        """,
        params,
    ).fetchall()
    return [
        StoredTurn(
            turn_id=row["turn_id"],
            conv_uuid=row["conv_uuid"],
            conv_title=row["conv_title"],
            ts=row["ts"],
            ordinal=int(row["ordinal"]),
            human_text=row["human_text"],
            assistant_text=row["assistant_text"],
            full_text=row["full_text"],
            source=row["source"],
            source_id=row["source_id"],
            speaker_spans_json=row["speaker_spans"],
        )
        for row in rows
    ]


def status(connection: sqlite3.Connection) -> dict[str, int | str]:
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS rows_total,
            SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) AS active_turns,
            SUM(CASE WHEN active = 0 THEN 1 ELSE 0 END) AS superseded_turns,
            COUNT(DISTINCT conv_uuid) AS conversations,
            COUNT(DISTINCT source) AS sources
        FROM turns
        """
    ).fetchone()
    version = connection.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()["value"]
    return {
        "schema_version": version,
        "rows_total": int(row["rows_total"] or 0),
        "active_turns": int(row["active_turns"] or 0),
        "superseded_turns": int(row["superseded_turns"] or 0),
        "conversations": int(row["conversations"] or 0),
        "sources": int(row["sources"] or 0),
    }


def apply_action(stats: IndexStats, action: str) -> IndexStats:
    values = stats.__dict__.copy()
    if action == "appended":
        values["appended"] += 1
    elif action == "skipped":
        values["skipped"] += 1
    elif action == "superseded":
        values["superseded"] += 1
    else:
        raise ValueError(f"unknown index action: {action}")
    return IndexStats(**values)


def _fts5_trigram_available(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("CREATE VIRTUAL TABLE temp._fts_probe USING fts5(x, tokenize='trigram')")
        connection.execute("DROP TABLE temp._fts_probe")
    except sqlite3.Error:
        return False
    return True


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _overlapping_active_turn_ids(connection: sqlite3.Connection, segment: Segment) -> list[int]:
    incoming_ids = _source_id_set(segment.source_msg_ids_json)
    if not incoming_ids:
        return []
    rows = connection.execute(
        """
        SELECT id, source_msg_ids
        FROM turns
        WHERE source = ? AND conv_uuid = ? AND active = 1 AND source_key <> ?
        """,
        (segment.source, segment.conv_id, segment.source_key),
    ).fetchall()
    return [
        int(row["id"])
        for row in rows
        if incoming_ids.intersection(_source_id_set(row["source_msg_ids"]))
    ]


def _deactivate_turns(connection: sqlite3.Connection, stale_ids: list[int], superseded_by: int) -> None:
    for stale_id in stale_ids:
        connection.execute(
            "UPDATE turns SET active = 0, superseded_by = ? WHERE id = ?",
            (superseded_by, stale_id),
        )
        connection.execute("DELETE FROM turn_fts WHERE rowid = ?", (stale_id,))


def _source_id_set(source_msg_ids_json: str) -> set[tuple[str, str]]:
    raw = json.loads(source_msg_ids_json)
    return {
        (str(item[0]), str(item[1]))
        for item in raw
        if isinstance(item, list) and len(item) == 2
    }


def _speaker_spans_json(segment: Segment) -> str:
    return stable_json([
        {
            "speaker_id": span.speaker_id,
            "speaker_name": span.speaker_name,
            "text": span.text,
        }
        for span in segment.speaker_spans
    ])
