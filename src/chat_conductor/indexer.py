from __future__ import annotations

from pathlib import Path

from .adapters import SourceAdapter, resolve_adapter
from .models import IndexStats
from .store import apply_action, connect, init_schema, upsert_segment


def index_source(
    source: Path,
    index_path: Path,
    *,
    source_name: str | None = None,
    adapter: SourceAdapter | None = None,
) -> IndexStats:
    source_adapter = adapter or resolve_adapter(source, source_name)
    batch = source_adapter.load(source)

    connection = connect(index_path)
    init_schema(connection)

    stats = IndexStats(conversations=batch.conversations, messages=batch.messages)
    try:
        with connection:
            for segment in batch.segments:
                stats = IndexStats(
                    conversations=stats.conversations,
                    messages=stats.messages,
                    turns_seen=stats.turns_seen + 1,
                    appended=stats.appended,
                    skipped=stats.skipped,
                    superseded=stats.superseded,
                )
                stats = apply_action(stats, upsert_segment(connection, segment))
    finally:
        connection.close()
    return stats


def index_export(source: Path, index_path: Path) -> IndexStats:
    return index_source(source, index_path, source_name="claude")
