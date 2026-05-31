from __future__ import annotations

from pathlib import Path

from .export import conversations_json_from_export, load_conversations
from .models import IndexStats
from .store import apply_action, connect, init_schema, upsert_turn
from .turns import iter_messages, iter_turns


def index_export(source: Path, index_path: Path) -> IndexStats:
    with conversations_json_from_export(source) as conversations_path:
        conversations = load_conversations(conversations_path)

    connection = connect(index_path)
    init_schema(connection)

    stats = IndexStats(conversations=len(conversations))
    try:
        with connection:
            for conversation in conversations:
                messages = list(iter_messages(conversation))
                stats = IndexStats(
                    conversations=stats.conversations,
                    messages=stats.messages + len(messages),
                    turns_seen=stats.turns_seen,
                    appended=stats.appended,
                    skipped=stats.skipped,
                    superseded=stats.superseded,
                )
                for turn in iter_turns(conversation):
                    stats = IndexStats(
                        conversations=stats.conversations,
                        messages=stats.messages,
                        turns_seen=stats.turns_seen + 1,
                        appended=stats.appended,
                        skipped=stats.skipped,
                        superseded=stats.superseded,
                    )
                    stats = apply_action(stats, upsert_turn(connection, turn))
    finally:
        connection.close()
    return stats
