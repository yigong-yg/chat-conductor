from __future__ import annotations

from collections.abc import Iterator

from .models import Message, Turn
from .text import normalize_text, sha256_hex, stable_json, turn_text


def iter_messages(conversation: dict) -> Iterator[Message]:
    conv_uuid = _require_string(conversation, "uuid", "conversation")
    messages = conversation.get("chat_messages") or []
    if not isinstance(messages, list):
        return

    for raw in messages:
        if not isinstance(raw, dict):
            continue
        msg_uuid = _require_string(raw, "uuid", f"message in {conv_uuid}")
        sender = _require_string(raw, "sender", f"message {msg_uuid}")
        if sender not in {"human", "assistant"}:
            continue
        yield Message(
            conv_uuid=conv_uuid,
            msg_uuid=msg_uuid,
            sender=sender,
            text=extract_message_text(raw),
            created_at=str(raw.get("created_at") or ""),
        )


def iter_turns(conversation: dict) -> Iterator[Turn]:
    conv_uuid = _require_string(conversation, "uuid", "conversation")
    conv_title = str(conversation.get("name") or "")
    collapsed = _collapse_same_sender(iter_messages(conversation))
    ordinal = 0
    index = 0

    while index < len(collapsed):
        current_sender, current_messages = collapsed[index]
        next_sender = collapsed[index + 1][0] if index + 1 < len(collapsed) else None
        next_messages = collapsed[index + 1][1] if index + 1 < len(collapsed) else []

        if current_sender == "human":
            human_messages = current_messages
            if next_sender == "assistant":
                assistant_messages = next_messages
                index += 2
            else:
                assistant_messages = []
                index += 1
        else:
            human_messages = []
            assistant_messages = current_messages
            index += 1

        source_messages = [*human_messages, *assistant_messages]
        if not source_messages:
            continue
        human_text = normalize_text("\n\n".join(message.text for message in human_messages if message.text))
        assistant_text = normalize_text(
            "\n\n".join(message.text for message in assistant_messages if message.text)
        )
        full_text = turn_text(human_text, assistant_text)
        identities = [[message.conv_uuid, message.msg_uuid] for message in source_messages]
        source_msg_ids_json = stable_json(identities)
        source_key = sha256_hex(source_msg_ids_json)
        content_hash = sha256_hex(full_text)
        yield Turn(
            turn_id="turn_" + source_key[:24],
            source_key=source_key,
            conv_uuid=conv_uuid,
            conv_title=conv_title,
            ts=next((message.created_at for message in source_messages if message.created_at), ""),
            ordinal=ordinal,
            human_text=human_text,
            assistant_text=assistant_text,
            full_text=full_text,
            source_msg_ids_json=source_msg_ids_json,
            content_hash=content_hash,
        )
        ordinal += 1


def extract_message_text(message: dict) -> str:
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return normalize_text(text)

    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return normalize_text("\n\n".join(part for part in parts if part.strip()))


def _collapse_same_sender(messages: Iterator[Message]) -> list[tuple[str, list[Message]]]:
    collapsed: list[tuple[str, list[Message]]] = []
    for message in messages:
        if collapsed and collapsed[-1][0] == message.sender:
            collapsed[-1][1].append(message)
        else:
            collapsed.append((message.sender, [message]))
    return collapsed


def _require_string(mapping: dict, key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required {key!r} on {label}")
    return value
