"""Stable message identity and the frozen ``payload-v1`` content hash.

Identity depends only on the dataset and the native identity tuple; it never
depends on text, timestamps, display names, import order or file paths. The
payload hash is a nonunique content-change detector, never a dedup key.
"""

from __future__ import annotations

from ..text import sha256_hex, stable_json

IDENTITY_VERSION = "chat-evidence-id-v1"
PAYLOAD_VERSION = "payload-v1"


def source_namespace(platform: str, account_key: str) -> str:
    return f"{platform}:{account_key}"


def message_id(dataset_id: str, source_namespace: str, conversation_id: str, native_message_id: str) -> str:
    identity = [IDENTITY_VERSION, dataset_id, source_namespace, conversation_id, native_message_id]
    return "m_" + sha256_hex(stable_json(identity))


def payload_sha256(
    *,
    sender_id: str,
    ts_original: str,
    ts_utc: str,
    source_timezone: str,
    source_order: str,
    order_basis: str,
    kind: str,
    text_origin: str,
    text: str,
) -> str:
    """Frozen payload-v1 definition: a positional JSON array, hashed as UTF-8.

    Covers sender, original and UTC timestamps, declared zone, source order and
    its basis, kind, text origin and exact text. Excludes identity, raw paths,
    display names and import metadata.
    """
    payload = [
        PAYLOAD_VERSION,
        sender_id,
        ts_original,
        ts_utc,
        source_timezone,
        source_order,
        order_basis,
        kind,
        text_origin,
        text,
    ]
    return sha256_hex(stable_json(payload))


def payload_of_row(row: dict) -> str:
    return payload_sha256(
        sender_id=row["sender_id"],
        ts_original=row["ts_original"],
        ts_utc=row["ts_utc"],
        source_timezone=row["source_timezone"],
        source_order=row["source_order"],
        order_basis=row["order_basis"],
        kind=row["kind"],
        text_origin=row["text_origin"],
        text=row["text"],
    )
