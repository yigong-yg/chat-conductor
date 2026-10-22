"""Tool-owned query receipts, written once per ID inside the resolved run directory.

A receipt records what a tool executed and returned. Its ``body_sha256`` only
detects accidental edits; it is unkeyed, so verification always replays the
receipt against the canonical pinned evidence instead of trusting it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import secrets

from . import TOOL_NAME, TOOL_VERSION
from .errors import InvalidInput
from .fsutil import write_new_file
from .jsonutil import canonical_bytes, loads_strict, sha256_json
from .paths import RECEIPT_ID_RE, contained_path
from .runs import Run

RECEIPT_FORMAT = "chat-evidence-receipt"
RECEIPT_VERSION = 1
CURSOR_PREFIX = "c1"


def new_receipt_id() -> str:
    return f"q_{secrets.token_hex(12)}"


def receipt_header(run: Run, receipt_id: str, operation: str) -> dict:
    return {
        "receipt_format": RECEIPT_FORMAT,
        "receipt_version": RECEIPT_VERSION,
        "receipt_id": receipt_id,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "corpus_revision": run.corpus_revision,
        "operation": operation,
    }


def seal(body: dict) -> dict:
    sealed = {key: value for key, value in body.items() if key != "body_sha256"}
    sealed["body_sha256"] = sha256_json(sealed)
    return sealed


def write_receipt(run: Run, receipt: dict) -> None:
    path = contained_path(run.directory, "receipts", f"{receipt['receipt_id']}.json")
    write_new_file(path, canonical_bytes(seal(receipt)) + b"\n")


def load_receipt(run: Run, receipt_id: object) -> tuple[dict, bytes]:
    """Load a receipt of this run by ID; return it with its exact file bytes."""
    if not isinstance(receipt_id, str) or not RECEIPT_ID_RE.fullmatch(receipt_id):
        raise InvalidInput("receipt.invalid_id", "receipt ids look like q_<letters, digits, '_' or '-'>")
    path = contained_path(run.directory, "receipts", f"{receipt_id}.json")
    try:
        data = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        raise InvalidInput("receipt.missing", "a referenced receipt does not exist in this run", details={"receipt_id": receipt_id}) from None
    receipt = loads_strict(data, what="receipt")
    if not isinstance(receipt, dict):
        raise InvalidInput("receipt.malformed", "a receipt is not a JSON object", details={"receipt_id": receipt_id})
    body = {key: value for key, value in receipt.items() if key != "body_sha256"}
    if receipt.get("body_sha256") != sha256_json(body):
        raise InvalidInput("receipt.digest_mismatch", "a receipt changed after the tool wrote it", details={"receipt_id": receipt_id})
    if receipt.get("receipt_format") != RECEIPT_FORMAT or receipt.get("receipt_version") != RECEIPT_VERSION:
        raise InvalidInput("receipt.unsupported", "unsupported receipt format", details={"receipt_id": receipt_id})
    if receipt.get("receipt_id") != receipt_id:
        raise InvalidInput("receipt.id_mismatch", "a receipt's id differs from its file name", details={"receipt_id": receipt_id})
    if receipt.get("run_id") != run.run_id or receipt.get("dataset_id") != run.dataset_id:
        raise InvalidInput("receipt.other_run", "the receipt belongs to a different run or dataset", details={"receipt_id": receipt_id})
    if receipt.get("corpus_revision") != run.corpus_revision:
        raise InvalidInput("receipt.stale_revision", "the receipt was produced at a different corpus revision", details={"receipt_id": receipt_id})
    return receipt, data


def encode_cursor(payload: dict) -> str:
    body = canonical_bytes(payload)
    encoded = base64.urlsafe_b64encode(body).decode("ascii").rstrip("=")
    return f"{CURSOR_PREFIX}.{encoded}.{hashlib.sha256(body).hexdigest()[:16]}"


def decode_cursor(token: str, binding: dict) -> dict:
    """Decode an opaque continuation cursor and require it to match ``binding``."""
    try:
        prefix, encoded, check = token.split(".")
        if prefix != CURSOR_PREFIX:
            raise ValueError
        body = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error):
        raise InvalidInput("cursor.malformed", "the continuation cursor is malformed") from None
    if hashlib.sha256(body).hexdigest()[:16] != check:
        raise InvalidInput("cursor.corrupt", "the continuation cursor is corrupt")
    payload = loads_strict(body, what="cursor")
    if not isinstance(payload, dict):
        raise InvalidInput("cursor.malformed", "the continuation cursor is malformed")
    for key, value in binding.items():
        if payload.get(key) != value:
            raise InvalidInput(
                "cursor.binding",
                "the cursor belongs to a different run, operation, argument set, revision or ordering",
                details={"field": key},
            )
    return payload
