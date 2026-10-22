"""Strict JSON parsing and canonical JSON encoding.

Strict parsing rejects duplicate object keys, non-finite numbers, byte-order
marks, invalid UTF-8 and lone surrogate escapes. Canonical encoding matches
``chat_conductor.text.stable_json`` and additionally refuses non-finite floats.
"""

from __future__ import annotations

import hashlib
import json
import math

from .errors import InvalidInput


class _DuplicateKey(ValueError):
    pass


class _NonFinite(ValueError):
    pass


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(name: str) -> object:
    raise _NonFinite(name)


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise _NonFinite(literal)
    return value


def loads_strict(data: bytes | str, *, what: str) -> object:
    """Parse JSON strictly; raise InvalidInput with a ``<what>.*`` code on failure."""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidInput(f"{what}.encoding", f"not valid UTF-8 at byte {exc.start}") from None
    else:
        text = data
    if text.startswith("﻿"):
        raise InvalidInput(f"{what}.bom", "a UTF-8 byte-order mark is not accepted")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except _DuplicateKey as exc:
        raise InvalidInput(f"{what}.duplicate_key", "an object repeats a key", details={"key": str(exc)[:80]}) from None
    except _NonFinite:
        raise InvalidInput(f"{what}.nonfinite_number", "NaN and infinite numbers are not accepted") from None
    except json.JSONDecodeError as exc:
        raise InvalidInput(
            f"{what}.invalid_json",
            "malformed JSON",
            details={"line": exc.lineno, "column": exc.colno},
        ) from None
    except (ValueError, RecursionError):
        raise InvalidInput(f"{what}.invalid_json", "malformed or excessively large JSON value") from None
    _reject_surrogates(value, what)
    return value


def _reject_surrogates(value: object, what: str) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            _check_text(item, what)
        elif isinstance(item, dict):
            for key, child in item.items():
                _check_text(key, what)
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)


def _check_text(text: str, what: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidInput(f"{what}.lone_surrogate", "a string contains an unpaired surrogate escape") from None


def dumps_canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_bytes(value: object) -> bytes:
    return dumps_canonical(value).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
