"""Private import profiles: sender mapping, declared zone and identity/order rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from . import identity
from .errors import ConfigError, EvidenceError, InvalidInput
from .jsonutil import canonical_bytes, loads_strict, sha256_bytes
from .records import ORDER_BASES
from .timeutil import load_zone

PROFILE_FORMAT = "chat-evidence-profile"
PROFILE_VERSION = 1
_PROFILE_KEYS = frozenset({
    "profile_format",
    "profile_version",
    "adapter",
    "adapter_version",
    "platform",
    "account_key",
    "source_timezone",
    "order_policy",
    "senders",
})
_PLATFORM_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class Profile:
    adapter: str
    adapter_version: str
    platform: str
    account_key: str
    source_timezone: str
    order_policy: str
    senders: Mapping[str, str]
    canonical: bytes
    sha256: str

    @property
    def source_namespace(self) -> str:
        return identity.source_namespace(self.platform, self.account_key)


def load_profile(path: Path) -> Profile:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ConfigError("profile.unreadable", "cannot read the profile file") from exc
    try:
        raw = loads_strict(data, what="profile")
        return _validate(raw)
    except EvidenceError as exc:
        raise ConfigError(exc.code, exc.message, details=exc.details) from None


def _validate(raw: object) -> Profile:
    if not isinstance(raw, dict):
        raise InvalidInput("profile.not_object", "the profile must be a JSON object")
    missing = sorted(_PROFILE_KEYS - raw.keys())
    unknown = sorted(raw.keys() - _PROFILE_KEYS)
    if missing or unknown:
        raise InvalidInput("profile.fields", "the profile has missing or unknown fields", details={"missing": missing, "unknown": unknown})
    if raw["profile_format"] != PROFILE_FORMAT or type(raw["profile_version"]) is not int or raw["profile_version"] != PROFILE_VERSION:
        raise InvalidInput("profile.format", "unsupported profile format or version")
    for key in ("adapter", "adapter_version", "account_key"):
        if not _clean_string(raw[key]):
            raise InvalidInput("profile.field_invalid", "profile fields must be nonempty strings without control characters", details={"field": key})
    if not isinstance(raw["platform"], str) or not _PLATFORM_RE.fullmatch(raw["platform"]):
        raise InvalidInput("profile.platform", "platform must be lowercase letters, digits, '_', '.' or '-'")
    load_zone(raw["source_timezone"])
    if raw["order_policy"] not in ORDER_BASES:
        raise InvalidInput("profile.order_policy", "order_policy must be 'native' or 'timestamp_only'")
    senders = raw["senders"]
    if not isinstance(senders, dict) or not senders:
        raise InvalidInput("profile.senders", "senders must map native sender keys to stable sender ids")
    for key, value in senders.items():
        if not key or not _clean_string(value):
            raise InvalidInput("profile.senders", "sender keys and ids must be nonempty strings without control characters")
    canonical = canonical_bytes(raw)
    return Profile(
        adapter=raw["adapter"],
        adapter_version=raw["adapter_version"],
        platform=raw["platform"],
        account_key=raw["account_key"],
        source_timezone=raw["source_timezone"],
        order_policy=raw["order_policy"],
        senders=MappingProxyType(dict(senders)),
        canonical=canonical,
        sha256=sha256_bytes(canonical),
    )


def _clean_string(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)
