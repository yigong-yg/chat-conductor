"""Timestamps, explicit IANA zones and the canonical UTC format.

Canonical UTC instants are fixed-width ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` strings,
so lexical order equals chronological order. Local wall times are resolved
only in an explicitly declared IANA zone; ambiguous or skipped local times
fail instead of guessing, and the operator's own zone is never a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from .errors import ConfigError, InvalidInput

CANONICAL_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z")
_ZONE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*(?:/[A-Za-z0-9_+-]+)*")
# Names that resolve to the host's configuration rather than a declared zone.
_HOST_ZONE_NAMES = frozenset({"localtime", "posixrules", "Factory"})
_SOURCE_TIMESTAMP_RE = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?"
    r"(?P<offset>Z|[+-][0-9]{2}:[0-9]{2})?"
)
_LOCAL_BOUND_RE = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"(?:T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})(?:\.(?P<fraction>[0-9]{1,6}))?)?)?"
)


def load_zone(name: object) -> ZoneInfo:
    if not isinstance(name, str) or not _ZONE_NAME_RE.fullmatch(name) or name in _HOST_ZONE_NAMES:
        raise InvalidInput("time.zone_invalid", "time zones must be explicit IANA names such as Asia/Shanghai")
    return _load_zone(name)


@lru_cache(maxsize=64)
def _load_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        if not available_timezones():
            raise ConfigError(
                "time.tzdata_missing",
                "no IANA time zone database is available; install the tzdata package",
            ) from None
        raise InvalidInput("time.zone_unknown", "unknown IANA time zone") from None


def format_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("format_utc requires an aware datetime")
    value = moment.astimezone(UTC)
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}T"
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}.{value.microsecond:06d}Z"
    )


def utc_now() -> str:
    return format_utc(datetime.now(UTC))


def is_canonical_utc(value: object) -> bool:
    if not isinstance(value, str) or not CANONICAL_UTC_RE.fullmatch(value):
        return False
    try:
        _datetime_from_parts(value[0:4], value[5:7], value[8:10], value[11:13], value[14:16], value[17:19], value[20:26])
    except ValueError:
        return False
    return True


def resolve_source_timestamp(original: str, zone_name: str) -> str:
    """Convert an exported ISO 8601 timestamp to canonical UTC.

    A timestamp without an offset is a wall time in ``zone_name``. A timestamp
    with ``Z`` or ``+HH:MM`` must carry the declared zone's offset at that
    instant; otherwise the declaration and the data disagree.
    """
    match = _SOURCE_TIMESTAMP_RE.fullmatch(original) if isinstance(original, str) else None
    if match is None:
        raise InvalidInput(
            "time.unparseable",
            "timestamps must be ISO 8601 date-times with seconds and at most microsecond precision",
        )
    zone = load_zone(zone_name)
    fraction = match["fraction"] or ""
    try:
        naive = _datetime_from_parts(
            match["year"], match["month"], match["day"],
            match["hour"], match["minute"], match["second"], fraction.ljust(6, "0") if fraction else "0",
        )
    except ValueError:
        raise InvalidInput("time.invalid_date", "the timestamp is not a valid calendar date and time") from None
    offset = match["offset"]
    if offset is None:
        aware = resolve_wall_time(naive, zone)
    else:
        delta = _parse_offset(offset)
        aware = naive.replace(tzinfo=timezone(delta))
        if aware.astimezone(zone).utcoffset() != delta:
            raise InvalidInput(
                "time.offset_mismatch",
                "the timestamp's UTC offset differs from the declared source zone at that instant",
            )
    return format_utc(aware)


def resolve_wall_time(naive: datetime, zone: ZoneInfo) -> datetime:
    early = naive.replace(tzinfo=zone, fold=0)
    late = naive.replace(tzinfo=zone, fold=1)
    if early.utcoffset() != late.utcoffset():
        if early.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive:
            raise InvalidInput(
                "time.ambiguous_local",
                "the local time occurs twice in the declared zone (clock change)",
            )
        raise InvalidInput(
            "time.nonexistent_local",
            "the local time does not exist in the declared zone (clock change)",
        )
    return early


@dataclass(frozen=True)
class LocalBound:
    local: str
    utc: str
    kind: str  # "date" or "datetime"

    def to_dict(self) -> dict:
        return {"local": self.local, "utc": self.utc, "kind": self.kind}


def local_bound(value: str, zone: ZoneInfo, *, label: str) -> LocalBound:
    match = _LOCAL_BOUND_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise InvalidInput(
            "time.bound_format",
            f"--{label} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM[:SS[.ffffff]] local time without an offset",
        )
    fraction = match["fraction"] or ""
    try:
        naive = _datetime_from_parts(
            match["year"], match["month"], match["day"],
            match["hour"] or "0", match["minute"] or "0", match["second"] or "0",
            fraction.ljust(6, "0") if fraction else "0",
        )
    except ValueError:
        raise InvalidInput("time.invalid_date", f"--{label} is not a valid calendar date and time") from None
    aware = resolve_wall_time(naive, zone)
    return LocalBound(local=value, utc=format_utc(aware), kind="date" if match["hour"] is None else "datetime")


def convert_window(from_value: str | None, to_value: str | None, zone_name: str | None) -> dict:
    """Convert optional half-open local bounds ``[from, to)`` to UTC once.

    A date-only value denotes the start (local midnight) of that date, so a
    date-only end excludes that whole date. Missing bounds are unbounded and
    labelled as such; they never assert anything about corpus completeness.
    """
    if from_value is None and to_value is None:
        if zone_name is not None:
            load_zone(zone_name)
        return {
            "zone": zone_name,
            "interval": "half-open [from, to)",
            "from": None,
            "to": None,
            "unbounded": ["from", "to"],
        }
    if zone_name is None:
        raise InvalidInput("time.zone_required", "--tz with an explicit IANA zone is required with --from/--to")
    zone = load_zone(zone_name)
    lower = local_bound(from_value, zone, label="from") if from_value is not None else None
    upper = local_bound(to_value, zone, label="to") if to_value is not None else None
    if lower is not None and upper is not None and lower.utc >= upper.utc:
        raise InvalidInput("time.empty_window", "--from must be earlier than --to")
    window = {
        "zone": zone_name,
        "interval": "half-open [from, to)",
        "from": lower.to_dict() if lower else None,
        "to": upper.to_dict() if upper else None,
        "unbounded": [name for name, bound in (("from", lower), ("to", upper)) if bound is None],
    }
    if upper is not None and upper.kind == "date":
        window["date_only_end"] = "start of that date; the date itself is excluded"
    return window


def _datetime_from_parts(year: str, month: str, day: str, hour: str, minute: str, second: str, micro: str) -> datetime:
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), int(micro))


def _parse_offset(offset: str) -> timedelta:
    if offset == "Z":
        return timedelta(0)
    hours, minutes = int(offset[1:3]), int(offset[4:6])
    if hours > 23 or minutes > 59:
        raise InvalidInput("time.invalid_offset", "the UTC offset is out of range")
    delta = timedelta(hours=hours, minutes=minutes)
    return -delta if offset[0] == "-" else delta
