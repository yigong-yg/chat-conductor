from __future__ import annotations

import json
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def conversations_json_from_export(source: Path) -> Iterator[Path]:
    source = source.expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="chat-conductor-export-") as tmp:
            dest = Path(tmp)
            with zipfile.ZipFile(source) as archive:
                _safe_extract(archive, dest)
            yield _find_conversations_json(dest)
        return

    if source.is_dir():
        yield _find_conversations_json(source)
        return

    if source.is_file() and source.name == "conversations.json":
        yield source
        return

    raise FileNotFoundError(f"expected a .zip export, export directory, or conversations.json: {source}")


def load_conversations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected top-level conversation list in {path}")
    return data


def _find_conversations_json(root: Path) -> Path:
    direct = root / "conversations.json"
    if direct.is_file():
        return direct
    matches = sorted(root.glob("**/conversations.json"))
    if not matches:
        raise FileNotFoundError(f"could not locate conversations.json under {root}")
    if len(matches) > 1:
        joined = ", ".join(str(match) for match in matches[:5])
        raise ValueError(f"found multiple conversations.json files under {root}: {joined}")
    return matches[0]


def _safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        try:
            target.relative_to(dest)
        except ValueError as exc:
            raise ValueError(f"refusing unsafe zip member path: {member.filename}") from exc
        archive.extract(member, dest)
