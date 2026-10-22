"""Private directories and atomic file writes for dataset files."""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    restrict_permissions(path, PRIVATE_DIR_MODE)
    return path


def restrict_permissions(path: Path, mode: int) -> None:
    # POSIX permission bits only; on Windows os.chmod can only toggle read-only.
    if os.name == "posix":
        try:
            os.chmod(path, mode)
        except OSError:
            pass


def make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR)


def write_new_file(path: Path, data: bytes, *, mode: int = PRIVATE_FILE_MODE) -> None:
    """Atomically create ``path`` with ``data``; raise FileExistsError if it exists.

    Readers never observe a partially written file, and an existing file is
    never replaced, which keeps per-ID records immutable.
    """
    tmp = _write_temp(path, data, mode)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise
        except OSError:
            # Filesystem without hard links: reserve the name, then swap in the data.
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode))
            os.replace(tmp, path)
    finally:
        _unlink_quiet(tmp)


def replace_file(path: Path, data: bytes, *, mode: int = PRIVATE_FILE_MODE) -> None:
    """Atomically create or replace ``path`` with ``data``."""
    tmp = _write_temp(path, data, mode)
    try:
        os.replace(tmp, path)
    finally:
        _unlink_quiet(tmp)


def temp_sibling(path: Path, label: str) -> Path:
    return path.with_name(f".{path.name}.{label}-{secrets.token_hex(8)}.tmp")


def _write_temp(path: Path, data: bytes, mode: int) -> Path:
    tmp = temp_sibling(path, "write")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _unlink_quiet(tmp)
        raise
    return tmp


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
