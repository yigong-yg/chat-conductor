"""Content-addressed preservation of export bytes under ``raw/<export-digest>/``.

The export digest is the SHA-256 of a canonical manifest listing every file's
relative path, size and SHA-256. Preserved files are read-only; adapters parse
the preserved copy, and every read re-checks the manifest digest.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError, InvalidInput
from .fsutil import ensure_private_dir, make_read_only
from .jsonutil import canonical_bytes, sha256_bytes
from .paths import contained_path
from .store import Dataset

RAW_MANIFEST_FORMAT = "chat-evidence-raw-manifest"
RAW_MANIFEST_VERSION = 1
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class RawFileEntry:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class PreservedExport:
    export_sha256: str
    directory: Path
    manifest_relpath: str
    files: tuple[RawFileEntry, ...]

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(entry.path for entry in self.files)

    def read_bytes(self, relpath: str) -> bytes:
        entry = next((item for item in self.files if item.path == relpath), None)
        if entry is None:
            raise ConfigError("raw.unknown_file", "the file is not in the raw manifest")
        data = contained_path(self.directory / "files", *relpath.split("/")).read_bytes()
        if sha256_bytes(data) != entry.sha256 or len(data) != entry.size:
            raise ConfigError("raw.tampered", "a preserved raw file no longer matches its manifest")
        return data


def preserve_export(dataset: Dataset, source: Path) -> PreservedExport:
    try:
        resolved = Path(source).resolve(strict=True)
    except OSError:
        raise ConfigError("ingest.export_missing", "the export path does not exist") from None
    root = dataset.root.resolve()
    if resolved.is_relative_to(root) or root.is_relative_to(resolved):
        # Copying a tree that contains the dataset would also copy the staging area into itself.
        raise ConfigError("ingest.export_overlaps_dataset", "the export path and the dataset directory overlap")
    ensure_private_dir(dataset.raw_dir)
    staging = dataset.raw_dir / f".incoming-{secrets.token_hex(8)}"
    files_dir = ensure_private_dir(staging / "files")
    try:
        entries: list[RawFileEntry] = []
        if resolved.is_file():
            root_kind = "file"
            entries.append(_copy_into(resolved, files_dir, resolved.name))
        elif resolved.is_dir():
            root_kind = "directory"
            for path in _regular_files(resolved):
                entries.append(_copy_into(path, files_dir, path.relative_to(resolved).as_posix()))
        else:
            raise ConfigError("ingest.export_type", "the export must be a regular file or a directory")
        if not entries:
            raise InvalidInput("ingest.export_empty", "the export contains no files")
        entries.sort(key=lambda item: item.path)
        manifest = {
            "manifest_format": RAW_MANIFEST_FORMAT,
            "manifest_version": RAW_MANIFEST_VERSION,
            "root_kind": root_kind,
            "files": [entry.to_dict() for entry in entries],
        }
        manifest_bytes = canonical_bytes(manifest)
        export_sha256 = sha256_bytes(manifest_bytes)
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        make_read_only(manifest_path)
        target = dataset.raw_dir / export_sha256
        try:
            os.rename(staging, target)
        except OSError:
            if not target.is_dir():
                raise
            _verify_existing(target, manifest_bytes, entries)
            remove_tree(staging)
    except BaseException:
        if staging.exists():
            remove_tree(staging)
        raise
    return PreservedExport(
        export_sha256=export_sha256,
        directory=target,
        manifest_relpath=f"raw/{export_sha256}/manifest.json",
        files=tuple(entries),
    )


def remove_tree(path: Path) -> None:
    for root, dirs, files in os.walk(path):
        for name in (*dirs, *files):
            try:
                os.chmod(Path(root) / name, stat.S_IRWXU)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _regular_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in sorted(dirs):
            if (base / name).is_symlink():
                raise ConfigError("ingest.export_symlink", "the export directory contains a symbolic link")
        dirs.sort()
        for name in sorted(files):
            path = base / name
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                raise ConfigError("ingest.export_symlink", "the export directory contains a symbolic link")
            if not stat.S_ISREG(mode):
                raise ConfigError("ingest.export_special_file", "the export contains a non-regular file")
            found.append(path)
    return found


def _copy_into(source: Path, files_dir: Path, relpath: str) -> RawFileEntry:
    parts = relpath.split("/")
    if any(part in ("", ".", "..") or any(ord(ch) < 0x20 for ch in part) for part in parts):
        raise ConfigError("ingest.export_path", "the export contains an unsupported file name")
    destination = files_dir.joinpath(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(_CHUNK):
            digest.update(chunk)
            writer.write(chunk)
            size += len(chunk)
    make_read_only(destination)
    return RawFileEntry(path=relpath, sha256=digest.hexdigest(), size=size)


def _verify_existing(target: Path, manifest_bytes: bytes, entries: list[RawFileEntry]) -> None:
    try:
        existing = (target / "manifest.json").read_bytes()
    except OSError as exc:
        raise ConfigError("raw.tampered", "the preserved raw export has no manifest") from exc
    if existing != manifest_bytes:
        raise ConfigError("raw.tampered", "the preserved raw manifest differs from the export")
    for entry in entries:
        data = target.joinpath("files", *entry.path.split("/")).read_bytes()
        if sha256_bytes(data) != entry.sha256:
            raise ConfigError("raw.tampered", "a preserved raw file no longer matches its manifest")
