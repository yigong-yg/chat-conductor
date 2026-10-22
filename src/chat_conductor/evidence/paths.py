"""Dataset location, identifier patterns and path-containment checks."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from .errors import ConfigError, InvalidInput

DATASET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
RUN_ID_RE = re.compile(r"run_[A-Za-z0-9_-]{1,120}")
RECEIPT_ID_RE = re.compile(r"q_[A-Za-z0-9_-]{1,120}")
CLAIM_ID_RE = re.compile(r"c_[A-Za-z0-9_-]+")
MESSAGE_ID_RE = re.compile(r"m_[0-9a-f]{64}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")


def default_evidence_root() -> Path:
    """Per-user data directory that holds one subdirectory per dataset."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise ConfigError(
                "dataset.no_user_data_dir",
                "LOCALAPPDATA is not set; pass --dataset with an absolute directory",
            )
        return Path(base) / "chat-conductor" / "evidence"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "chat-conductor" / "evidence"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local" / "share"
    return base / "chat-conductor" / "evidence"


def find_git_worktree(path: Path) -> Path | None:
    """Return the nearest ancestor of the resolved path that holds a ``.git`` entry.

    Both a ``.git`` directory (main worktree) and a ``.git`` file (linked
    worktree or submodule) count. Links are resolved first, so a symlink that
    points into a worktree is detected.
    """
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        try:
            os.lstat(candidate / ".git")
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            raise ConfigError(
                "dataset.path_unverifiable",
                "cannot inspect a parent directory for a Git worktree",
            ) from exc
        return candidate
    return None


def resolve_dataset_dir(dataset: str | None, dataset_id: str | None) -> tuple[Path, str | None]:
    """Resolve an explicit dataset selection to a directory outside every worktree.

    Returns the resolved directory and, for ``--dataset-id`` selection, the
    dataset id the directory must hold.
    """
    if (dataset is None) == (dataset_id is None):
        raise ConfigError(
            "dataset.selection",
            "select exactly one dataset with --dataset <absolute-directory> or --dataset-id <id>",
        )
    if dataset is not None:
        candidate = Path(dataset).expanduser()
        if not candidate.is_absolute():
            raise ConfigError("dataset.relative_path", "--dataset must be an absolute directory")
        expected_id = None
    else:
        assert dataset_id is not None
        if not DATASET_ID_RE.fullmatch(dataset_id):
            raise ConfigError(
                "dataset.invalid_id",
                "dataset ids use 1-64 letters, digits, '_' or '-' and start with a letter or digit",
            )
        candidate = default_evidence_root() / dataset_id
        expected_id = dataset_id
    resolved = candidate.resolve()
    if find_git_worktree(resolved) is not None:
        raise ConfigError(
            "dataset.inside_worktree",
            "dataset directories and every derivative must resolve outside all Git worktrees",
        )
    return resolved, expected_id


def contained_path(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and require the resolved result to stay inside it."""
    root_resolved = root.resolve()
    resolved = root_resolved.joinpath(*parts).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise InvalidInput("path.escape", "a referenced file resolves outside its directory")
    return resolved
