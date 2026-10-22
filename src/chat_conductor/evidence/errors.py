"""Error types and process exit codes for the evidence tools.

Messages are written for the local operator. They name fields, counts and
stable codes, never source text, query terms, SQL parameters or private
identities; details of that kind belong in private files inside the dataset.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_RUNTIME = 3


class EvidenceError(Exception):
    """Base error carrying a stable machine-readable code."""

    exit_code = EXIT_RUNTIME

    def __init__(self, code: str, message: str, *, details: dict | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        payload: dict = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class InvalidInput(EvidenceError):
    """Invalid evidence, draft, query input, or a rejected import batch (exit 2)."""

    exit_code = EXIT_INVALID


class ConfigError(EvidenceError):
    """Runtime, schema, storage or configuration failure (exit 3)."""

    exit_code = EXIT_RUNTIME


class BudgetExceeded(ConfigError):
    """A row, byte, step, time or size budget stopped an operation (exit 3)."""
