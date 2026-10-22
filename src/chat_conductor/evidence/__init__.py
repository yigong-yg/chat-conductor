"""Chat Evidence: local, source-backed retrieval and citation verification (prototype).

Separate from the turn-oriented ``chat_conductor`` CLI and its index. Entry point::

    python -m chat_conductor.evidence --dataset <absolute-directory> <command> ...

Commands: ``init``, ``ingest``, ``run start``, ``q search``, ``q ctx``,
``q range``, ``q sql`` and ``verify``. Only the explicitly synthetic
``synthetic-fixture`` adapter exists; a real export adapter awaits an
inspected, redacted export contract.
"""

from .. import __version__

TOOL_NAME = "chat-conductor-evidence"
TOOL_VERSION = __version__

__all__ = ["TOOL_NAME", "TOOL_VERSION"]
