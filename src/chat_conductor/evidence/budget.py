"""Execution budgets enforced through SQLite's progress handler."""

from __future__ import annotations

import sqlite3
import time

PROGRESS_INTERVAL = 1000

# Indirection so tests can simulate elapsed time deterministically.
clock = time.monotonic


class Budget:
    """Abort a statement once a VM-step count or a wall-clock deadline is exceeded."""

    def __init__(self, *, deadline_ms: int, max_steps: int) -> None:
        self.deadline_ms = deadline_ms
        self.max_steps = max_steps
        self.started = clock()
        self.steps = 0
        self.reason: str | None = None

    def __call__(self) -> int:
        self.steps += PROGRESS_INTERVAL
        if self.steps > self.max_steps:
            self.reason = "steps"
            return 1
        if self.expired():
            self.reason = "deadline"
            return 1
        return 0

    def expired(self) -> bool:
        return (clock() - self.started) * 1000 > self.deadline_ms

    def elapsed_ms(self) -> int:
        return int((clock() - self.started) * 1000)

    def install(self, connection: sqlite3.Connection) -> None:
        connection.set_progress_handler(self, PROGRESS_INTERVAL)
