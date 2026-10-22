"""Shared helpers for the Chat Evidence tests. All data here is invented."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

from chat_conductor.evidence import cli
from chat_conductor.evidence.importer import ingest
from chat_conductor.evidence.jsonutil import canonical_bytes, loads_strict, sha256_json
from chat_conductor.evidence.runs import Run, start_run
from chat_conductor.evidence.store import Dataset, init_dataset

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "evidence"
NAMESPACE = "synthetic:acct-1"


def profile_dict(**overrides: object) -> dict:
    profile = {
        "profile_format": "chat-evidence-profile",
        "profile_version": 1,
        "adapter": "synthetic-fixture",
        "adapter_version": "1",
        "platform": "synthetic",
        "account_key": "acct-1",
        "source_timezone": "Asia/Shanghai",
        "order_policy": "native",
        "senders": {"ana": "s_ana", "ben": "s_ben", "chen": "s_chen"},
    }
    profile.update(overrides)
    return profile


def fixture_dict(conversations: dict[str, list[dict]]) -> dict:
    return {
        "format": "chat-evidence-synthetic-fixture",
        "format_version": 1,
        "synthetic": True,
        "conversations": [
            {"conversation_id": conversation_id, "messages": messages}
            for conversation_id, messages in conversations.items()
        ],
    }


def message(
    native_id: str,
    sender: str,
    ts: str,
    text: str | None = None,
    *,
    kind: str = "text",
    order: str | None = None,
    transcript: str | None = None,
    placeholder: str | None = None,
) -> dict:
    item: dict = {"native_id": native_id, "sender": sender, "ts": ts, "kind": kind}
    if order is not None:
        item["order"] = order
    for key, value in (("text", text), ("transcript", transcript), ("placeholder", placeholder)):
        if value is not None:
            item[key] = value
    return item


def numbered(prefix: str, count: int, *, day: str = "2026-03-01", start_minute: int = 0, text: str = "{prefix} note {i}") -> list[dict]:
    """``count`` text messages one minute apart, with native order keys."""
    senders = ("ana", "ben", "chen")
    items = []
    for i in range(count):
        minute = start_minute + i
        ts = f"{day}T{8 + minute // 60:02d}:{minute % 60:02d}:00"
        items.append(message(f"{prefix}-{i:04d}", senders[i % 3], ts, text.format(prefix=prefix, i=i), order=f"{minute:06d}"))
    return items


def reseal(receipt: dict) -> dict:
    """Recompute a receipt digest after a deliberate edit (simulates tampering)."""
    body = {key: value for key, value in receipt.items() if key != "body_sha256"}
    body["body_sha256"] = sha256_json(body)
    return body


class EvidenceCase(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory(prefix="chat-evidence-test-", ignore_cleanup_errors=True)
        self.addCleanup(holder.cleanup)
        self.tmp = Path(holder.name).resolve()
        self.assertFalse(self.tmp.is_relative_to(ROOT), "tests must run outside the checkout")

    def new_dataset(self, name: str = "ds", dataset_id: str = "ds-test") -> Dataset:
        dataset = Dataset(self.tmp / name)
        init_dataset(dataset, requested_id=dataset_id)
        return dataset

    def write_json(self, name: str, value: object) -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def profile(self, name: str = "profile.json", **overrides: object) -> Path:
        return self.write_json(name, profile_dict(**overrides))

    def ingest(self, dataset: Dataset, conversations: dict[str, list[dict]], *, name: str = "export.json", profile: Path | None = None) -> dict:
        export = self.write_json(name, fixture_dict(conversations))
        return ingest(dataset, export, adapter_name="synthetic-fixture", profile_path=profile or self.profile())

    def run_for(self, dataset: Dataset, question: str = "What was agreed?", revision: int | None = None) -> Run:
        return start_run(dataset, question, revision=revision)

    def read_receipt(self, run: Run, receipt_id: str) -> dict:
        return loads_strict((run.receipts_dir / f"{receipt_id}.json").read_bytes(), what="receipt")

    def write_receipt_raw(self, run: Run, receipt: dict) -> None:
        path = run.receipts_dir / f"{receipt['receipt_id']}.json"
        path.unlink(missing_ok=True)
        path.write_bytes(canonical_bytes(receipt) + b"\n")

    def cli(self, *args: object) -> tuple[int, str, str]:
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main([str(arg) for arg in args])
        stdout.flush()
        return code, stdout.buffer.getvalue().decode("utf-8"), stderr.getvalue()
