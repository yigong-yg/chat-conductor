"""Synthetic end-to-end demonstration of the Chat Evidence prototype.

    python tests/evidence_demo.py --dataset <new absolute directory outside every Git worktree>

Uses only the committed, invented fixtures in tests/fixtures/evidence. Every
step runs the real entry point (``python -m chat_conductor.evidence``) in a
subprocess and prints the command, its exit code and a short summary:

    init -> ingest export 1 -> run start (pins revision 1) -> ingest export 2
    -> q search / q ctx / q range / q sql -> three-claim draft
    -> verify (exit 0) -> verify --check (exit 0)
    -> altered draft: verify (exit 2) -> verify --check (exit 2)

This checks evidence-integrity behaviour on synthetic data. It is not a
retrieval-quality or answer-accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "evidence"
QUESTION = "Who keeps the shed key, and how often did Ana mention watering?"
QUOTE = "the shed key stays with Ben until 4月10日"
NAMESPACE = "synthetic:demo-account"


class DemoFailure(RuntimeError):
    pass


class Demo:
    def __init__(self, dataset: Path, *, echo=print) -> None:
        self.dataset = dataset
        self.echo = echo
        python_path = [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
        self.env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}
        self.exit_codes: dict[str, int] = {}

    def evidence(self, label: str, *args: object, expect: int = 0) -> dict:
        arguments = [str(arg) for arg in args]
        self.echo(f"$ python -m chat_conductor.evidence --dataset <dataset> {shlex.join(arguments)}")
        completed = subprocess.run(
            [sys.executable, "-m", "chat_conductor.evidence", "--dataset", str(self.dataset), *arguments],
            capture_output=True,
            env=self.env,
        )
        self.exit_codes[label] = completed.returncode
        stdout = completed.stdout.decode("utf-8").strip()
        stderr = completed.stderr.decode("utf-8").strip()
        self.echo(f"  -> exit {completed.returncode}")
        if completed.returncode != expect:
            raise DemoFailure(f"{label}: expected exit {expect}, got {completed.returncode}: {stderr[:500]}")
        return json.loads(stdout) if stdout else json.loads(stderr)

    def run(self) -> dict:
        fixtures = FIXTURES
        profile = fixtures / "demo-profile.json"
        created = self.evidence("init", "init", "--id", "demo-garden")
        self.echo(f"  dataset {created['dataset_id']} (schema v{created['schema_version']})")
        first = self.evidence("ingest-1", "ingest", fixtures / "demo-export-1.json", "--adapter", "synthetic-fixture", "--profile", profile)
        self.echo(f"  revision {first['revision']}: {first['counts']['new']} new messages")
        run = self.evidence("run", "run", "start", "--question", QUESTION)
        run_id = run["run_id"]
        self.echo(f"  {run_id} pinned to revision {run['corpus_revision']}")
        second = self.evidence("ingest-2", "ingest", fixtures / "demo-export-2.json", "--adapter", "synthetic-fixture", "--profile", profile)
        self.echo(f"  revision {second['revision']}: {second['counts']['new']} new, {second['counts']['overlap_existing']} overlapping (provenance only)")
        gate = self.evidence("adapter-gate", "ingest", fixtures / "demo-export-1.json", "--adapter", "wechat", "--profile", profile, expect=3)
        self.echo(f"  {gate['error']['code']}: real adapters await a redacted export contract")

        found = self.evidence("search", "q", "search", "--run", run_id, "shed key")
        target = found["rows"][0]
        self.echo(f"  {found['total_matches']} matches at the pinned revision ({found['status']}); target {target['id'][:14]}...")
        window = self.evidence("ctx", "q", "ctx", "--run", run_id, target["id"])
        info = window["window"]
        self.echo(f"  context {info['before_count']} before / {info['after_count']} after, complete={window['context_complete']}")
        ranged = self.evidence(
            "range", "q", "range", "--run", run_id, "--source-namespace", NAMESPACE, "--conversation", "conv-garden",
            "--from", "2026-03-02", "--to", "2026-03-03", "--tz", "Asia/Shanghai", "--limit", "5",
        )
        self.echo(f"  range page: {ranged['row_count']} of {ranged['total_matches']} ({ranged['status']}, has_more={ranged['has_more']})")
        counted = self.evidence(
            "sql", "q", "sql", "--run", run_id,
            "--sql-file", fixtures / "demo-count.sql",
            "--params-file", fixtures / "demo-count.params.json",
            "--scope-file", fixtures / "demo-count.scope.json",
            "--utc-param", "from_utc=2026-03-01", "--utc-param", "to_utc=2026-03-04", "--tz", "Asia/Shanghai",
        )
        self.echo(f"  message_count = {counted['rows'][0][0]} (complete={counted['complete']})")
        absent = self.evidence("absent", "q", "search", "--run", run_id, "lawyer", "--source-namespace", NAMESPACE, "--conversation", "conv-garden")
        self.echo(f"  'lawyer': {absent['total_matches']} matches ({absent['status']})")

        start = target["text"].index(QUOTE)
        draft = {
            "schema_version": 1,
            "dataset_id": created["dataset_id"],
            "run_id": run_id,
            "corpus_revision": run["corpus_revision"],
            "question": QUESTION,
            "claims": [
                {
                    "id": "c_quote_1", "type": "quote", "message_id": target["id"], "sender_id": target["sender_id"],
                    "ts_utc": target["ts_utc"], "start": start, "end": start + len(QUOTE), "text": QUOTE,
                    "context_receipt_id": window["receipt_id"],
                },
                {"id": "c_count_1", "type": "statistic", "receipt_id": counted["receipt_id"], "row": 0, "column": "message_count", "value": counted["rows"][0][0]},
                {"id": "c_absent_1", "type": "not_found", "receipt_ids": [absent["receipt_id"]]},
            ],
        }
        draft_path = Path(run["draft_path"])
        draft_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verified = self.evidence("verify", "verify", "--run", run_id)
        self.echo(f"  {verified['status']}; report {verified['report']}")
        checked = self.evidence("check", "verify", "--run", run_id, "--check")
        self.echo(f"  report is {checked['status']}")

        altered = json.loads(json.dumps(draft))
        altered["claims"][0]["text"] = QUOTE.replace("Ben", "Bob")
        draft_path.write_text(json.dumps(altered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rejected = self.evidence("verify-altered", "verify", "--run", run_id, expect=2)
        self.echo(f"  {rejected['status']}: {[failure['code'] for failure in rejected['failures']]}")
        stale = self.evidence("check-altered", "verify", "--run", run_id, "--check", expect=2)
        self.echo(f"  report is {stale['status']}: {stale['problems']}")
        return {
            "run_id": run_id,
            "revisions": (first["revision"], second["revision"]),
            "pinned_revision": run["corpus_revision"],
            "matches": found["total_matches"],
            "window": (info["before_count"], info["after_count"], window["context_complete"]),
            "count": counted["rows"][0][0],
            "absent": absent["total_matches"],
            "verified": verified["status"],
            "altered": [failure["code"] for failure in rejected["failures"]],
            "check_after_alteration": stale["status"],
            "exit_codes": dict(self.exit_codes),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, type=Path, help="new absolute directory outside every Git worktree")
    args = parser.parse_args(argv)
    try:
        summary = Demo(args.dataset).run()
    except DemoFailure as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
