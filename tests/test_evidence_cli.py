from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

import _path  # noqa: F401

from evidence_demo import Demo
from evidence_support import ROOT, EvidenceCase


class EvidenceCliTests(EvidenceCase):
    def test_python_m_entry_point_runs_the_synthetic_demonstration(self) -> None:
        lines: list[str] = []
        summary = Demo(self.tmp / "demo-dataset", echo=lines.append).run()
        self.assertEqual(summary["revisions"], (1, 2))
        self.assertEqual(summary["pinned_revision"], 1)
        self.assertEqual(summary["matches"], 2)
        self.assertEqual(summary["window"], (20, 20, True))
        self.assertEqual(summary["count"], 7)
        self.assertEqual(summary["absent"], 0)
        self.assertEqual(summary["verified"], "pass")
        self.assertEqual(summary["altered"], ["quote.text_mismatch"])
        self.assertEqual(summary["check_after_alteration"], "not_verified")
        self.assertEqual(summary["exit_codes"], {
            "init": 0, "ingest-1": 0, "run": 0, "ingest-2": 0, "adapter-gate": 3, "search": 0, "ctx": 0,
            "range": 0, "sql": 0, "absent": 0, "verify": 0, "check": 0, "verify-altered": 2, "check-altered": 2,
        })
        self.assertTrue(any("q sql" in line for line in lines))

    def test_demo_script_runs_standalone(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "evidence_demo.py"), "--dataset", str(self.tmp / "standalone")],
            capture_output=True,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8"))
        summary = json.loads(completed.stdout.decode("utf-8")[completed.stdout.decode("utf-8").index("{\n"):])
        self.assertEqual(summary["verified"], "pass")

    def test_usage_errors_and_dataset_selection(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.cli("--dataset", self.tmp / "x", "q", "search", "--run", "run_x")
        self.assertEqual(caught.exception.code, 2)
        exit_code, _, stderr = self.cli("run", "start", "--question", "q")
        self.assertEqual(exit_code, 3)
        self.assertIn("dataset.selection", stderr)
        exit_code, _, stderr = self.cli("--dataset", self.tmp / "missing", "q", "ctx", "--run", "run_x", "m_" + "0" * 64)
        self.assertEqual(exit_code, 3)
        self.assertIn("dataset.not_initialized", stderr)


if __name__ == "__main__":
    unittest.main()
