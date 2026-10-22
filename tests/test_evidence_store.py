from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import _path  # noqa: F401

from evidence_support import NAMESPACE, ROOT, EvidenceCase, fixture_dict, message, numbered, profile_dict

from chat_conductor.evidence import importer
from chat_conductor.evidence.errors import ConfigError, InvalidInput
from chat_conductor.evidence.identity import message_id, payload_sha256
from chat_conductor.evidence.importer import ImportRejected, ingest
from chat_conductor.evidence.paths import find_git_worktree
from chat_conductor.evidence.schema import APPLICATION_ID
from chat_conductor.evidence.store import Dataset, init_dataset, locate_dataset


def _ids(dataset: Dataset) -> dict[tuple[str, str, str], str]:
    connection = sqlite3.connect(dataset.db_path)
    try:
        rows = connection.execute("SELECT source_namespace, conversation_id, native_message_id, id FROM messages").fetchall()
    finally:
        connection.close()
    return {tuple(row[:3]): row[3] for row in rows}


def _scalar(dataset: Dataset, sql: str, params: tuple = ()) -> object:
    connection = sqlite3.connect(dataset.db_path)
    try:
        return connection.execute(sql, params).fetchone()[0]
    finally:
        connection.close()


class SchemaAndLocationTests(EvidenceCase):
    def test_init_creates_exactly_four_domain_tables_and_private_layout(self) -> None:
        dataset = self.new_dataset()
        connection = sqlite3.connect(dataset.db_path)
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
            self.assertEqual(tables, {"dataset", "import_batches", "messages", "message_origins"})
            self.assertLessEqual(
                {"messages_by_conversation_time", "messages_by_sender_time", "messages_by_revision", "origins_by_message", "origins_by_revision"},
                indexes,
            )
            self.assertEqual(connection.execute("PRAGMA application_id").fetchone()[0], APPLICATION_ID)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("SELECT singleton, dataset_id, schema_version FROM dataset").fetchall(), [(1, "ds-test", 1)])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM import_batches").fetchone()[0], 0)
        finally:
            connection.close()
        for name in ("raw", "profiles", "import-reports", "runs"):
            self.assertTrue((dataset.root / name).is_dir(), name)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(dataset.root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(dataset.db_path.stat().st_mode), 0o600)

    def test_init_refuses_a_nonempty_directory(self) -> None:
        target = self.tmp / "occupied"
        target.mkdir()
        (target / "index.sqlite3").write_bytes(b"")
        with self.assertRaises(ConfigError) as caught:
            init_dataset(Dataset(target))
        self.assertEqual(caught.exception.code, "dataset.not_empty")
        self.assertFalse((target / "dataset.sqlite3").exists())

    def test_refuses_turn_index_foreign_and_drifted_schemas(self) -> None:
        from chat_conductor.store import connect as connect_turn_index, init_schema as init_turn_index

        turn_dir = self.tmp / "turn-index"
        turn_dir.mkdir()
        turn_index = connect_turn_index(turn_dir / "dataset.sqlite3")
        init_turn_index(turn_index)
        turn_index.close()
        foreign_dir = self.tmp / "foreign"
        foreign_dir.mkdir()
        foreign = sqlite3.connect(foreign_dir / "dataset.sqlite3")
        foreign.execute("CREATE TABLE notes (x)")
        foreign.close()
        version_dataset = self.new_dataset("version", "ds-version")
        connection = sqlite3.connect(version_dataset.db_path)
        connection.execute("PRAGMA user_version = 2")
        connection.close()
        drift_dataset = self.new_dataset("drift", "ds-drift")
        connection = sqlite3.connect(drift_dataset.db_path)
        connection.execute("DROP TRIGGER messages_no_update")
        connection.close()
        cases = {
            turn_dir: "dataset.turn_index",
            foreign_dir: "dataset.not_evidence",
            version_dataset.root: "dataset.unsupported_version",
            drift_dataset.root: "dataset.schema_mismatch",
        }
        for directory, code in cases.items():
            with self.subTest(code=code):
                dataset = Dataset(directory)
                connection = dataset.connect_ro()
                try:
                    with self.assertRaises(ConfigError) as caught:
                        dataset.info(connection)
                finally:
                    connection.close()
                self.assertEqual(caught.exception.code, code)
                exit_code, _, stderr = self.cli("--dataset", directory, "run", "start", "--question", "q")
                self.assertEqual(exit_code, 3)
                self.assertIn(code, stderr)

    def test_worktree_paths_are_rejected_after_resolving_links(self) -> None:
        repo = self.tmp / "repo"
        (repo / ".git").mkdir(parents=True)
        linked = self.tmp / "linked-worktree"
        linked.mkdir()
        (linked / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n", encoding="utf-8")
        candidates = [repo / "data" / "ds", linked / "ds"]
        try:
            alias = self.tmp / "innocent-looking"
            alias.symlink_to(repo, target_is_directory=True)
            candidates.append(alias / "ds")
        except (OSError, NotImplementedError):
            pass
        if find_git_worktree(ROOT) is not None:
            candidates.append(ROOT / "private-evidence")
        for candidate in candidates:
            with self.subTest(candidate=candidate.name):
                with self.assertRaises(ConfigError) as caught:
                    locate_dataset(str(candidate), None)
                self.assertEqual(caught.exception.code, "dataset.inside_worktree")
                exit_code, stdout, stderr = self.cli("--dataset", candidate, "init")
                self.assertEqual((exit_code, stdout), (3, ""))
                self.assertIn("dataset.inside_worktree", stderr)
                self.assertFalse(candidate.exists())
        with self.assertRaises(ConfigError) as caught:
            locate_dataset("relative/ds", None)
        self.assertEqual(caught.exception.code, "dataset.relative_path")
        with self.assertRaises(ConfigError):
            locate_dataset(None, None)

    @unittest.skipIf(sys.platform in ("win32", "darwin"), "XDG data home applies to other platforms")
    def test_dataset_id_selection_uses_the_user_data_directory(self) -> None:
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(self.tmp / "xdg")}):
            exit_code, stdout, _ = self.cli("--dataset-id", "garden-1", "init")
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout)["dataset_id"], "garden-1")
            self.assertTrue((self.tmp / "xdg" / "chat-conductor" / "evidence" / "garden-1" / "dataset.sqlite3").is_file())
            exit_code, _, stderr = self.cli("--dataset-id", "garden-2", "init", "--id", "other")
            self.assertEqual(exit_code, 3)
            self.assertIn("dataset.id_mismatch", stderr)
            exit_code, _, stderr = self.cli("--dataset-id", "../escape", "init")
            self.assertEqual(exit_code, 3)
            self.assertIn("dataset.invalid_id", stderr)
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(ROOT / "ignored-data")}):
            if find_git_worktree(ROOT) is not None:
                exit_code, _, stderr = self.cli("--dataset-id", "inside", "init")
                self.assertEqual(exit_code, 3)
                self.assertIn("dataset.inside_worktree", stderr)


class IdentityTests(EvidenceCase):
    def test_message_id_is_the_versioned_native_identity_digest(self) -> None:
        identity = ["chat-evidence-id-v1", "ds-test", NAMESPACE, "conv-a", "n1"]
        expected = "m_" + hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(message_id("ds-test", NAMESPACE, "conv-a", "n1"), expected)
        self.assertRegex(expected, r"^m_[0-9a-f]{64}$")
        self.assertNotEqual(message_id("ds-other", NAMESPACE, "conv-a", "n1"), expected)

    def test_ids_are_stable_across_reorder_overlap_and_file_layout(self) -> None:
        first = [message(f"n{i}", "ana", f"2026-03-01T09:0{i}:00", f"text {i}", order=f"{i:03d}") for i in range(1, 5)]
        later = list(reversed(first[2:])) + [message("n5", "ben", "2026-03-01T09:05:00", "text 5", order="005")]
        dataset = self.new_dataset("a")
        self.ingest(dataset, {"conv-a": first}, name="first.json")
        report = self.ingest(dataset, {"conv-a": later}, name="nested/second-export.json")
        self.assertEqual((report["status"], report["revision"]), ("committed", 2))
        self.assertEqual(report["counts"]["new"], 1)
        self.assertEqual(report["counts"]["overlap_existing"], 2)
        ids_a = _ids(dataset)
        self.assertEqual(len(ids_a), 5)
        n3 = ids_a[(NAMESPACE, "conv-a", "n3")]
        connection = sqlite3.connect(dataset.db_path)
        origins = connection.execute(
            "SELECT import_revision, raw_locator FROM message_origins WHERE message_id = ? ORDER BY import_revision", (n3,)
        ).fetchall()
        connection.close()
        self.assertEqual(origins, [(1, "first.json#/conversations/0/messages/2"), (2, "second-export.json#/conversations/0/messages/1")])

        reverse = self.new_dataset("b")
        self.ingest(reverse, {"conv-a": later}, name="b-later.json")
        self.ingest(reverse, {"conv-a": first}, name="b-first.json")
        self.assertEqual(_ids(reverse), ids_a)
        for (namespace, conversation, native), value in ids_a.items():
            self.assertEqual(value, message_id("ds-test", namespace, conversation, native))

    def test_repeated_identical_messages_survive_with_a_shared_payload_hash(self) -> None:
        dataset = self.new_dataset()
        profile = self.profile(order_policy="timestamp_only")
        repeated = [message(f"same-{i}", "ben", "2026-03-01T09:10:00", "好的") for i in range(3)]
        report = self.ingest(dataset, {"conv-a": repeated}, profile=profile)
        self.assertEqual(report["counts"]["new"], 3)
        connection = sqlite3.connect(dataset.db_path)
        rows = connection.execute("SELECT id, payload_sha256, order_basis, source_order, text FROM messages").fetchall()
        connection.close()
        self.assertEqual(len({row[0] for row in rows}), 3)
        self.assertEqual(len({row[1] for row in rows}), 1)
        self.assertEqual({(row[2], row[3], row[4]) for row in rows}, {("timestamp_only", "", "好的")})

    def test_native_id_collisions_across_conversations_and_accounts_stay_distinct(self) -> None:
        dataset = self.new_dataset()
        self.ingest(dataset, {
            "conv-a": [message("n1", "ana", "2026-03-01T09:00:00", "hello", order="1")],
            "conv-b": [message("n1", "ana", "2026-03-01T09:00:00", "hello", order="1")],
        })
        other_account = self.profile("profile-2.json", account_key="acct-2")
        self.ingest(dataset, {"conv-a": [message("n1", "ana", "2026-03-01T09:00:00", "hello", order="1")]}, name="other.json", profile=other_account)
        ids = _ids(dataset)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids.values())), 3)
        self.assertIn(("synthetic:acct-2", "conv-a", "n1"), ids)

    def test_payload_hash_is_frozen_payload_v1(self) -> None:
        payload = ["payload-v1", "s_ana", "2026-03-01T09:00:00", "2026-03-01T01:00:00.000000Z", "Asia/Shanghai", "001", "native", "text", "export_text", "hi"]
        expected = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        actual = payload_sha256(
            sender_id="s_ana", ts_original="2026-03-01T09:00:00", ts_utc="2026-03-01T01:00:00.000000Z",
            source_timezone="Asia/Shanghai", source_order="001", order_basis="native", kind="text",
            text_origin="export_text", text="hi",
        )
        self.assertEqual(actual, expected)


class ImportTests(EvidenceCase):
    def test_identical_reimport_is_a_noop_and_new_files_add_provenance_only(self) -> None:
        dataset = self.new_dataset()
        conversations = {"conv-a": numbered("n", 4)}
        first = self.ingest(dataset, conversations)
        again = self.ingest(dataset, conversations)
        self.assertEqual((first["status"], first["revision"]), ("committed", 1))
        self.assertEqual((again["status"], again["revision"]), ("noop", 1))
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM import_batches"), 1)
        copy = self.ingest(dataset, conversations, name="renamed-copy.json")
        self.assertEqual((copy["status"], copy["revision"]), ("committed", 2))
        self.assertEqual((copy["counts"]["new"], copy["counts"]["overlap_existing"]), (0, 4))
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM messages"), 4)
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM message_origins"), 8)
        widened = self.profile("profile-wide.json", senders={"ana": "s_ana", "ben": "s_ben", "chen": "s_chen", "dee": "s_dee"})
        rebound = self.ingest(dataset, conversations, profile=widened)
        self.assertEqual((rebound["status"], rebound["revision"], rebound["counts"]["new"]), ("committed", 3, 0))

    def test_changed_payload_quarantines_the_entire_batch(self) -> None:
        dataset = self.new_dataset()
        original = numbered("n", 3)
        self.ingest(dataset, {"conv-a": original})
        changed = [dict(original[0], text="edited text"), *original[1:], *numbered("m", 2, start_minute=30)]
        export = self.write_json("changed.json", fixture_dict({"conv-a": changed}))
        with self.assertRaises(ImportRejected) as caught:
            ingest(dataset, export, adapter_name="synthetic-fixture", profile_path=self.profile())
        summary = caught.exception.details
        self.assertEqual(summary["problem_codes"], {"conflict.changed_payload": 1})
        self.assertNotIn("edited text", json.dumps(summary, ensure_ascii=False))
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM import_batches"), 1)
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM messages"), 3)
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM message_origins"), 3)
        report = json.loads(Path(summary["report"]).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["diagnostics"][0]["native_message_id"], "n-0000")
        self.assertEqual(len(list(dataset.raw_dir.iterdir())), 2)
        exit_code, stdout, stderr = self.cli("--dataset", dataset.root, "ingest", export, "--adapter", "synthetic-fixture", "--profile", self.profile())
        self.assertEqual((exit_code, stdout), (2, ""))
        self.assertIn("ingest.rejected", stderr)
        self.assertNotIn("edited text", stderr)

    def test_duplicates_inside_one_batch(self) -> None:
        dataset = self.new_dataset()
        conflicting = [message("n1", "ana", "2026-03-01T09:00:00", "one", order="1"), message("n1", "ana", "2026-03-01T09:00:00", "two", order="1")]
        with self.assertRaises(ImportRejected) as caught:
            self.ingest(dataset, {"conv-a": conflicting})
        self.assertEqual(caught.exception.details["problem_codes"], {"conflict.duplicate_changed_payload": 1})
        export_dir = self.tmp / "split-export"
        export_dir.mkdir()
        shared = numbered("n", 3)
        (export_dir / "part-1.json").write_text(json.dumps(fixture_dict({"conv-a": shared[:2]})), encoding="utf-8")
        (export_dir / "part-2.json").write_text(json.dumps(fixture_dict({"conv-a": shared[1:]})), encoding="utf-8")
        report = ingest(dataset, export_dir, adapter_name="synthetic-fixture", profile_path=self.profile())
        self.assertEqual(report["counts"]["records"], 4)
        self.assertEqual(report["counts"]["new"], 3)
        self.assertEqual(report["counts"]["overlap_in_batch"], 1)
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM message_origins"), 4)
        self.assertEqual(_scalar(dataset, "SELECT record_count - appended_count FROM import_batches"), 1)

    def test_invalid_records_reject_the_batch_with_private_diagnostics(self) -> None:
        dataset = self.new_dataset()
        profile = self.profile(source_timezone="America/New_York")
        secret = "private words that must not reach stderr"
        records = [
            message("ok-1", "ana", "2026-03-01T09:00:00", secret, order="01"),
            message("bad-sender", "zed", "2026-03-01T09:01:00", "x", order="02"),
            message("bad-kind", "ana", "2026-03-01T09:02:00", "x", kind="sticker", order="03"),
            {"native_id": "no-ts", "sender": "ana", "kind": "text", "text": "x", "order": "04"},
            message("ambiguous", "ana", "2026-11-01T01:30:00", "x", order="05"),
            message("skipped", "ana", "2026-03-08T02:30:00", "x", order="06"),
            message("date-only", "ana", "2026-03-01", "x", order="07"),
            message("offset", "ana", "2026-03-01T09:00:00+08:00", "x", order="08"),
            message("too-precise", "ana", "2026-03-01T09:00:00.1234567", "x", order="09"),
            message("no-order", "ana", "2026-03-01T09:00:00", "x"),
            {**message("two-texts", "ana", "2026-03-01T09:00:00", "x", order="10"), "placeholder": "[x]"},
            message("text-placeholder", "ana", "2026-03-01T09:00:00", kind="text", placeholder="[x]", order="11"),
            message("image-transcript", "ana", "2026-03-01T09:00:00", kind="image", transcript="x", order="12"),
            message("nul", "ana", "2026-03-01T09:00:00", "a\x00b", order="13"),
            {**message("extra", "ana", "2026-03-01T09:00:00", "x", order="14"), "display_name": "Ana"},
        ]
        export = self.write_json("invalid.json", fixture_dict({"conv-a": records}))
        exit_code, stdout, stderr = self.cli("--dataset", dataset.root, "ingest", export, "--adapter", "synthetic-fixture", "--profile", profile)
        self.assertEqual((exit_code, stdout), (2, ""))
        self.assertNotIn(secret, stderr)
        codes = json.loads(stderr)["error"]["details"]["problem_codes"]
        for code in (
            "fixture.unknown_sender", "fixture.kind_unknown", "time.missing", "time.ambiguous_local",
            "time.nonexistent_local", "time.unparseable", "time.offset_mismatch", "fixture.native_order_missing",
            "fixture.text_field_count", "record.text_kind_requires_export_text", "record.transcript_kind",
            "record.nul_character", "fixture.unknown_field",
        ):
            self.assertIn(code, codes)
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM import_batches"), 0)
        self.assertEqual(_scalar(dataset, "SELECT COUNT(*) FROM messages"), 0)

    def test_zones_must_be_explicit_iana_names(self) -> None:
        dataset = self.new_dataset()
        for zone in ("localtime", "+08:00", "Mars/Olympus_Mons", ""):
            with self.subTest(zone=zone):
                with self.assertRaises(ConfigError) as caught:
                    self.ingest(dataset, {"conv-a": numbered("n", 1)}, profile=self.profile(f"profile-{len(zone)}.json", source_timezone=zone))
                self.assertTrue(caught.exception.code.startswith("time."), caught.exception.code)

    def test_only_documents_declaring_themselves_synthetic_are_parsed(self) -> None:
        dataset = self.new_dataset()
        document = fixture_dict({"conv-a": numbered("n", 2)})
        document["synthetic"] = False
        export = self.write_json("real-looking.json", document)
        with self.assertRaises(ImportRejected) as caught:
            ingest(dataset, export, adapter_name="synthetic-fixture", profile_path=self.profile())
        self.assertEqual(caught.exception.details["problem_codes"], {"fixture.not_synthetic_fixture": 1})
        notes = self.tmp / "with-notes"
        notes.mkdir()
        (notes / "chat.json").write_text(json.dumps(fixture_dict({"conv-a": numbered("n", 1)})), encoding="utf-8")
        (notes / "media.bin").write_bytes(b"\x00\x01")
        with self.assertRaises(ImportRejected) as caught:
            ingest(dataset, notes, adapter_name="synthetic-fixture", profile_path=self.profile())
        self.assertEqual(caught.exception.details["problem_codes"], {"fixture.unsupported_file": 1})

    def test_exports_overlapping_the_dataset_are_refused(self) -> None:
        parent = self.tmp / "exports"
        parent.mkdir()
        (parent / "chat.json").write_text(json.dumps(fixture_dict({"conv-a": numbered("n", 1)})), encoding="utf-8")
        dataset = Dataset(parent / "dataset")
        init_dataset(dataset, requested_id="ds-test")
        inside = dataset.root / "dropped-export.json"
        inside.write_text(json.dumps(fixture_dict({"conv-a": numbered("n", 1)})), encoding="utf-8")
        for export in (parent, inside):
            with self.subTest(export=export.name):
                with self.assertRaises(ConfigError) as caught:
                    ingest(dataset, export, adapter_name="synthetic-fixture", profile_path=self.profile())
                self.assertEqual(caught.exception.code, "ingest.export_overlaps_dataset")
        self.assertEqual([path.name for path in dataset.raw_dir.iterdir()], [])

    def test_real_export_adapters_are_not_available(self) -> None:
        dataset = self.new_dataset()
        export = self.write_json("export.json", fixture_dict({"conv-a": numbered("n", 1)}))
        for adapter in ("wechat", "claude", "whatsapp"):
            with self.subTest(adapter=adapter):
                with self.assertRaises(ConfigError) as caught:
                    ingest(dataset, export, adapter_name=adapter, profile_path=self.profile())
                self.assertEqual(caught.exception.code, "ingest.adapter_unavailable")
        exit_code, _, stderr = self.cli("--dataset", dataset.root, "ingest", export, "--adapter", "wechat", "--profile", self.profile())
        self.assertEqual(exit_code, 3)
        self.assertIn("redacted export contract", stderr)

    def test_exact_text_and_raw_bytes_are_preserved(self) -> None:
        dataset = self.new_dataset()
        texts = [
            "a  b\r\nc\t",
            " leading and trailing  ",
            "café",
            "café",
            "👨‍👩‍👧 family",
            "‏right-to-left mark",
            "line\n\nbreaks\n",
            "全角，标点。",
        ]
        records = [message(f"t{i}", "ana", f"2026-03-01T09:{i:02d}:00", text, order=f"{i:02d}") for i, text in enumerate(texts)]
        export = self.write_json("texts.json", fixture_dict({"conv-a": records}))
        report = ingest(dataset, export, adapter_name="synthetic-fixture", profile_path=self.profile())
        connection = sqlite3.connect(dataset.db_path)
        stored = dict(connection.execute("SELECT native_message_id, text FROM messages").fetchall())
        payloads = dict(connection.execute("SELECT native_message_id, payload_sha256 FROM messages").fetchall())
        connection.close()
        self.assertEqual([stored[f"t{i}"] for i in range(len(texts))], texts)
        self.assertNotEqual(payloads["t2"], payloads["t3"])
        raw_dir = dataset.root / Path(report["raw_manifest_path"]).parent
        preserved = raw_dir / "files" / "texts.json"
        self.assertEqual(preserved.read_bytes(), export.read_bytes())
        manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"], [{"path": "texts.json", "sha256": hashlib.sha256(export.read_bytes()).hexdigest(), "size": export.stat().st_size}])
        self.assertEqual(hashlib.sha256((raw_dir / "manifest.json").read_bytes()).hexdigest(), report["export_sha256"])
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(preserved.stat().st_mode) & 0o222, 0)

    def test_failed_import_rolls_back_every_database_change(self) -> None:
        dataset = self.new_dataset()
        original = importer._insert_origins

        def fail_midway(connection, revision, dataset_id, records):
            original(connection, revision, dataset_id, records[:1])
            raise RuntimeError("simulated failure during origin inserts")

        with mock.patch.object(importer, "_insert_origins", fail_midway):
            with self.assertRaises(RuntimeError):
                self.ingest(dataset, {"conv-a": numbered("n", 5)})
        for table in ("import_batches", "messages", "message_origins"):
            self.assertEqual(_scalar(dataset, f"SELECT COUNT(*) FROM {table}"), 0, table)
        retry = self.ingest(dataset, {"conv-a": numbered("n", 5)})
        self.assertEqual((retry["status"], retry["revision"]), ("committed", 1))
        self.assertEqual(self.ingest(dataset, {"conv-a": numbered("n", 5)})["status"], "noop")

    def test_interrupted_import_process_leaves_no_partial_revision(self) -> None:
        dataset = self.new_dataset()
        export = self.write_json("export.json", fixture_dict({"conv-a": numbered("n", 40)}))
        profile = self.profile()
        script = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "from chat_conductor.evidence import importer\n"
            "from chat_conductor.evidence.store import Dataset\n"
            "def die(*args, **kwargs):\n"
            "    os._exit(9)\n"
            "importer._check_totals = die\n"
            "importer.ingest(Dataset(Path(sys.argv[1])), Path(sys.argv[2]), adapter_name='synthetic-fixture', profile_path=Path(sys.argv[3]))\n"
        )
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        completed = subprocess.run([sys.executable, "-c", script, str(dataset.root), str(export), str(profile)], env=env, capture_output=True)
        self.assertEqual(completed.returncode, 9, completed.stderr)
        for table in ("import_batches", "messages", "message_origins"):
            self.assertEqual(_scalar(dataset, f"SELECT COUNT(*) FROM {table}"), 0, table)
        report = ingest(dataset, export, adapter_name="synthetic-fixture", profile_path=profile)
        self.assertEqual((report["status"], report["revision"], report["counts"]["new"]), ("committed", 1, 40))

    def test_sealed_backdated_and_mutating_writes_are_rejected(self) -> None:
        dataset = self.new_dataset()
        self.ingest(dataset, {"conv-a": numbered("n", 3)}, name="one.json")
        self.ingest(dataset, {"conv-a": numbered("m", 2, start_minute=10)}, name="two.json")
        connection = dataset.connect_rw()
        existing_id, existing_text = connection.execute("SELECT id, text FROM messages WHERE first_seen_revision = 1 LIMIT 1").fetchone()

        def message_row(revision: int, native: str = "late", identifier: str | None = None) -> tuple:
            identifier = identifier or message_id("ds-test", NAMESPACE, "conv-a", native)
            return (identifier, NAMESPACE, "conv-a", native, "s_ana", "2026-03-01T01:00:00.000000Z", "x", "Asia/Shanghai", "9", "native", "text", "export_text", "late", "0" * 64, revision)

        insert = "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        attempts = {
            "sealed latest batch": (insert, message_row(2), "message insert exceeds the current import"),
            "backdated batch": (insert, message_row(1), "message insert exceeds the current import"),
            "origin into sealed batch": ("INSERT INTO message_origins VALUES (2, 'x.json#/9', ?)", (existing_id,), "origin insert exceeds the current import"),
            "replace existing": ("INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", message_row(2, "x", existing_id), "exceeds the current import"),
            "skip a revision": ("INSERT INTO import_batches VALUES (4, 'i4', ?, 'a', '1', ?, 'p', 't', 0, 0)", ("a" * 64, "b" * 64), "consecutive"),
            "update text": ("UPDATE messages SET text = 'edited' WHERE id = ?", (existing_id,), "immutable"),
            "delete message": ("DELETE FROM messages WHERE id = ?", (existing_id,), "immutable"),
            "update dataset": ("UPDATE dataset SET dataset_id = 'other'", (), "immutable"),
            "delete batch": ("DELETE FROM import_batches", (), "append-only"),
            "update origin": ("UPDATE message_origins SET raw_locator = 'x'", (), "append-only"),
        }
        try:
            for label, (sql, params, expected) in attempts.items():
                with self.subTest(label):
                    with self.assertRaises(sqlite3.DatabaseError) as caught:
                        connection.execute(sql, params)
                    self.assertIn(expected, str(caught.exception))
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO import_batches VALUES (3, 'i3', ?, 'a', '1', ?, 'p', 't', 1, 1)", ("c" * 64, "d" * 64))
            connection.execute(insert, message_row(3))
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(insert, message_row(3, "late-2"))
            with self.assertRaises(sqlite3.DatabaseError) as caught:
                connection.execute("INSERT INTO import_batches VALUES (4, 'i4', ?, 'a', '1', ?, 'p', 't', 0, 0)", ("e" * 64, "f" * 64))
            self.assertIn("prior import totals are incomplete", str(caught.exception))
            connection.execute("ROLLBACK")
            self.assertEqual(connection.execute("SELECT text FROM messages WHERE id = ?", (existing_id,)).fetchone()[0], existing_text)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM import_batches").fetchone()[0], 2)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
