"""``python -m chat_conductor.evidence``: the opt-in evidence command line.

Results are JSON on stdout. Errors are JSON on stderr with a stable code and no
source text. Exit codes: 0 success, 2 invalid evidence/draft/input (including a
rejected import batch), 3 runtime, schema or configuration failure.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import traceback
from pathlib import Path

from .errors import EXIT_INVALID, EXIT_OK, EXIT_RUNTIME, ConfigError, EvidenceError, InvalidInput
from .jsonutil import dumps_canonical, loads_strict
from .records import KINDS, TEXT_ORIGINS
from .store import locate_dataset

PROG = "python -m chat_conductor.evidence"
DEBUG_ENV = "CHAT_EVIDENCE_DEBUG"


def build_parser() -> argparse.ArgumentParser:
    from .query import CASE_POLICIES, CONTEXT_RADIUS, Limits

    defaults = Limits()
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Chat Evidence prototype: pinned retrieval and citation verification for one explicit dataset.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", help="absolute dataset directory (must resolve outside every Git worktree)")
    selection.add_argument("--dataset-id", help="dataset id under the per-user data directory")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a new dataset in a new or empty directory")
    init.add_argument("--id", dest="requested_id", help="dataset id (default: generated)")

    ingest = commands.add_parser("ingest", help="import one export as one new corpus revision")
    ingest.add_argument("export", type=Path, help="export file or directory")
    ingest.add_argument("--adapter", required=True, help="adapter name; only 'synthetic-fixture' exists")
    ingest.add_argument("--profile", required=True, type=Path, help="private profile JSON")

    run = commands.add_parser("run", help="analysis runs").add_subparsers(dest="run_command", required=True)
    start = run.add_parser("start", help="pin a new run to the latest (or given) committed revision")
    start.add_argument("--question", required=True)
    start.add_argument("--revision", type=int)

    query = commands.add_parser("q", help="bounded queries within a run").add_subparsers(dest="query_command", required=True)

    search = query.add_parser("search", help="literal substring search; all terms must match")
    search.add_argument("--run", required=True)
    search.add_argument("terms", nargs="+", help="literal terms (use -- before a term that starts with '-')")
    search.add_argument("--case", choices=CASE_POLICIES, default="ascii-insensitive")
    _add_filters(search, conversation_required=False)
    _add_limits(search, defaults)

    ranged = query.add_parser("range", help="chronological messages of one conversation")
    ranged.add_argument("--run", required=True)
    _add_filters(ranged, conversation_required=True)
    _add_limits(ranged, defaults)

    ctx = query.add_parser("ctx", help="target plus neighbours in its conversation, all kinds")
    ctx.add_argument("--run", required=True)
    ctx.add_argument("message_id")
    ctx.add_argument("--radius", type=int, default=CONTEXT_RADIUS)
    _add_limits(ctx, defaults)

    sql = query.add_parser("sql", help="one read-only statement over an isolated pinned snapshot")
    sql.add_argument("--run", required=True)
    sql.add_argument("--sql-file", required=True, type=Path)
    sql.add_argument("--scope-file", required=True, type=Path, help="JSON: definition, unit, filters, missing_data[, denominator, period, timezone]")
    sql.add_argument("--params-file", type=Path, help="JSON object of named parameter values")
    sql.add_argument("--utc-param", action="append", default=[], metavar="NAME=LOCAL", help="bind a local date/time converted to UTC in --tz")
    sql.add_argument("--tz", help="IANA zone for --utc-param")
    sql.add_argument("--max-rows", type=int, default=200)
    sql.add_argument("--max-bytes", type=int, default=64 * 1024)
    sql.add_argument("--deadline-ms", type=int, default=5000)
    sql.add_argument("--max-steps", type=int, default=50_000_000)
    sql.add_argument("--snapshot-max-bytes", type=int, default=512 * 1024 * 1024)
    sql.add_argument("--snapshot-max-seconds", type=float, default=30.0)

    verify = commands.add_parser("verify", help="verify the run's analysis.json and render report.md")
    verify.add_argument("--run", required=True)
    verify.add_argument("--draft", type=Path, help="copy this draft into the run as analysis.json first")
    verify.add_argument("--check", action="store_true", help="only check that report.md still matches its verification")
    return parser


def _add_filters(parser: argparse.ArgumentParser, *, conversation_required: bool) -> None:
    parser.add_argument("--source-namespace", required=conversation_required)
    parser.add_argument("--conversation", required=conversation_required)
    parser.add_argument("--sender")
    parser.add_argument("--kind", action="append", choices=KINDS)
    parser.add_argument("--text-origin", action="append", choices=TEXT_ORIGINS)
    parser.add_argument("--from", dest="date_from", help="inclusive local start, YYYY-MM-DD[THH:MM[:SS]]")
    parser.add_argument("--to", dest="date_to", help="exclusive local end; a date means the start of that date")
    parser.add_argument("--tz", help="explicit IANA zone for --from/--to")


def _add_limits(parser: argparse.ArgumentParser, defaults) -> None:
    parser.add_argument("--limit", type=int, default=defaults.max_rows, help="maximum rows per page")
    parser.add_argument("--max-bytes", type=int, default=defaults.max_bytes, help="maximum UTF-8 JSON bytes per page")
    parser.add_argument("--deadline-ms", type=int, default=defaults.deadline_ms)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--cursor", help="continuation cursor from the previous page")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = locate_dataset(args.dataset, args.dataset_id)
        return _dispatch(args, dataset)
    except EvidenceError as exc:
        _error(exc.to_dict())
        return exc.exit_code
    except sqlite3.Error as exc:
        _error({"code": "sqlite.error", "message": f"SQLite failed ({type(exc).__name__})"})
        return EXIT_RUNTIME
    except OSError as exc:
        _error({"code": "os.error", "message": f"a file operation failed ({type(exc).__name__})"})
        return EXIT_RUNTIME
    except Exception as exc:  # noqa: BLE001 - never echo private data from unexpected errors
        if os.environ.get(DEBUG_ENV):
            traceback.print_exc()
        _error({"code": "internal.error", "message": f"unexpected {type(exc).__name__}; set {DEBUG_ENV}=1 for a traceback"})
        return EXIT_RUNTIME


def _dispatch(args: argparse.Namespace, dataset) -> int:
    from .query import Limits, context, range_query, search

    if args.command == "init":
        from .store import init_dataset

        info = init_dataset(dataset, requested_id=args.requested_id)
        _emit({"dataset_id": info.dataset_id, "dataset_dir": str(dataset.root), "schema_version": info.schema_version})
        return EXIT_OK
    if args.command == "ingest":
        from .importer import ingest

        _emit(ingest(dataset, args.export, adapter_name=args.adapter, profile_path=args.profile))
        return EXIT_OK
    if args.command == "run":
        from .runs import start_run

        run = start_run(dataset, args.question, revision=args.revision)
        _emit({
            "run_id": run.run_id,
            "dataset_id": run.dataset_id,
            "corpus_revision": run.corpus_revision,
            "created_at": run.created_at,
            "run_dir": str(run.directory),
            "draft_path": str(run.analysis_path),
        })
        return EXIT_OK
    if args.command == "q":
        if args.query_command == "sql":
            return _sql(args, dataset)
        limits = Limits(
            max_rows=args.limit,
            max_bytes=args.max_bytes,
            deadline_ms=args.deadline_ms,
            max_steps=args.max_steps,
        )
        if args.query_command == "ctx":
            result = context(dataset, args.run, args.message_id, radius=args.radius, limits=limits, cursor=args.cursor)
        else:
            arguments = {
                "source_namespace": args.source_namespace,
                "conversation_id": args.conversation,
                "sender_id": args.sender,
                "kinds": args.kind,
                "text_origins": args.text_origin,
                "from": args.date_from,
                "to": args.date_to,
                "tz": args.tz,
            }
            if args.query_command == "search":
                result = search(dataset, args.run, {"terms": args.terms, "case": args.case, **arguments}, limits=limits, cursor=args.cursor)
            else:
                result = range_query(dataset, args.run, arguments, limits=limits, cursor=args.cursor)
        _write(result.output)
        return EXIT_OK
    if args.command == "verify":
        from .verify import check_report, verify_run

        if args.check:
            if args.draft is not None:
                raise InvalidInput("verify.arguments", "--check does not take --draft")
            summary = check_report(dataset, args.run)
            _emit(summary)
            return EXIT_OK if summary["status"] == "verified" else EXIT_INVALID
        outcome = verify_run(dataset, args.run, draft_source=args.draft)
        record = outcome.record
        _emit({
            "status": outcome.status,
            "run_id": record["run_id"],
            "corpus_revision": record["corpus_revision"],
            "failures": [{key: item[key] for key in ("code", "claim_id", "field")} for item in record["failures"]],
            "report": str(outcome.report_path) if outcome.report_path else None,
            "report_sha256": record["report_sha256"],
            "verification": str(dataset.runs_dir / record["run_id"] / "verification.json"),
        })
        return outcome.exit_code
    raise ConfigError("cli.unknown_command", "unknown command")


def _sql(args: argparse.Namespace, dataset) -> int:
    from .sqlquery import SqlLimits, sql_query

    sql_text = _read_text(args.sql_file, "sql")
    parameters = loads_strict(_read_bytes(args.params_file, "params"), what="params") if args.params_file else {}
    scope = loads_strict(_read_bytes(args.scope_file, "scope"), what="scope")
    limits = SqlLimits(
        max_rows=args.max_rows,
        max_bytes=args.max_bytes,
        deadline_ms=args.deadline_ms,
        max_steps=args.max_steps,
        snapshot_max_bytes=args.snapshot_max_bytes,
        snapshot_max_seconds=args.snapshot_max_seconds,
    )
    result = sql_query(dataset, args.run, sql_text, parameters, scope, utc_assignments=args.utc_param, tz=args.tz, limits=limits)
    _write(result.output)
    return EXIT_OK


def _read_bytes(path: Path, what: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ConfigError(f"{what}.unreadable", f"cannot read the {what} file") from exc


def _read_text(path: Path, what: str) -> str:
    data = _read_bytes(path, what)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidInput(f"{what}.encoding", f"the {what} file is not valid UTF-8") from None


def _emit(payload: dict) -> None:
    _write(dumps_canonical(payload).encode("utf-8"))


def _write(data: bytes) -> None:
    sys.stdout.buffer.write(data + b"\n")
    sys.stdout.buffer.flush()


def _error(payload: dict) -> None:
    sys.stderr.write(dumps_canonical({"error": payload}) + "\n")
    sys.stderr.flush()
