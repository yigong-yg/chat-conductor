from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .adapters import ADAPTERS, resolve_adapter
from .indexer import index_source
from .evaluator import load_eval_cases, report_to_dict, run_eval
from .rehearse import DEFAULT_REHEARSE_BUDGET, DEFAULT_REHEARSE_LIMIT, DEFAULT_NEIGHBOR_RADIUS, rehearse
from .server import run_server
from .store import connect, default_index_path, init_schema, search, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chat-conductor")
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to the local SQLite index cache. Defaults to user app data for Claude indexes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a chat archive, auto-detecting the source format.")
    index_parser.add_argument("export", type=Path)
    index_parser.add_argument(
        "--source",
        choices=[adapter.name for adapter in ADAPTERS],
        help="Force a source adapter instead of auto-detecting.",
    )

    search_parser = subparsers.add_parser("search", help="Search indexed segments.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--after")
    search_parser.add_argument("--before")
    search_parser.add_argument("--role", choices=["human", "assistant"])
    search_parser.add_argument("--conv", help="Conversation id or title substring.")
    search_parser.add_argument("--source", help="Source filter, e.g. claude or wechat.")
    search_parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    rehearse_parser = subparsers.add_parser("rehearse", help="Return a budgeted recall block.")
    rehearse_parser.add_argument("query")
    rehearse_parser.add_argument("--limit", type=int, default=DEFAULT_REHEARSE_LIMIT)
    rehearse_parser.add_argument("--token-budget", type=int, default=DEFAULT_REHEARSE_BUDGET)
    rehearse_parser.add_argument("--neighbor-radius", type=int, default=DEFAULT_NEIGHBOR_RADIUS)
    rehearse_parser.add_argument("--after")
    rehearse_parser.add_argument("--before")
    rehearse_parser.add_argument("--role", choices=["human", "assistant"])
    rehearse_parser.add_argument("--conv", help="Conversation id or title substring.")

    serve_parser = subparsers.add_parser("serve", help="Serve search tools over MCP.")
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )

    eval_parser = subparsers.add_parser("eval", help="Run a recall@k eval file.")
    eval_parser.add_argument("cases", type=Path)
    eval_parser.add_argument("--k", type=int, default=10)

    subparsers.add_parser("status", help="Show index status.")

    args = parser.parse_args(argv)
    index_path = args.index or default_index_path()

    if args.command == "index":
        adapter = resolve_adapter(args.export, args.source)
        if adapter.name != "claude" and args.index is None:
            parser.error(f"indexing source {adapter.name!r} requires an explicit --index path")
        stats = index_source(args.export, index_path, adapter=adapter)
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        print(f"source: {adapter.name}")
        print(f"index: {index_path.expanduser()}")
        return 0

    if args.command == "serve":
        run_server(index_path, transport=args.transport)
        return 0

    connection = connect(index_path)
    init_schema(connection)
    try:
        if args.command == "search":
            results = search(
                connection,
                args.query,
                limit=args.limit,
                after=args.after,
                before=args.before,
                role=args.role,
                conv=args.conv,
                source=args.source,
            )
            if args.json:
                print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
            else:
                _print_results(results)
            return 0
        if args.command == "status":
            print(json.dumps(status(connection), ensure_ascii=False, indent=2))
            print(f"index: {index_path.expanduser()}")
            return 0
        if args.command == "rehearse":
            print(
                rehearse(
                    connection,
                    args.query,
                    limit=args.limit,
                    token_budget=args.token_budget,
                    neighbor_radius=args.neighbor_radius,
                    after=args.after,
                    before=args.before,
                    role=args.role,
                    conv=args.conv,
                )
            )
            return 0
        if args.command == "eval":
            cases = load_eval_cases(args.cases)
            report = run_eval(connection, cases, default_k=args.k)
            print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))
            return 0
    finally:
        connection.close()

    parser.error(f"unknown command: {args.command}")
    return 2


def _print_results(results: list) -> None:
    for index, result in enumerate(results, start=1):
        print(f"{index}. score={result.score:.6f} source={result.source} role={result.role} date={result.date}")
        print(f"   conv={result.conv_title} ({result.conv_uuid})")
        print(f"   turn={result.turn_id}")
        preview = result.text.replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:297] + "..."
        print(f"   {preview}")
        print()
