from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .indexer import index_export
from .rehearse import DEFAULT_REHEARSE_BUDGET, DEFAULT_REHEARSE_LIMIT, DEFAULT_NEIGHBOR_RADIUS, rehearse
from .server import run_server
from .store import connect, default_index_path, init_schema, search, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chat-conductor")
    parser.add_argument(
        "--index",
        type=Path,
        default=default_index_path(),
        help="Path to the local SQLite index cache. Defaults to user app data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a Claude export zip or directory.")
    index_parser.add_argument("export", type=Path)

    search_parser = subparsers.add_parser("search", help="Search indexed turns.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--after")
    search_parser.add_argument("--before")
    search_parser.add_argument("--role", choices=["human", "assistant"])
    search_parser.add_argument("--conv", help="Conversation UUID or title substring.")
    search_parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    rehearse_parser = subparsers.add_parser("rehearse", help="Return a budgeted recall block.")
    rehearse_parser.add_argument("query")
    rehearse_parser.add_argument("--limit", type=int, default=DEFAULT_REHEARSE_LIMIT)
    rehearse_parser.add_argument("--token-budget", type=int, default=DEFAULT_REHEARSE_BUDGET)
    rehearse_parser.add_argument("--neighbor-radius", type=int, default=DEFAULT_NEIGHBOR_RADIUS)
    rehearse_parser.add_argument("--after")
    rehearse_parser.add_argument("--before")
    rehearse_parser.add_argument("--role", choices=["human", "assistant"])
    rehearse_parser.add_argument("--conv", help="Conversation UUID or title substring.")

    serve_parser = subparsers.add_parser("serve", help="Serve search tools over MCP.")
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )

    subparsers.add_parser("status", help="Show index status.")

    args = parser.parse_args(argv)

    if args.command == "index":
        stats = index_export(args.export, args.index)
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        print(f"index: {args.index.expanduser()}")
        return 0

    if args.command == "serve":
        run_server(args.index, transport=args.transport)
        return 0

    connection = connect(args.index)
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
            )
            if args.json:
                print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
            else:
                _print_results(results)
            return 0
        if args.command == "status":
            print(json.dumps(status(connection), ensure_ascii=False, indent=2))
            print(f"index: {args.index.expanduser()}")
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
    finally:
        connection.close()

    parser.error(f"unknown command: {args.command}")
    return 2


def _print_results(results: list) -> None:
    for index, result in enumerate(results, start=1):
        print(f"{index}. score={result.score:.6f} role={result.role} date={result.date}")
        print(f"   conv={result.conv_title} ({result.conv_uuid})")
        print(f"   turn={result.turn_id}")
        preview = result.text.replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:297] + "..."
        print(f"   {preview}")
        print()
