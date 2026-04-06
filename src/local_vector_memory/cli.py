"""CLI entry point for local-vector-memory."""
from __future__ import annotations

import argparse
import json
import sys

from .core import LocalVectorMemory
from . import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lvm",
        description="Local Vector Memory — zero-cloud vector search with Ollama + Qdrant",
    )
    parser.add_argument("--version", action="version", version=f"lvm {__version__}")
    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Initialize the vector database")

    # add
    p_add = sub.add_parser("add", help="Add a text memory")
    p_add.add_argument("text", help="Text to store")
    p_add.add_argument("--source", default="manual", help="Source label")

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=6, help="Max results")
    p_search.add_argument("--json", action="store_true", help="Raw JSON output")

    # stats
    sub.add_parser("stats", help="Show database stats")

    # reindex
    p_reindex = sub.add_parser("reindex", help="Reindex markdown files")
    p_reindex.add_argument("--dir", required=True, help="Directory to index")
    p_reindex.add_argument("--glob", default="**/*.md", help="File glob pattern")

    # delete
    p_del = sub.add_parser("delete", help="Delete entries by source")
    p_del.add_argument("source", help="Source to delete")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    lvm = LocalVectorMemory()

    if args.command == "init":
        lvm.init_db()
        print(f"✅ Initialized at {lvm.db_path}")

    elif args.command == "add":
        result = lvm.add(args.text, source=args.source)
        print(f"✅ Added ({result['chunks']} chunk)")

    elif args.command == "search":
        results = lvm.search(args.query, limit=args.limit)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for i, r in enumerate(results, 1):
                print(f"\n{'─'*60}")
                print(f"#{i} score={r['score']} source={r['source']}")
                text = r['text']
                if len(text) > 200:
                    text = text[:200] + "..."
                print(text)

    elif args.command == "stats":
        stats = lvm.stats()
        print(f"Collection: {stats['collection']}")
        print(f"Vectors:    {stats['count']}")
        if 'db_path' in stats:
            print(f"DB path:    {stats['db_path']}")

    elif args.command == "reindex":
        print(f"🔄 Reindexing {args.dir} ({args.glob})...")
        result = lvm.reindex(args.dir, glob_pattern=args.glob, verbose=True)
        print(f"\n✅ Done: {result['files']} files, {result['total_chunks']} chunks")

    elif args.command == "delete":
        result = lvm.delete_source(args.source)
        print(f"✅ Deleted source: {args.source}")


if __name__ == "__main__":
    main()
