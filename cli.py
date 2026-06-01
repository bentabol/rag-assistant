"""
cli.py — Command-line interface for the RAG Assistant.

Usage:
  python cli.py ingest path/to/doc.pdf https://example.com
  python cli.py query "What is the main topic?"
  python cli.py stats
  python cli.py ingest --reset path/to/doc.pdf
"""

import argparse
import sys
from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG Assistant CLI — ask questions about your documents",
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Load and index documents")
    p_ingest.add_argument("sources", nargs="+", help="File paths or URLs")
    p_ingest.add_argument("--reset", action="store_true",
                          help="Clear existing index before ingesting")

    # query
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--top-k", type=int, default=5,
                         help="Number of chunks to retrieve (default: 5)")
    p_query.add_argument("--model", default="claude-3-5-haiku-20241022",
                         help="Claude model to use")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    rag = RAGPipeline()

    if args.command == "ingest":
        rag.ingest(args.sources, reset=args.reset)

    elif args.command == "query":
        result = rag.query(args.question, top_k=args.top_k, model=args.model)
        # Exit code 0 = success
        sys.exit(0)

    elif args.command == "stats":
        rag.stats()


if __name__ == "__main__":
    main()
