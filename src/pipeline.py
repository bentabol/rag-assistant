"""
pipeline.py — Main RAG pipeline orchestrating ingestion → retrieval → generation.
"""

from __future__ import annotations

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ingestion  import load_documents
from .embeddings import index_documents, retrieve, collection_stats
from .llm        import generate_answer

console = Console()


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage:
        rag = RAGPipeline()
        rag.ingest(["path/to/doc.pdf", "https://example.com/article"])
        result = rag.query("What is the main finding?")
        print(result["answer"])
    """

    def ingest(self, sources: List[str], reset: bool = False) -> dict:
        """
        Load, chunk and index documents.

        Args:
            sources: List of file paths or URLs
            reset:   If True, clears existing index before ingesting
        Returns:
            Stats dict
        """
        console.print(Panel("[bold]RAG Ingestion[/bold]", style="cyan"))
        chunks = load_documents(sources)
        if not chunks:
            console.print("[red]No documents loaded. Aborting.[/red]")
            return {"chunks": 0}
        index_documents(chunks, reset=reset)
        stats = collection_stats()
        console.print(f"\n[green]✓ Ingestion complete.[/green] Total in DB: {stats['total_chunks']} chunks")
        return stats

    def query(
        self,
        question: str,
        top_k: int = 5,
        model: str | None = None,
    ) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Returns:
            {
              "question": str,
              "answer":   str,
              "sources":  List[str],
              "chunks":   List[dict],
              "tokens":   dict,
              "model":    str,
            }
        """
        console.print(f"\n[bold cyan]Query:[/bold cyan] {question}")

        # Retrieve
        chunks = retrieve(question, top_k=top_k)
        self._print_chunks(chunks)

        # Generate
        result = generate_answer(question, chunks, model=model)
        result["question"] = question
        result["chunks"]   = chunks

        # Display
        console.print(Panel(
            result["answer"],
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))
        console.print(f"[dim]Sources: {', '.join(result['sources'])} | "
                      f"Tokens in/out: {result['tokens']['input']}/{result['tokens']['output']}[/dim]")
        return result

    def stats(self) -> dict:
        """Return current index statistics."""
        s = collection_stats()
        console.print(f"[bold]Index stats:[/bold] {s}")
        return s

    @staticmethod
    def _print_chunks(chunks: List[dict]) -> None:
        table = Table(title="Retrieved Chunks", show_lines=True)
        table.add_column("#",       style="dim", width=3)
        table.add_column("Source",  style="cyan", max_width=30)
        table.add_column("Score",   style="green", width=7)
        table.add_column("Preview", max_width=60)
        for i, c in enumerate(chunks, 1):
            table.add_row(
                str(i),
                c["source"],
                str(c["score"]),
                c["text"][:120].replace("\n", " ") + "…",
            )
        console.print(table)
