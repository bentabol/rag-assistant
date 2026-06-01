"""
ingestion.py — Document loading, chunking and preprocessing.

Supports:
  - PDF files
  - Plain text / Markdown files
  - Web URLs (HTML → clean text)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
from rich.console import Console

console = Console()

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 120   # overlap to preserve context across chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdf(path: str | Path) -> List[Document]:
    """Extract text from a PDF and return chunked Documents."""
    path = Path(path)
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": i + 1, "type": "pdf"},
                )
            )
    chunks = splitter.split_documents(pages)
    console.print(f"[green]✓[/green] PDF '{path.name}' → {len(chunks)} chunks")
    return chunks


def load_text(path: str | Path) -> List[Document]:
    """Load a plain text / Markdown file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    doc = Document(
        page_content=text,
        metadata={"source": path.name, "type": "text"},
    )
    chunks = splitter.split_documents([doc])
    console.print(f"[green]✓[/green] Text '{path.name}' → {len(chunks)} chunks")
    return chunks


def load_url(url: str) -> List[Document]:
    """Fetch a web page and extract clean text."""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    doc = Document(
        page_content=text,
        metadata={"source": url, "type": "web"},
    )
    chunks = splitter.split_documents([doc])
    console.print(f"[green]✓[/green] URL '{url[:60]}…' → {len(chunks)} chunks")
    return chunks


def load_documents(paths_or_urls: List[str]) -> List[Document]:
    """
    Auto-detect and load documents from a list of paths or URLs.
    Returns all chunks combined.
    """
    all_chunks: List[Document] = []

    for item in paths_or_urls:
        try:
            if item.startswith("http://") or item.startswith("https://"):
                all_chunks.extend(load_url(item))
            else:
                p = Path(item)
                if not p.exists():
                    console.print(f"[yellow]⚠[/yellow] File not found: {item}")
                    continue
                if p.suffix.lower() == ".pdf":
                    all_chunks.extend(load_pdf(p))
                else:
                    all_chunks.extend(load_text(p))
        except Exception as e:
            console.print(f"[red]✗[/red] Error loading '{item}': {e}")

    console.print(f"\n[bold]Total chunks ready:[/bold] {len(all_chunks)}")
    return all_chunks
