"""
embeddings.py — Embedding model + ChromaDB vector store.

Uses sentence-transformers/all-MiniLM-L6-v2 (local, free, 384-dim).
ChromaDB persists to disk so re-indexing is only needed on first run
or when documents change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR      = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_documents"

# ── Singleton embedding model ─────────────────────────────────────────────────
_model: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        console.print(f"[dim]Loading embedding model '{EMBEDDING_MODEL}'…[/dim]")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        console.print("[green]✓[/green] Embedding model ready")
    return _model


# ── ChromaDB client ───────────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Indexing ──────────────────────────────────────────────────────────────────

def _doc_id(content: str, source: str, index: int) -> str:
    """Deterministic ID for a chunk."""
    raw = f"{source}:{index}:{content[:80]}"
    return hashlib.md5(raw.encode()).hexdigest()


def index_documents(chunks: List[Document], reset: bool = False) -> chromadb.Collection:
    """
    Embed and store document chunks in ChromaDB.
    If reset=True, clears the collection first.
    """
    client     = get_chroma_client()
    collection = get_collection(client)

    if reset:
        client.delete_collection(COLLECTION_NAME)
        collection = get_collection(client)
        console.print("[yellow]Collection reset[/yellow]")

    model = get_embedding_model()

    texts     = [c.page_content for c in chunks]
    metadatas = [c.metadata      for c in chunks]
    ids       = [_doc_id(c.page_content, c.metadata.get("source", ""), i)
                 for i, c in enumerate(chunks)]

    # Batch embedding
    console.print(f"[dim]Embedding {len(texts)} chunks…[/dim]")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # Upsert in batches of 500
    batch = 500
    for i in range(0, len(texts), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            documents=texts[i : i + batch],
            embeddings=embeddings[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    console.print(f"[green]✓[/green] Indexed {len(texts)} chunks into ChromaDB")
    return collection


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> List[dict]:
    """
    Embed query and retrieve top_k most similar chunks.
    Returns list of dicts: {text, source, score, metadata}
    """
    model      = get_embedding_model()
    client     = get_chroma_client()
    collection = get_collection(client)

    if collection.count() == 0:
        raise ValueError("No documents indexed yet. Run 'ingest' first.")

    q_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "page":     meta.get("page"),
            "type":     meta.get("type", ""),
            "score":    round(1 - dist, 4),   # cosine similarity
            "metadata": meta,
        })

    return chunks


def collection_stats() -> dict:
    """Return basic stats about the current collection."""
    client     = get_chroma_client()
    collection = get_collection(client)
    count = collection.count()
    return {"total_chunks": count, "collection": COLLECTION_NAME}
