# RAG Assistant 🔍

**Production-ready Retrieval-Augmented Generation system** — ask questions about any document and get accurate, cited answers.

Built with LangChain · ChromaDB · Sentence Transformers · Claude API (Anthropic) · Gradio

---

## What it does

Upload PDFs, text files or web URLs → the system indexes them into a local vector database → ask any question → get a precise answer with source citations.

No hallucinations: the LLM is constrained to answer only from the provided documents.

```
Documents (PDF / TXT / URL)
        │
        ▼
  [Chunking + Cleaning]          ← RecursiveTextSplitter (800 chars, 120 overlap)
        │
        ▼
  [Embedding]                    ← all-MiniLM-L6-v2 (local, free, 384-dim)
        │
        ▼
  [Vector Store]                 ← ChromaDB (persistent, cosine similarity)
        │
  Query ▼
  [Semantic Retrieval]           ← Top-K most relevant chunks
        │
        ▼
  [LLM Generation]               ← Claude 3.5 Haiku (citation-constrained prompt)
        │
        ▼
  Answer with [source, page] citations
```

---

## Stack

| Component        | Technology                          |
|-----------------|-------------------------------------|
| Orchestration    | LangChain                           |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector DB        | ChromaDB (persistent)               |
| LLM              | Claude 3.5 Haiku (Anthropic API)    |
| UI               | Gradio                              |
| CLI              | argparse                            |
| Document parsing | PyPDF, BeautifulSoup4               |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/bentabol/rag-assistant
cd rag-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3a. Web UI

```bash
python app.py
# Open http://localhost:7860
```

### 3b. CLI

```bash
# Ingest documents
python cli.py ingest data/docs/paper.pdf https://example.com/article

# Ask a question
python cli.py query "What are the main conclusions?"

# Check index stats
python cli.py stats
```

### 3c. Python API

```python
from src.pipeline import RAGPipeline

rag = RAGPipeline()

# Ingest
rag.ingest(["paper.pdf", "https://arxiv.org/abs/..."])

# Query
result = rag.query("What methodology was used?")
print(result["answer"])
# → "The authors used a transformer-based architecture... [paper.pdf, p.4]"
```

---

## Features

- **Multi-format ingestion** — PDF, TXT, Markdown, web URLs
- **Smart chunking** — RecursiveTextSplitter with configurable size/overlap
- **Local embeddings** — no API cost for indexing (sentence-transformers)
- **Persistent vector store** — ChromaDB on disk, survives restarts
- **Citation-constrained LLM** — Claude prompted to cite sources and admit uncertainty
- **Gradio UI** — clean chat interface with file upload
- **CLI** — scriptable for automation and batch processing
- **Rich terminal output** — formatted tables, progress bars

---

## Project structure

```
rag-assistant/
├── src/
│   ├── ingestion.py     # Document loading & chunking
│   ├── embeddings.py    # Embedding model + ChromaDB
│   ├── llm.py           # Claude API integration
│   └── pipeline.py      # Main RAG orchestrator
├── app.py               # Gradio web UI
├── cli.py               # CLI interface
├── data/
│   ├── docs/            # Place your documents here
│   └── chroma_db/       # Persistent vector store (auto-created)
├── tests/
├── requirements.txt
└── .env.example
```

---

## Configuration

| Variable          | Default                    | Description                  |
|-------------------|---------------------------|------------------------------|
| `ANTHROPIC_API_KEY` | —                        | Required. Get at console.anthropic.com |
| `CHUNK_SIZE`      | 800                        | Characters per chunk         |
| `CHUNK_OVERLAP`   | 120                        | Overlap between chunks       |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`        | Local embedding model        |
| `top_k`           | 5                          | Chunks retrieved per query   |

---

## Author

**Guillermo Bentabol García** — ML Engineer & AI Systems Builder

- Portfolio: [portfolio-next-three-sepia.vercel.app](https://portfolio-next-three-sepia.vercel.app)
- LinkedIn: [linkedin.com/in/guillermo-bentabol-garcia-380ab623a](https://www.linkedin.com/in/guillermo-bentabol-garcia-380ab623a/)
- GitHub: [github.com/bentabol](https://github.com/bentabol)
