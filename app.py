"""
app.py — Gradio web interface for the RAG Assistant.

Run: python app.py
Then open http://localhost:7860
"""

from __future__ import annotations

import gradio as gr
from src.pipeline import RAGPipeline

rag = RAGPipeline()


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_ingest(files, urls_text: str, reset: bool):
    sources = []

    if files:
        sources.extend([f.name for f in files])

    if urls_text.strip():
        sources.extend([u.strip() for u in urls_text.strip().splitlines() if u.strip()])

    if not sources:
        return "⚠️ No sources provided. Upload files or enter URLs."

    try:
        stats = rag.ingest(sources, reset=reset)
        return f"✅ Ingestion complete.\n📦 Total chunks in index: {stats.get('total_chunks', '?')}"
    except Exception as e:
        return f"❌ Error: {e}"


def handle_query(question: str, top_k: int, history):
    if not question.strip():
        return history, ""
    try:
        result = rag.query(question, top_k=top_k, model=None)
        answer = result["answer"]
        sources_str = "\n".join(f"· {s}" for s in result["sources"])
        tokens_str  = (f"*Tokens — in: {result['tokens']['input']} | "
                       f"out: {result['tokens']['output']} | "
                       f"model: {result['model']}*")
        full_answer = f"{answer}\n\n**Sources:**\n{sources_str}\n\n{tokens_str}"
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": full_answer})
    except ValueError as e:
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": f"⚠️ {e}"})
    except Exception as e:
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": f"❌ Error: {e}"})
    return history, ""


def handle_stats():
    try:
        s = rag.stats()
        return f"📊 Chunks indexed: {s['total_chunks']} | Collection: {s['collection']}"
    except Exception as e:
        return f"❌ {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="RAG Assistant") as demo:

    gr.Markdown(
        """
        # 🔍 RAG Assistant
        **Retrieval-Augmented Generation** — Ask questions about your documents.
        Upload PDFs, text files or paste URLs, then ask anything.
        """
    )

    with gr.Tabs():
        # ── Tab 1: Ingest ──────────────────────────────────────────────────
        with gr.Tab("📥 Ingest Documents"):
            gr.Markdown("Upload documents or paste URLs to index them into the vector database.")
            with gr.Row():
                file_upload = gr.File(
                    label="Upload Files (PDF, TXT, MD)",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"],
                )
                with gr.Column():
                    urls_box = gr.Textbox(
                        label="Web URLs (one per line)",
                        placeholder="https://example.com/article\nhttps://...",
                        lines=5,
                    )
                    reset_cb = gr.Checkbox(label="Reset index before ingesting", value=False)
            ingest_btn    = gr.Button("🚀 Ingest", variant="primary")
            ingest_output = gr.Textbox(label="Result", lines=3, interactive=False)
            ingest_btn.click(
                handle_ingest,
                inputs=[file_upload, urls_box, reset_cb],
                outputs=ingest_output,
            )

        # ── Tab 2: Query ───────────────────────────────────────────────────
        with gr.Tab("💬 Ask"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    label="Question",
                    scale=5,
                    lines=1,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Top-K chunks",
                    scale=1,
                )
            with gr.Row():
                ask_btn   = gr.Button("Ask →", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", scale=1)

            ask_btn.click(
                handle_query,
                inputs=[question_box, top_k_slider, chatbot],
                outputs=[chatbot, question_box],
            )
            question_box.submit(
                handle_query,
                inputs=[question_box, top_k_slider, chatbot],
                outputs=[chatbot, question_box],
            )
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_box])



        # ── Tab 3: Stats ───────────────────────────────────────────────────
        with gr.Tab("📊 Index Stats"):
            stats_btn    = gr.Button("Refresh stats")
            stats_output = gr.Textbox(label="", lines=3, interactive=False)
            stats_btn.click(handle_stats, outputs=stats_output)

    gr.Markdown(
        "*Built with LangChain · ChromaDB · Sentence Transformers · Claude API · Gradio*"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="orange"),
        css=".gradio-container { max-width: 900px; margin: auto; } footer { display: none !important; }",
    )
