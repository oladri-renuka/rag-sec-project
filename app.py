# app.py  ← root level, NOT app/app.py
import sys
import os
sys.path.insert(0, '.')

import gradio as gr
from src.ingestion.chunker import Chunk  # noqa
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerRetriever
from src.generation.pipeline import build_prompt, call_groq
import time
# Auto-rebuild ChromaDB index if missing
if not os.path.exists("data/chromadb"):
    print("Rebuilding ChromaDB index from embeddings...")
    import subprocess
    subprocess.run(["python", "rebuild_index.py"], check=True)

print("Loading retrievers...")
dense    = DenseRetriever()
bm25     = BM25Retriever()
hybrid   = HybridRetriever()
reranker = RerankerRetriever()
print("All retrievers ready.")

RETRIEVERS = {
    "Dense (Semantic)":          dense,
    "BM25 (Keyword)":            bm25,
    "Hybrid RRF (Dense + BM25)": hybrid,
    "Hybrid + Reranker":         reranker,
}

SECTIONS = {
    "All Sections":        None,
    "Business (1)":        "section_1",
    "Risk Factors (1A)":   "section_1A",
    "MD&A (7)":            "section_7",
    "Market Risk (7A)":    "section_7A",
    "Financial Stmts (8)": "section_8",
}

COMPANIES = {
    "All Companies":                None,
    "Advanced Micro Devices (AMD)": "ADVANCED MICRO DEVICES INC",
    "Abbott Laboratories":          "ABBOTT LABORATORIES",
    "AAR Corp":                     "AAR CORP",
    "Adams Resources & Energy":     "ADAMS RESOURCES & ENERGY, INC.",
    "Air Products & Chemicals":     "Air Products & Chemicals, Inc.",
    "ACME United":                  "ACME UNITED CORP",
    "BK Technologies":              "BK Technologies Corp",
    "CECO Environmental":           "CECO ENVIRONMENTAL CORP",
    "Matson":                       "Matson, Inc.",
    "Worlds Inc":                   "WORLDS INC",
}


def answer_question(question, retriever_name, section_filter,
                    company_filter, num_chunks, groq_key):

    if not question.strip():
        return "Please enter a question.", "", ""

    if not groq_key.strip():
        return "Please enter your Groq API key in the settings box.", "", ""

    # Temporarily set the key for this request
    os.environ["GROQ_API_KEY"] = groq_key.strip()

    filters = {}
    sec = SECTIONS.get(section_filter)
    com = COMPANIES.get(company_filter)
    if sec:
        filters["section"] = sec
    if com:
        filters["company"] = com

    retriever = RETRIEVERS[retriever_name]
    t0 = time.time()
    try:
        chunks = retriever.retrieve(
            question, k=int(num_chunks),
            filters=filters if filters else None
        )
    except Exception as e:
        return f"Retrieval error: {e}", "", ""

    retrieval_time = time.time() - t0

    if not chunks:
        return "No relevant chunks found. Try removing filters.", "", ""

    prompt = build_prompt(question, chunks)
    t1 = time.time()
    try:
        answer = call_groq(prompt)
    except Exception as e:
        answer = f"Generation error: {e}"
    gen_time = time.time() - t1

    timing = (f"Retrieval: {retrieval_time:.2f}s | "
              f"Generation: {gen_time:.1f}s | "
              f"Total: {retrieval_time + gen_time:.1f}s | "
              f"Chunks: {len(chunks)}")

    sources = ""
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        score = c.get("rerank_score", c.get("rrf_score", c.get("score", 0)))
        sources += (
            f"[{i}] {meta['company']}\n"
            f"    Section: {meta['section_name']} | "
            f"Date: {meta['filing_date']} | Score: {score:.4f}\n"
            f"    {c['text'][:250]}...\n\n"
        )

    return answer, sources, timing


with gr.Blocks(title="SEC 10-K RAG System") as demo:
    gr.Markdown("# 📊 SEC 10-K Filing RAG System")
    gr.Markdown(
        "Ask questions about 10-K filings from 10 public companies (1993–2021).  \n"
        "Powered by: `BAAI/bge-small` · `ChromaDB` · `BM25` · `Cross-encoder` · `Llama 3.3 70B`"
    )

    with gr.Row():
        groq_key = gr.Textbox(
            label="🔑 Groq API Key (free at console.groq.com)",
            placeholder="gsk_...",
            type="password",
            scale=2
        )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are AMD's main competitive risks?",
                lines=2
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Search & Answer", variant="primary")

    with gr.Row():
        retriever_choice = gr.Dropdown(
            choices=list(RETRIEVERS.keys()),
            value="Hybrid RRF (Dense + BM25)",
            label="Retrieval Strategy"
        )
        section_filter = gr.Dropdown(
            choices=list(SECTIONS.keys()),
            value="All Sections",
            label="Filter by Section"
        )
        company_filter = gr.Dropdown(
            choices=list(COMPANIES.keys()),
            value="All Companies",
            label="Filter by Company"
        )
        num_chunks = gr.Slider(
            minimum=3, maximum=10, value=5, step=1,
            label="Chunks to retrieve"
        )

    timing_display = gr.Textbox(label="Timing", interactive=False, lines=1)

    with gr.Tabs():
        with gr.Tab("Answer"):
            answer_display = gr.Textbox(
                label="Answer", lines=10, interactive=False
            )
        with gr.Tab("Retrieved Sources"):
            sources_display = gr.Textbox(
                label="Sources", lines=20, interactive=False
            )

    gr.Examples(
        examples=[
            ["What are AMD's main competitive risks related to Intel?",
             "Dense (Semantic)", "Risk Factors (1A)",
             "Advanced Micro Devices (AMD)", 5],
            ["How did Abbott Laboratories describe their revenue sources?",
             "Hybrid RRF (Dense + BM25)", "All Sections",
             "Abbott Laboratories", 5],
            ["What cybersecurity risks do companies disclose?",
             "Hybrid + Reranker", "Risk Factors (1A)", "All Companies", 5],
            ["How did COVID-19 impact AAR Corp's business?",
             "Hybrid RRF (Dense + BM25)", "All Sections", "AAR Corp", 5],
        ],
        inputs=[question, retriever_choice, section_filter,
                company_filter, num_chunks],
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question, retriever_choice, section_filter,
                company_filter, num_chunks, groq_key],
        outputs=[answer_display, sources_display, timing_display]
    )
    question.submit(
        fn=answer_question,
        inputs=[question, retriever_choice, section_filter,
                company_filter, num_chunks, groq_key],
        outputs=[answer_display, sources_display, timing_display]
    )

if __name__ == "__main__":
    demo.launch()