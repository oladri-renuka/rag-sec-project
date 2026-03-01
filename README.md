# SEC 10-K RAG System — Multi-Strategy Retrieval Benchmark

> A production-grade RAG pipeline that benchmarks 4 retrieval strategies against a hand-labeled evaluation set of SEC 10-K filings. Built to answer a real question: **does hybrid retrieval actually beat dense-only in financial documents?**

**[🚀 Live Demo](https://huggingface.co/spaces/oladri-Renuka/rag-sec-filings)** · **[Source](https://github.com/oladri-renuka/rag-sec-project)**

---

## What This Actually Does

Most RAG tutorials stop at "it returns relevant chunks." This project asks harder questions:

- Which retrieval strategy produces more *faithful* answers — ones that don't hallucinate beyond the source?
- Does adding a cross-encoder reranker justify the 8x latency cost?
- When does BM25 outperform dense embeddings, and why?

I built an evaluation framework with a 29-question golden dataset to answer these with real numbers, not intuition.

---

## Results

Evaluated across **29 questions**, **4 retrieval strategies**, **116 total RAG runs** using Llama 3.2 3B via Ollama locally.

| Retriever | ROUGE-L | Context Precision | Faithfulness | Latency |
|-----------|:-------:|:-----------------:|:------------:|:-------:|
| Dense (BGE-small) | 0.1484 | 0.890 | 0.833 | 0.04s |
| BM25 (Okapi) | 0.1481 | 0.821 | 0.767 | 0.03s |
| **Hybrid RRF** | **0.1510** | 0.876 | **0.898** | 0.06s |
| Hybrid + Reranker | 0.1422 | 0.890 | 0.877 | 0.34s |

**Key finding**: Hybrid RRF wins on faithfulness (0.898 vs 0.767 for BM25 baseline — a 17% improvement). The cross-encoder reranker adds 0.28s latency for marginal quality gains over hybrid alone, which doesn't justify the cost at this scale.

**By difficulty**:
| Difficulty | ROUGE-L | Context Precision |
|------------|:-------:|:-----------------:|
| Easy | 0.163 | 0.909 |
| Medium | 0.147 | 0.745 |
| Hard | 0.126 | 0.969 |

Hard questions had *higher* context precision — they tend to be company-specific, so retrieval is more targeted. But ROUGE-L drops because the answers require synthesis across multiple filings.

---

## What Failed and What I Learned

**Section label mapping was wrong.** The dataset's integer section index (0-19) doesn't map linearly to section names. Section index 10 has 60k sentences — that's not "Legal Proceedings," it's mislabeled data. I fixed this by extracting the true section from the `sentenceID` field (`0000001750_10-K_2020_section_1A_0`) instead of trusting the integer label. The lesson: always sanity-check dataset documentation against actual data.

**Groq rate limits broke the evaluation run.** The free tier allows ~30 requests/minute. Running 116 evaluations in sequence hit the limit after question 6. I rebuilt the eval loop with exponential backoff, then switched to local Ollama (3B model) for evaluation — slower but unlimited. For production eval, you'd either pay for API access or use a local 70B model via GGUF.

**ChromaDB's Rust backend broke on HuggingFace Spaces.** The new ChromaDB version uses a Rust-based backend that doesn't recognize collections built with the Python backend locally. Solution: auto-detect missing collection on startup and rebuild from `chunks.pkl` + `embeddings.npz`. This added ~90s to first cold start but made deployment reliable.

**The cross-encoder reranker underperformed expectations.** `ms-marco-MiniLM-L-6-v2` was trained on web search data (MS MARCO), not financial documents. The reranking scores for financial text were often negative or near-zero, indicating poor domain fit. A reranker fine-tuned on financial QA would likely show more meaningful gains. This is the clearest next step for improving the pipeline.

**What I'd do differently**: Start with FAISS instead of ChromaDB for portability. FAISS is a single file, works identically everywhere, and has no database migration issues. ChromaDB adds nice filtering syntax but the deployment friction isn't worth it for a project at this scale.

---

## Architecture

```
Raw Dataset (HuggingFace)
         │
         ▼
  Section Extraction
  (sentenceID parsing)
         │
         ▼
  Sentence Grouping          ← 8 sentences per chunk
  (by docID + section)       ← filters section_1, 1A, 7, 7A, 8
         │
         ▼
   19,316 Chunks
         │
    ┌────┴────┐
    ▼         ▼
 BGE-small   BM25
 Embeddings  Index
 (384-dim)   (Okapi BM25)
    │         │
    ▼         ▼
 ChromaDB  In-memory
 (cosine)  token index
    │         │
    └────┬────┘
         ▼
   Reciprocal Rank
   Fusion (k=60)
         │
         ▼
  Cross-Encoder          ← ms-marco-MiniLM-L-6-v2
  Reranking              ← scores query+chunk jointly
         │
         ▼
   Top-5 Chunks
         │
         ▼
  Prompt Template        ← SYSTEM + context + question
         │
         ▼
  Llama 3.3 70B          ← via Groq API (free tier)
  or Llama 3.2 3B        ← via Ollama (local)
         │
         ▼
   Cited Answer
```

---

## Dataset

**[JanosAudran/financial-reports-sec](https://huggingface.co/datasets/JanosAudran/financial-reports-sec)** — 200k sentences from 188 10-K filings across 10 companies (1993–2021).

Key insight: the dataset includes market return labels (1d, 5d, 30d windows post-filing). This enables a bonus analysis: does the sentiment of retrieved context correlate with stock returns? I haven't run this yet — it's the most interesting extension.

Companies: AMD, Abbott Laboratories, AAR Corp, Adams Resources & Energy, Air Products & Chemicals, ACME United, BK Technologies, CECO Environmental, Matson, Worlds Inc.

Sections used: Business (1), Risk Factors (1A), MD&A (7), Market Risk (7A), Financial Statements (8).

---

## Evaluation Methodology

**Golden dataset**: 29 hand-labeled Q&A pairs covering all 10 companies, 3 difficulty levels, and cross-company questions. Took ~3 hours to create. Available at `data/golden_dataset/questions.json`.

**Metrics**:
- **ROUGE-L**: Longest common subsequence F1 between generated answer and ground truth
- **Context Precision**: Fraction of retrieved chunks from the correct company (proxy for retrieval relevance)
- **Answer Relevancy**: Fraction of question keywords present in answer (proxy for on-topic response)
- **Faithfulness**: Fraction of answer sentences with >40% lexical overlap with retrieved context (proxy for hallucination rate)

These are lightweight proxies, not RAGAS (which requires an LLM judge). The tradeoff: faster, free, reproducible — but less nuanced than LLM-as-judge metrics.

---

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | BAAI/bge-small-en-v1.5 | Best MTEB score under 100MB, runs on CPU |
| Vector store | ChromaDB | Persistent, filtering support, local |
| Sparse retrieval | rank-bm25 (Okapi BM25) | Fast, no training needed |
| Reranker | ms-marco-MiniLM-L-6-v2 | 90MB, CPU-feasible, ~300ms |
| LLM (local) | Llama 3.2 3B via Ollama | Zero cost, unlimited, 10s/answer |
| LLM (cloud) | Llama 3.3 70B via Groq | Free tier, 0.8s/answer, 30 req/min |
| UI | Gradio | Fast to build, native HF Spaces support |
| Tracking | MLflow | Local experiment logging |

**Total cost**: $0. All models are free, all APIs have free tiers, all infrastructure is free.

---

## Run It Yourself

```bash
git clone https://github.com/oladri-renuka/rag-sec-project
cd rag-sec-project
python -m venv venv && source venv/bin/activate
pip install datasets sentence-transformers chromadb rank-bm25 gradio requests numpy pandas

# Build the pipeline
python data.py                              # download + explore dataset
python src/ingestion/chunker.py            # 19,316 chunks → data/chunks.pkl
python src/embeddings/embedder.py          # embeddings → data/embeddings.npz
python src/retrieval/dense_retriever.py    # ChromaDB index

# Run evaluation
export GROQ_API_KEY=your_key_here
python scripts/run_evaluation.py           # ~20 min with Ollama

# Launch UI
python app/app.py                          # http://localhost:7860
```

---

## Project Structure

```
rag_sec_project/
├── src/
│   ├── ingestion/
│   │   └── chunker.py          # section-aware chunking
│   ├── embeddings/
│   │   └── embedder.py         # BGE-small batch embedding
│   ├── retrieval/
│   │   ├── dense_retriever.py  # ChromaDB + cosine similarity
│   │   ├── bm25_retriever.py   # Okapi BM25
│   │   ├── hybrid_retriever.py # RRF fusion (k=60)
│   │   └── reranker.py         # cross-encoder reranking
│   └── generation/
│       └── pipeline.py         # prompt builder + LLM clients
├── app/
│   └── app.py                  # Gradio UI
├── scripts/
│   ├── create_golden_dataset.py
│   └── run_evaluation.py
├── data/
│   └── golden_dataset/
│       └── questions.json      # 29 hand-labeled Q&A pairs
└── experiments/
    └── results/
        ├── full_results.json
        └── metrics_summary.csv
```

---

## What's Next

In priority order, based on what would actually move the metrics:

1. **Domain-adapted reranker** — fine-tune a cross-encoder on financial QA pairs. The MS MARCO reranker is the weakest link.
2. **Larger embedding model** — swap BGE-small for BGE-large on Kaggle T4 and re-run the benchmark. Expected +3-5% on all metrics.
3. **Market return correlation** — use the dataset's 30d return labels to analyze whether risk factor sentiment predicts stock movement post-filing.
4. **HyDE retrieval** — generate a hypothetical answer first, embed it, then retrieve. Often outperforms query-only retrieval for complex financial questions.
5. **Temporal filtering** — add date range filters so you can ask "what did AMD say about competition *before 2010*?"

---

## Why This Exists

I built this to answer a question I kept seeing hand-waved in RAG tutorials: "hybrid retrieval is better." Better by how much? On what queries? At what latency cost? The benchmark gives actual numbers. The answer turns out to be: hybrid wins on faithfulness by 17% over BM25 and 8% over dense-only, at 2x the latency of either. Whether that tradeoff is worth it depends on your use case.

---

*Built by [Renuka Oladri](https://github.com/oladri-renuka). If you use this as a reference, a star is appreciated.*
