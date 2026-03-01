# src/generation/pipeline.py
import os
import sys
import time
import requests
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.retrieval.dense_retriever import DenseRetriever

# ── LLM Clients ──────────────────────────────────────

def call_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 200}  # shorter answers = faster eval
        },
        timeout=120
    )
    return response.json()["response"]

def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile",
              max_retries: int = 3) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY env variable.")

    for attempt in range(max_retries):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,   # reduced to use fewer tokens
                "temperature": 0.1
            },
            timeout=30
        )
        data = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        # Rate limited — wait and retry
        if "error" in data:
            error_msg = data["error"].get("message", "")
            print(f"    Groq error: {error_msg[:80]}")
            if "rate" in error_msg.lower() or "429" in str(response.status_code):
                wait = 10 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise ValueError(f"Groq API error: {error_msg}")

    raise ValueError("Max retries exceeded")

# ── Prompt Template ───────────────────────────────────

SYSTEM_PROMPT = """You are an expert financial analyst assistant specializing in SEC 10-K filings.
Answer questions using ONLY the provided context from SEC filings.
- If the answer is not in the context, say "The provided filings do not contain enough information to answer this."
- Always cite which company and filing date your answer comes from.
- Be specific and precise with numbers and facts.
- Do not speculate or add information not present in the context."""

def build_prompt(query: str, chunks: list) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        context_parts.append(
            f"[{i}] Company: {meta['company']} | "
            f"Section: {meta['section_name']} | "
            f"Date: {meta['filing_date']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""{SYSTEM_PROMPT}

CONTEXT FROM SEC FILINGS:
{context}

QUESTION: {query}

ANSWER (cite sources using [1], [2], etc.):"""


# ── RAG Pipeline ─────────────────────────────────────

class RAGPipeline:
    def __init__(self, llm: str = "ollama", db_path: str = "data/chromadb"):
        self.llm = llm
        self.retriever = DenseRetriever(db_path=db_path)
        print(f"Pipeline ready. LLM: {llm}")

    def query(self,
              question: str,
              k: int = 5,
              filters: dict = None,
              verbose: bool = True) -> dict:

        start = time.time()

        # Step 1: Retrieve
        chunks = self.retriever.retrieve(question, k=k, filters=filters)

        retrieval_time = time.time() - start

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUERY: {question}")
            print(f"{'='*60}")
            print(f"\nRetrieved {len(chunks)} chunks in {retrieval_time:.2f}s:")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] {c['metadata']['company']} | "
                      f"{c['metadata']['section_name']} | "
                      f"{c['metadata']['filing_date']} | "
                      f"Score: {c['score']:.4f}")

        # Step 2: Build prompt
        prompt = build_prompt(question, chunks)

        # Step 3: Generate
        gen_start = time.time()
        if verbose:
            print(f"\nGenerating answer with {self.llm}...")

        if self.llm == "ollama":
            answer = call_ollama(prompt)
        elif self.llm == "groq":
            answer = call_groq(prompt)
        else:
            raise ValueError(f"Unknown LLM: {self.llm}")

        gen_time = time.time() - gen_start
        total_time = time.time() - start

        if verbose:
            print(f"\nANSWER (generated in {gen_time:.1f}s):")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            print(f"Total latency: {total_time:.1f}s")

        return {
            "question":       question,
            "answer":         answer,
            "chunks":         chunks,
            "retrieval_time": round(retrieval_time, 3),
            "gen_time":       round(gen_time, 3),
            "total_time":     round(total_time, 3),
            "llm":            self.llm,
        }


# ── Test it ──────────────────────────────────────────

if __name__ == "__main__":
    pipeline = RAGPipeline(llm="groq")

    questions = [
        "What are the main risk factors for AMD related to competition?",
        "How did Abbott Laboratories describe their revenue sources?",
        "What cybersecurity risks does Adams Resources disclose?",
        "What was AMD's strategy for competing in the processor market?",
    ]

    for q in questions:
        result = pipeline.query(q, k=5)
        print("\n")