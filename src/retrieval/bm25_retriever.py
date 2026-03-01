# src/retrieval/bm25_retriever.py
import pickle
import os
import sys
from rank_bm25 import BM25Okapi
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Must import Chunk so pickle can deserialize it
from src.ingestion.chunker import Chunk  # noqa: F401

def tokenize(text: str) -> list:
    """Simple financial-aware tokenizer."""
    text = text.lower()
    # Keep numbers and percentages intact — important for financials
    text = re.sub(r'[^\w\s\$\%\.]', ' ', text)
    tokens = text.split()
    # Remove very short tokens except numbers
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    return tokens

class BM25Retriever:
    def __init__(self, chunks_path: str = "data/chunks.pkl"):
        print("Building BM25 index...")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Tokenize all chunks
        corpus = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        print(f"BM25 index built over {len(self.chunks):,} chunks")

    def retrieve(self, query: str, k: int = 5, filters: dict = None) -> list:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filters manually
        if filters:
            for i, chunk in enumerate(self.chunks):
                for key, val in filters.items():
                    if chunk.metadata.get(key) != val:
                        scores[i] = 0.0

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text":     self.chunks[idx].text,
                    "metadata": self.chunks[idx].metadata,
                    "score":    round(float(scores[idx]), 4)
                })

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    test_queries = [
        "What are the main risk factors for AMD related to competition?",
        "How did Abbott Laboratories describe their revenue sources?",
        "What cybersecurity risks does Adams Resources disclose?",
        "What was AMD's strategy for competing in the processor market?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 55)
        results = retriever.retrieve(query, k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Score: {r['score']:.4f} | "
                  f"{r['metadata']['company']} | "
                  f"{r['metadata']['section_name']} | "
                  f"{r['metadata']['filing_date']}")
            print(f"       {r['text'][:150]}...")