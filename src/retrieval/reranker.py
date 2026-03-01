# src/retrieval/reranker.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import DenseRetriever

class RerankerRetriever:
    """
    Hybrid RRF → Cross-encoder reranking pipeline.
    Step 1: Get top-20 candidates from Hybrid retriever
    Step 2: Score each candidate with cross-encoder (joint query+chunk scoring)
    Step 3: Return top-k reranked results
    """
    def __init__(self, db_path: str = "data/chromadb",
                 chunks_path: str = "data/chunks.pkl"):
        from sentence_transformers import CrossEncoder
        
        print("Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512
        )
        self.hybrid = HybridRetriever(db_path=db_path, chunks_path=chunks_path)
        print("Reranker ready.")

    def retrieve(self, query: str, k: int = 5,
                 fetch: int = 20, filters: dict = None) -> list:
        # Step 1: Get top-20 from hybrid
        candidates = self.hybrid.retrieve(query, k=fetch,
                                          fetch=fetch, filters=filters)

        if not candidates:
            return []

        # Step 2: Score all candidates with cross-encoder
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Step 3: Attach scores and sort
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = round(float(scores[i]), 4)

        reranked = sorted(candidates,
                          key=lambda x: x["rerank_score"],
                          reverse=True)

        return reranked[:k]


if __name__ == "__main__":
    import time

    print("Initializing all retrievers for comparison...\n")
    dense   = DenseRetriever()
    hybrid  = HybridRetriever()
    reranker = RerankerRetriever()

    test_queries = [
        "What are the main risk factors for AMD related to competition?",
        "How did Abbott Laboratories describe their revenue sources?",
        "What cybersecurity risks does Adams Resources disclose?",
        "What was AMD's strategy for competing in the processor market?",
    ]

    print("\n" + "="*70)
    print("FULL COMPARISON: Dense vs Hybrid vs Reranked")
    print("="*70)

    for query in test_queries:
        print(f"\nQUERY: '{query}'")
        print("-"*70)

        t0 = time.time()
        d_results = dense.retrieve(query, k=3)
        d_time = time.time() - t0

        t0 = time.time()
        h_results = hybrid.retrieve(query, k=3)
        h_time = time.time() - t0

        t0 = time.time()
        r_results = reranker.retrieve(query, k=3)
        r_time = time.time() - t0

        print(f"{'DENSE (' + f'{d_time:.2f}s)':<40} "
              f"{'HYBRID (' + f'{h_time:.2f}s)':<40} "
              f"{'RERANKED (' + f'{r_time:.2f}s)':<40}")
        print("-"*70)

        for i in range(3):
            d = d_results[i] if i < len(d_results) else None
            h = h_results[i] if i < len(h_results) else None
            r = r_results[i] if i < len(r_results) else None

            d_str = f"{d['metadata']['company'][:18]} {d['metadata']['filing_date']}" if d else ""
            h_str = f"{h['metadata']['company'][:18]} {h['metadata']['filing_date']}" if h else ""
            r_str = f"{r['metadata']['company'][:18]} {r['metadata']['filing_date']} ({r['rerank_score']:.2f})" if r else ""

            print(f"[{i+1}] {d_str:<38} {h_str:<38} {r_str:<38}")