# src/retrieval/hybrid_retriever.py
import os
import sys
from collections import defaultdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever

class HybridRetriever:
    """
    Reciprocal Rank Fusion of dense + BM25 retrieval.
    RRF score = sum(1 / (k + rank)) across all retrievers.
    """
    def __init__(self, k: int = 60, db_path: str = "data/chromadb",
                 chunks_path: str = "data/chunks.pkl"):
        print("Initializing Hybrid Retriever...")
        self.dense   = DenseRetriever(db_path=db_path)
        self.sparse  = BM25Retriever(chunks_path=chunks_path)
        self.k       = k  # RRF constant (60 is standard from original paper)
        print("Hybrid Retriever ready.")

    def retrieve(self, query: str, k: int = 5,
                 fetch: int = 20, filters: dict = None) -> list:
        """
        Args:
            query:   user question
            k:       final number of results to return
            fetch:   how many to get from each retriever before merging
            filters: metadata filters (passed to dense retriever)
        """
        # Step 1: Get top-fetch from each retriever
        dense_results  = self.dense.retrieve(query, k=fetch, filters=filters)
        sparse_results = self.sparse.retrieve(query, k=fetch, filters=filters)

        # Step 2: Build chunk_id → result lookup
        all_chunks = {}
        for r in dense_results + sparse_results:
            # Use doc_id + first 50 chars as key (chunk_id not in dense results)
            cid = r["metadata"]["doc_id"] + "|" + r["text"][:50]
            all_chunks[cid] = r

        # Step 3: RRF scoring
        rrf_scores = defaultdict(float)

        for rank, result in enumerate(dense_results):
            cid = result["metadata"]["doc_id"] + "|" + result["text"][:50]
            rrf_scores[cid] += 1 / (self.k + rank + 1)

        for rank, result in enumerate(sparse_results):
            cid = result["metadata"]["doc_id"] + "|" + result["text"][:50]
            rrf_scores[cid] += 1 / (self.k + rank + 1)

        # Step 4: Sort by RRF score, return top-k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, rrf_score in ranked[:k]:
            chunk = all_chunks[cid].copy()
            chunk["rrf_score"]  = round(rrf_score, 6)
            chunk["orig_score"] = chunk["score"]  # keep original score too
            results.append(chunk)

        return results


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "What are the main risk factors for AMD related to competition?",
        "How did Abbott Laboratories describe their revenue sources?",
        "What cybersecurity risks does Adams Resources disclose?",
        "What was AMD's strategy for competing in the processor market?",
    ]

    # Compare all three retrievers side by side
    dense  = DenseRetriever()
    sparse = BM25Retriever()

    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON: Dense vs BM25 vs Hybrid")
    print("="*70)

    for query in test_queries:
        print(f"\nQUERY: '{query}'")
        print("-"*70)

        d_results = dense.retrieve(query, k=3)
        b_results = sparse.retrieve(query, k=3)
        h_results = retriever.retrieve(query, k=3)

        print(f"{'DENSE':<40} {'BM25':<40} {'HYBRID (RRF)':<40}")
        print(f"{'-'*38:<40} {'-'*38:<40} {'-'*38:<40}")

        for i in range(3):
            d = d_results[i] if i < len(d_results) else None
            b = b_results[i] if i < len(b_results) else None
            h = h_results[i] if i < len(h_results) else None

            d_str = f"{d['metadata']['company'][:20]} | {d['metadata']['filing_date']}" if d else ""
            b_str = f"{b['metadata']['company'][:20]} | {b['metadata']['filing_date']}" if b else ""
            h_str = f"{h['metadata']['company'][:20]} | {h['metadata']['filing_date']}" if h else ""

            print(f"[{i+1}] {d_str:<38} {b_str:<38} {h_str:<38}")

        # Show hybrid scores
        print(f"\n  Hybrid RRF scores: ", end="")
        for r in h_results:
            print(f"{r['rrf_score']:.5f}", end="  ")
        print()