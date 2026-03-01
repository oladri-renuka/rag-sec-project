# src/retrieval/dense_retriever.py
import pickle
import numpy as np
import chromadb
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.ingestion.chunker import Chunk

def build_index(
    chunks_path: str = "data/chunks.pkl",
    embeddings_path: str = "data/embeddings.npz",
    db_path: str = "data/chromadb"
):
    # Load chunks and embeddings
    print("Loading chunks and embeddings...")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    print(f"Loaded {len(chunks):,} chunks, embeddings shape: {embeddings.shape}")

    # Init ChromaDB
    print(f"\nBuilding ChromaDB index at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("sec_filings")
        print("Deleted existing collection")
    except:
        pass

    collection = client.create_collection(
        name="sec_filings",
        metadata={"hnsw:space": "cosine"}
    )

    # Insert in batches of 500 (ChromaDB limit)
    BATCH = 500
    total = len(chunks)
    start = time.time()

    for i in range(0, total, BATCH):
        batch_chunks = chunks[i:i+BATCH]
        batch_embeddings = embeddings[i:i+BATCH]

        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeddings.tolist(),
            documents=[c.text for c in batch_chunks],
            metadatas=[{
                "company":      c.metadata["company"],
                "ticker":       c.metadata["ticker"],
                "section":      c.metadata["section"],
                "section_name": c.metadata["section_name"],
                "filing_date":  c.metadata["filing_date"],
                "doc_id":       c.metadata["doc_id"],
            } for c in batch_chunks]
        )

        if (i // BATCH) % 5 == 0:
            pct = min(100, (i + BATCH) / total * 100)
            print(f"  {pct:.0f}% — inserted {min(i+BATCH, total):,}/{total:,}")

    elapsed = time.time() - start
    print(f"\nIndex built in {elapsed:.1f}s")
    print(f"Collection size: {collection.count():,} chunks")
    return collection


class DenseRetriever:
    def __init__(self, db_path: str = "data/chromadb"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("sec_filings")
        self.instruction = "Represent this financial question for retrieving relevant documents: "
        print(f"Retriever ready. Index has {self.collection.count():,} chunks.")

    def retrieve(self, query: str, k: int = 5, filters: dict = None):
        # Embed query with instruction prefix
        query_embedding = self.model.encode(
            self.instruction + query,
            normalize_embeddings=True
        ).tolist()

        # Build where clause for metadata filtering
        where = None
        if filters:
            # e.g. filters={"company": "ADVANCED MICRO DEVICES INC"}
            # or   filters={"section": "section_1A"}
            if len(filters) == 1:
                key, val = list(filters.items())[0]
                where = {key: {"$eq": val}}
            else:
                where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text":     doc,
                "metadata": meta,
                "score":    round(1 - dist, 4)  # convert distance to similarity
            })

        return retrieved


if __name__ == "__main__":
    # Step 1: Build the index
    build_index()

    # Step 2: Test retrieval
    print("\n" + "="*55)
    print("TESTING RETRIEVAL")
    print("="*55)

    retriever = DenseRetriever()

    test_queries = [
        "What are the main risk factors for AMD?",
        "How did Abbott Laboratories perform in terms of revenue?",
        "What cybersecurity risks does the company disclose?",
        "What is the company strategy for growth?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 55)
        results = retriever.retrieve(query, k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Score: {r['score']:.4f}")
            print(f"       Company: {r['metadata']['company']}")
            print(f"       Section: {r['metadata']['section_name']}")
            print(f"       Date:    {r['metadata']['filing_date']}")
            print(f"       Text:    {r['text'][:150]}...")
            print()

    # Step 3: Test section filtering
    print("="*55)
    print("TESTING FILTERED RETRIEVAL (Risk Factors only)")
    print("="*55)
    results = retriever.retrieve(
        "What operational risks does the company face?",
        k=3,
        filters={"section": "section_1A"}
    )
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['metadata']['company']} | Score: {r['score']:.4f}")
        print(f"       {r['text'][:200]}...")