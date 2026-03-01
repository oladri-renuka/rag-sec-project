# src/embeddings/embedder.py
import pickle
import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List

# Import our Chunk class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.ingestion.chunker import Chunk

def embed_chunks(
    chunks: List[Chunk],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    save_path: str = "data/embeddings.npz"
) -> np.ndarray:
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # BGE models need this instruction prefix for better retrieval
    instruction = "Represent this financial document sentence for retrieval: "
    texts = [instruction + chunk.text for chunk in chunks]
    
    print(f"Embedding {len(texts):,} chunks...")
    print(f"Batch size: {batch_size} | Estimated time: ~{len(texts) // batch_size // 3} min")
    
    start = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity
    )
    
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"  → {embeddings.shape[0]:,} chunks")
    print(f"  → {embeddings.shape[1]} dimensions per chunk")
    
    # Save embeddings + chunk IDs together
    chunk_ids = np.array([c.chunk_id for c in chunks])
    np.savez(save_path, 
             embeddings=embeddings, 
             chunk_ids=chunk_ids)
    print(f"\nSaved to {save_path}")
    print(f"File size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
    
    return embeddings


if __name__ == "__main__":
    # Load chunks
    print("Loading chunks...")
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks):,} chunks")

    # Install sentence-transformers if needed
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer

    # Embed
    embeddings = embed_chunks(
        chunks,
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=64,
        save_path="data/embeddings.npz"
    )

    # Quick sanity check
    print("\n" + "="*50)
    print("SANITY CHECK")
    print("="*50)
    
    # Check a similarity between two chunks from the same company
    amd_indices = [i for i, c in enumerate(chunks) 
                   if c.metadata['company'] == 'ADVANCED MICRO DEVICES INC'][:2]
    
    other_indices = [i for i, c in enumerate(chunks) 
                     if c.metadata['company'] == 'WORLDS INC'][:2]
    
    if len(amd_indices) >= 2 and len(other_indices) >= 2:
        same_sim = np.dot(embeddings[amd_indices[0]], embeddings[amd_indices[1]])
        diff_sim = np.dot(embeddings[amd_indices[0]], embeddings[other_indices[0]])
        print(f"Similarity: AMD chunk vs AMD chunk (same co.): {same_sim:.4f}")
        print(f"Similarity: AMD chunk vs WORLDS chunk (diff co.): {diff_sim:.4f}")
        print(f"→ Same company should score higher ✓" if same_sim > diff_sim else "→ Unexpected result, check data")