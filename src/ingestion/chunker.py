# src/ingestion/chunker.py
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Chunk:
    text: str
    chunk_id: str
    metadata: Dict = field(default_factory=dict)

# True section names extracted from sentenceID
SECTION_MAP = {
    'section_1':  'Business',
    'section_1A': 'Risk Factors',
    'section_1B': 'Unresolved Staff Comments',
    'section_2':  'Properties',
    'section_3':  'Legal Proceedings',
    'section_7':  'MD&A',
    'section_7A': 'Market Risk',
    'section_8':  'Financial Statements',
    'section_9A': 'Controls and Procedures',
}

# These are the sections we care about for RAG
KEY_SECTIONS = {'section_1', 'section_1A', 'section_7', 'section_7A', 'section_8'}

def extract_true_section(sentence_id: str) -> str:
    """
    Extract real section name from sentenceID.
    e.g. '0000001750_10-K_2020_section_1A_5' → 'section_1A'
    """
    parts = sentence_id.split('_')
    # Find the part that starts with 'section'
    for i, part in enumerate(parts):
        if part == 'section' and i + 1 < len(parts):
            return f"section_{parts[i+1]}"
    return 'unknown'

def build_chunks(df: pd.DataFrame, 
                 sentences_per_chunk: int = 8,
                 key_sections_only: bool = True) -> List[Chunk]:
    """
    Group consecutive sentences from same filing + section into chunks.
    
    Args:
        df: DataFrame from the dataset
        sentences_per_chunk: how many sentences per chunk
        key_sections_only: if True, only chunk Business/Risk/MDA/Financials
    """
    
    # Extract true section from sentenceID
    print("Extracting true section labels from sentenceID...")
    df = df.copy()
    df['true_section'] = df['sentenceID'].apply(extract_true_section)
    
    # Filter to key sections if requested
    if key_sections_only:
        df = df[df['true_section'].isin(KEY_SECTIONS)]
        print(f"Filtered to key sections: {len(df):,} sentences remaining")
    
    # Filter out very short sentences (noise)
    df['word_count'] = df['sentence'].str.split().str.len()
    df = df[df['word_count'] >= 5]
    print(f"After removing short sentences: {len(df):,} sentences")
    
    chunks = []
    groups = df.groupby(['docID', 'true_section'])
    print(f"Building chunks from {len(groups)} filing-section pairs...")
    
    for (doc_id, section), group in groups:
        # Sort by original sentence order
        group = group.sort_values('sentenceCount')
        sentences = group['sentence'].tolist()
        
        # Grab metadata from first row
        first_row = group.iloc[0]
        ticker = first_row['tickers'][0] if first_row['tickers'] else 'UNKNOWN'
        
        # Slide a window of N sentences
        for i in range(0, len(sentences), sentences_per_chunk):
            window = sentences[i : i + sentences_per_chunk]
            
            if len(window) < 3:  # skip tiny trailing chunks
                continue
            
            chunk_text = ' '.join(window)
            chunk_id = f"{doc_id}_{section}_chunk{i // sentences_per_chunk}"
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata={
                    'doc_id':       doc_id,
                    'section':      section,
                    'section_name': SECTION_MAP.get(section, section),
                    'company':      first_row['name'],
                    'ticker':       ticker,
                    'filing_date':  first_row['filingDate'],
                    'report_date':  first_row['reportDate'],
                    'cik':          first_row['cik'],
                    'label_1d':     int(first_row['labels']['1d']),
                    'label_30d':    int(first_row['labels']['30d']),
                    'chunk_index':  i // sentences_per_chunk,
                    'sentence_count': len(window),
                }
            )
            chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    from datasets import load_dataset
    import json

    print("Loading dataset...")
    ds = load_dataset("JanosAudran/financial-reports-sec", "small_full", split="train")
    df = ds.to_pandas()

    chunks = build_chunks(df, sentences_per_chunk=8, key_sections_only=True)

    print(f"\n{'='*50}")
    print(f"CHUNKING RESULTS")
    print(f"{'='*50}")
    print(f"Total chunks built:  {len(chunks):,}")
    
    # Stats by section
    from collections import Counter
    section_counts = Counter(c.metadata['section'] for c in chunks)
    print("\nChunks per section:")
    for section, count in sorted(section_counts.items()):
        name = SECTION_MAP.get(section, section)
        print(f"  {section:<15} ({name:<25}) {count:>5} chunks")
    
    # Stats by company
    print("\nChunks per company:")
    company_counts = Counter(c.metadata['company'] for c in chunks)
    for company, count in company_counts.most_common():
        print(f"  {company:<45} {count:>4} chunks")

    # Show a sample chunk
    print(f"\n{'='*50}")
    print("SAMPLE CHUNK")
    print(f"{'='*50}")
    sample = chunks[50]
    print(f"ID:      {sample.chunk_id}")
    print(f"Company: {sample.metadata['company']} ({sample.metadata['ticker']})")
    print(f"Section: {sample.metadata['section_name']}")
    print(f"Date:    {sample.metadata['filing_date']}")
    print(f"\nText preview:\n{sample.text[:400]}...")

    # Save chunks to disk
    import pickle, os
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"\nSaved {len(chunks):,} chunks to data/chunks.pkl")