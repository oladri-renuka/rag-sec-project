# data.py
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
ds = load_dataset("JanosAudran/financial-reports-sec", "small_full", split="train")
print(f"Loaded {len(ds)} sentences\n")

# Convert to pandas for easy exploration
df = ds.to_pandas()

# ── Basic stats ──
print("=" * 50)
print("BASIC STATS")
print("=" * 50)
print(f"Unique companies:  {df['name'].nunique()}")
print(f"Unique filings:    {df['docID'].nunique()}")
print(f"Date range:        {df['filingDate'].min()} → {df['filingDate'].max()}")

# ── Section breakdown ──
print("\n" + "=" * 50)
print("SENTENCES PER SECTION")
print("=" * 50)
section_map = {
    0: "section_1  (Business)",
    1: "section_10",
    2: "section_11",
    3: "section_12",
    4: "section_13",
    5: "section_14",
    6: "section_15",
    7: "section_1A (Risk Factors)",
    8: "section_1B",
    9: "section_2  (Properties)",
    10: "section_3  (Legal)",
    11: "section_4",
    12: "section_5",
    13: "section_6",
    14: "section_7  (MD&A)",
    15: "section_7A (Market Risk)",
    16: "section_8  (Financials)",
    17: "section_9",
    18: "section_9A (Controls)",
    19: "section_9B",
}
section_counts = df['section'].value_counts().sort_index()
for idx, count in section_counts.items():
    label = section_map.get(idx, f"section_{idx}")
    print(f"  [{idx:2d}] {label:<35} {count:>7,} sentences")

# ── Top companies by sentence count ──
print("\n" + "=" * 50)
print("TOP 10 COMPANIES BY SENTENCE COUNT")
print("=" * 50)
top_companies = df.groupby('name')['sentence'].count().sort_values(ascending=False).head(10)
for company, count in top_companies.items():
    print(f"  {company:<40} {count:>6,}")

# ── Sample sentences from key sections ──
print("\n" + "=" * 50)
print("SAMPLE: Risk Factors (section_1A = index 7)")
print("=" * 50)
risk_samples = df[df['section'] == 7]['sentence'].sample(3, random_state=42).tolist()
for i, s in enumerate(risk_samples, 1):
    print(f"\n  [{i}] {s[:200]}...")

print("\n" + "=" * 50)
print("SAMPLE: MD&A (section_7 = index 14)")
print("=" * 50)
mda_samples = df[df['section'] == 14]['sentence'].sample(3, random_state=42).tolist()
for i, s in enumerate(mda_samples, 1):
    print(f"\n  [{i}] {s[:200]}...")

# ── Sentence length distribution ──
print("\n" + "=" * 50)
print("SENTENCE LENGTH STATS (words)")
print("=" * 50)
df['word_count'] = df['sentence'].str.split().str.len()
print(f"  Mean:   {df['word_count'].mean():.1f} words")
print(f"  Median: {df['word_count'].median():.1f} words")
print(f"  Min:    {df['word_count'].min()}")
print(f"  Max:    {df['word_count'].max()}")
print(f"  >100 words: {(df['word_count'] > 100).sum():,} sentences ({(df['word_count'] > 100).mean()*100:.1f}%)")

# ── Save a small sample for quick testing later ──
print("\n" + "=" * 50)
print("SAVING SAMPLE FOR QUICK TESTING")
print("=" * 50)

# Key sections only: Business(0), Risk(7), MDA(14), Financials(16)
key_sections = [0, 7, 14, 16]
sample_df = df[df['section'].isin(key_sections)].groupby(
    ['docID', 'section']
).head(20)  # First 20 sentences per filing per section

sample_df.to_csv("data/sample_key_sections.csv", index=False)
print(f"  Saved {len(sample_df):,} rows to data/sample_key_sections.csv")
print(f"  Covers {sample_df['docID'].nunique()} filings, 4 key sections")
print("\nDone! Ready to build chunks.")