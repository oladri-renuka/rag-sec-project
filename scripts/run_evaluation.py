# scripts/run_evaluation.py
import json
import os
import sys
import time
import csv
from collections import defaultdict

sys.path.insert(0, '.')
from src.ingestion.chunker import Chunk  # noqa
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerRetriever
from src.generation.pipeline import build_prompt, call_ollama
# ── Simple metrics (no API needed) ──────────────────────

def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L: longest common subsequence recall."""
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
    if not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / len(pred_tokens) if pred_tokens else 0
    recall    = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def context_precision(chunks: list, question: str, company: str) -> float:
    """
    Simple proxy: fraction of retrieved chunks from the correct company.
    For cross-company questions returns 1.0 if any relevant chunk found.
    """
    if company == "MULTIPLE":
        return 1.0
    relevant = sum(
        1 for c in chunks
        if c["metadata"].get("company", "") == company
    )
    return round(relevant / len(chunks), 4) if chunks else 0.0


def answer_relevancy_proxy(answer: str, question: str) -> float:
    """
    Simple proxy: check key question words appear in answer.
    """
    question_words = set(question.lower().split()) - {
        "what", "how", "did", "does", "the", "a", "an", "is",
        "are", "was", "were", "their", "its", "for", "in", "of",
        "to", "and", "or", "do", "which", "when", "who"
    }
    if not question_words:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for w in question_words if w in answer_lower)
    return round(found / len(question_words), 4)


def faithfulness_proxy(answer: str, chunks: list) -> float:
    """
    Proxy: fraction of answer sentences supported by at least one chunk.
    """
    context = " ".join(c["text"].lower() for c in chunks)
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]
    if not sentences:
        return 1.0
    supported = 0
    for sent in sentences:
        words = set(sent.lower().split()) - {"the", "a", "an", "is", "are",
                                              "was", "were", "in", "of", "to"}
        if not words:
            continue
        overlap = sum(1 for w in words if w in context)
        if overlap / len(words) > 0.4:
            supported += 1
    return round(supported / len(sentences), 4)


# ── Main Evaluation ──────────────────────────────────────

def evaluate_retriever(retriever, questions: list,
                       retriever_name: str, k: int = 5) -> list:
    results = []
    print(f"\n{'='*60}")
    print(f"Evaluating: {retriever_name}")
    print(f"{'='*60}")

    for i, q in enumerate(questions):
        print(f"  [{i+1:02d}/{len(questions)}] {q['question'][:60]}...")
        t0 = time.time()

        # Retrieve
        try:
            chunks = retriever.retrieve(q["question"], k=k)
        except Exception as e:
            print(f"    ERROR retrieving: {e}")
            continue

        retrieval_time = time.time() - t0

        # Generate answer with Groq
        t1 = time.time()
        try:
            prompt  = build_prompt(q["question"], chunks)
            answer = call_ollama(prompt)
        except Exception as e:
            print(f"    ERROR generating: {e}")
            answer = ""

        gen_time = time.time() - t1

        # Compute metrics
        rl    = rouge_l(answer, q["ground_truth"])
        cp    = context_precision(chunks, q["question"], q["company"])
        ar    = answer_relevancy_proxy(answer, q["question"])
        faith = faithfulness_proxy(answer, chunks)

        result = {
            "retriever":        retriever_name,
            "question_id":      q["id"],
            "question":         q["question"],
            "company":          q["company"],
            "difficulty":       q["difficulty"],
            "section":          q["section"],
            "ground_truth":     q["ground_truth"],
            "answer":           answer,
            "rouge_l":          rl,
            "context_precision": cp,
            "answer_relevancy": ar,
            "faithfulness":     faith,
            "retrieval_time":   round(retrieval_time, 3),
            "gen_time":         round(gen_time, 3),
            "total_time":       round(retrieval_time + gen_time, 3),
        }
        results.append(result)

        print(f"    ROUGE-L: {rl:.3f} | CtxPrec: {cp:.3f} | "
              f"Relevancy: {ar:.3f} | Faith: {faith:.3f} | "
              f"Time: {retrieval_time + gen_time:.1f}s")

        # Small delay to avoid Groq rate limits
        time.sleep(0) 

    return results


def print_summary(all_results: list):
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)

    # Group by retriever
    by_retriever = defaultdict(list)
    for r in all_results:
        by_retriever[r["retriever"]].append(r)

    metrics = ["rouge_l", "context_precision", "answer_relevancy",
               "faithfulness", "retrieval_time"]

    # Header
    print(f"\n{'Retriever':<25} {'ROUGE-L':>8} {'CtxPrec':>8} "
          f"{'Relevancy':>10} {'Faith':>8} {'Ret(s)':>8}")
    print("-" * 70)

    for retriever_name, results in by_retriever.items():
        avgs = {m: sum(r[m] for r in results) / len(results)
                for m in metrics}
        print(f"{retriever_name:<25} "
              f"{avgs['rouge_l']:>8.4f} "
              f"{avgs['context_precision']:>8.4f} "
              f"{avgs['answer_relevancy']:>10.4f} "
              f"{avgs['faithfulness']:>8.4f} "
              f"{avgs['retrieval_time']:>8.3f}s")

    # By difficulty
    print(f"\n{'— By Difficulty —'}")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in all_results if r["difficulty"] == diff]
        if not diff_results:
            continue
        avg_rl = sum(r["rouge_l"] for r in diff_results) / len(diff_results)
        avg_cp = sum(r["context_precision"] for r in diff_results) / len(diff_results)
        print(f"  {diff:<8} ROUGE-L: {avg_rl:.4f} | CtxPrec: {avg_cp:.4f} "
              f"({len(diff_results)} questions)")


if __name__ == "__main__":
    # Load golden dataset
    with open("data/golden_dataset/questions.json") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    # Init retrievers
    print("\nInitializing retrievers...")
    dense    = DenseRetriever()
    bm25     = BM25Retriever()
    hybrid   = HybridRetriever()
    reranker = RerankerRetriever()

    retrievers = [
        ("Dense",    dense),
        ("BM25",     bm25),
        ("Hybrid",   hybrid),
        ("Reranked", reranker),
    ]

    all_results = []
    for name, retriever in retrievers:
        results = evaluate_retriever(retriever, questions, name, k=5)
        all_results.extend(results)

    # Save full results
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV for easy analysis
    if all_results:
        keys = list(all_results[0].keys())
        with open("experiments/results/metrics_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    print_summary(all_results)
    print(f"\nSaved results to experiments/results/")