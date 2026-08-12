# SQuAD 2.0 retrieval eval harness: ingest a sample of SQuAD contexts, run
# every question in that sample through retrieve_chunks(), and report
# Recall@k (doc-level and answer-span) for baseline vs. re-ranked retrieval.
#
# No LLM calls are made - this measures retrieval quality only.

import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm

from app.db import get_connection, create_table, insert_chunk, delete_chunks_by_source, delete_chunks_by_prefix
from app.ingest import chunk_tokens, embed_chunks, CHUNK_SIZE, OVERLAP
from app.retrieve import retrieve_chunks
from eval.squad_data import DEFAULT_DATA_DIR, download_squad, load_squad, iter_contexts, sample_contexts
from eval.metrics import evaluate_question, aggregate

SQUAD_PREFIX = "squad::"
RESULTS_DIR = Path(__file__).parent / "results"


def source_doc_for(title: str, para_idx: int) -> str:
    return f"{SQUAD_PREFIX}{title}::{para_idx}"


def ingest_context(conn, source_doc: str, text: str, chunk_size: int, overlap: int) -> int:
    """Mirrors app.ingest.ingest_file's body minus the file read. Deletes any
    existing chunks for this source_doc first so reruns with the same seed
    are idempotent rather than accumulating duplicate rows."""
    tokens = text.split()
    chunks = chunk_tokens(tokens, chunk_size, overlap) if tokens else []
    chunk_texts = [" ".join(c) for c in chunks]
    embeddings = embed_chunks(chunk_texts)

    delete_chunks_by_source(conn, source_doc)
    for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
        insert_chunk(conn, source_doc=source_doc, chunk_index=i, content=chunk_text, embedding=embedding)
    conn.commit()

    return len(chunk_texts)


def run_retrieval(question: str, top_k: int, min_similarity: float | None, rerank: bool, rerank_candidates: int) -> dict:
    result = retrieve_chunks(question, top_k=top_k, min_similarity=min_similarity, rerank=rerank, rerank_candidates=rerank_candidates)
    return {
        "source_docs": [r["source_doc"] for r in result["results"]],
        "contents": [r["content"] for r in result["results"]],
    }


def print_summary(label: str, summary: dict, k_values: list[int]) -> None:
    print(f"\n{label} (n={summary['num_questions']}, answerable={summary['num_answerable']}, unanswerable={summary['num_unanswerable']})")
    print(f"  {'k':>4}  {'doc_recall':>11}  {'  (answerable)':>15}  {'(unanswerable)':>16}  {'doc_mrr':>8}  {'answer_recall':>14}")
    for k in k_values:
        row = summary["per_k"][k]
        def fmt(v):
            return f"{v:.3f}" if v is not None else "n/a"
        print(f"  {k:>4}  {fmt(row['doc_recall']):>11}  {fmt(row['doc_recall_answerable']):>15}  {fmt(row['doc_recall_unanswerable']):>16}  {fmt(row['doc_mrr']):>8}  {fmt(row['answer_recall']):>14}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the RAG pipeline's retrieval against a sample of SQuAD 2.0.")
    parser.add_argument("--num-contexts", type=int, default=40, help="Number of SQuAD contexts (paragraphs) to sample and ingest")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for context sampling")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth (ignored if --k-values is set)")
    parser.add_argument("--k-values", type=str, default=None, help="Comma-separated k values to report, e.g. '1,3,5'")
    parser.add_argument("--min-similarity", type=float, default=None, help="Passthrough to retrieve_chunks")
    parser.add_argument("--compare-rerank", dest="compare_rerank", action="store_true", default=True, help="Also run re-ranked retrieval and report both (default: on)")
    parser.add_argument("--no-compare-rerank", dest="compare_rerank", action="store_false", help="Skip the re-ranked retrieval pass")
    parser.add_argument("--rerank-candidates", type=int, default=20, help="Candidate pool size for re-ranking")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Passthrough to ingestion")
    parser.add_argument("--overlap", type=int, default=OVERLAP, help="Passthrough to ingestion")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Cache dir for the downloaded SQuAD json")
    parser.add_argument("--force-download", action="store_true", help="Re-download SQuAD json even if cached")
    parser.add_argument("--output", type=Path, default=None, help="Path to write the JSON report (default: eval/results/<timestamp>.json)")
    parser.add_argument("--cleanup", action="store_true", help="Delete all squad:: chunks from the table after writing results")
    args = parser.parse_args()

    k_values = sorted(int(k) for k in args.k_values.split(",")) if args.k_values else [args.top_k]
    max_k = max(k_values)
    if args.compare_rerank and max_k > args.rerank_candidates:
        print(f"Warning: max k ({max_k}) exceeds --rerank-candidates ({args.rerank_candidates}); clamping rerank-candidates to {max_k}.")
        args.rerank_candidates = max_k

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    output_path = args.output or (RESULTS_DIR / f"{timestamp}.json")

    path = download_squad(args.data_dir, force=args.force_download)
    contexts = iter_contexts(load_squad(path))
    sampled = sample_contexts(contexts, args.num_contexts, args.seed)

    conn = get_connection()
    try:
        create_table(conn)

        print(f"Ingesting {len(sampled)} sampled SQuAD contexts...")
        for ctx in tqdm(sampled, desc="Ingesting"):
            source_doc = source_doc_for(ctx["title"], ctx["para_idx"])
            ingest_context(conn, source_doc, ctx["context"], args.chunk_size, args.overlap)

        questions = []
        for ctx in sampled:
            gold_source_doc = source_doc_for(ctx["title"], ctx["para_idx"])
            for q in ctx["questions"]:
                questions.append({
                    "id": q["id"],
                    "question": q["question"],
                    "title": ctx["title"],
                    "para_idx": ctx["para_idx"],
                    "gold_source_doc": gold_source_doc,
                    "gold_answers": q["answers"],
                    "is_impossible": q["is_impossible"],
                })

        print(f"Running retrieval for {len(questions)} questions (compare_rerank={args.compare_rerank})...")
        baseline_records, reranked_records = [], []
        question_rows = []
        for q in tqdm(questions, desc="Retrieving"):
            baseline = run_retrieval(q["question"], max_k, args.min_similarity, rerank=False, rerank_candidates=args.rerank_candidates)
            baseline_eval = evaluate_question(
                baseline["source_docs"], baseline["contents"], q["gold_source_doc"],
                q["gold_answers"], q["is_impossible"], k_values,
            )
            baseline_records.append({"is_impossible": q["is_impossible"], "eval": baseline_eval})

            row = {
                "question_id": q["id"],
                "question": q["question"],
                "title": q["title"],
                "para_idx": q["para_idx"],
                "gold_source_doc": q["gold_source_doc"],
                "is_impossible": q["is_impossible"],
                "gold_answers": q["gold_answers"],
                "baseline": {"retrieved_source_docs": baseline["source_docs"], **baseline_eval},
            }

            if args.compare_rerank:
                reranked = run_retrieval(q["question"], max_k, args.min_similarity, rerank=True, rerank_candidates=args.rerank_candidates)
                reranked_eval = evaluate_question(
                    reranked["source_docs"], reranked["contents"], q["gold_source_doc"],
                    q["gold_answers"], q["is_impossible"], k_values,
                )
                reranked_records.append({"is_impossible": q["is_impossible"], "eval": reranked_eval})
                row["reranked"] = {"retrieved_source_docs": reranked["source_docs"], **reranked_eval}

            question_rows.append(row)

        baseline_summary = aggregate(baseline_records, k_values)
        reranked_summary = aggregate(reranked_records, k_values) if args.compare_rerank else None

        report = {
            "meta": {
                "timestamp": timestamp,
                "seed": args.seed,
                "num_contexts": len(sampled),
                "num_questions": len(questions),
                "k_values": k_values,
                "min_similarity": args.min_similarity,
                "compare_rerank": args.compare_rerank,
                "rerank_candidates": args.rerank_candidates,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "squad_source": str(path),
            },
            "summary": {
                "baseline": baseline_summary,
                "reranked": reranked_summary,
            },
            "questions": question_rows,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print_summary("Baseline", baseline_summary, k_values)
        if reranked_summary:
            print_summary("Reranked", reranked_summary, k_values)
        print(f"\nReport written to {output_path}")

        if args.cleanup:
            removed = delete_chunks_by_prefix(conn, SQUAD_PREFIX)
            conn.commit()
            print(f"Cleanup: removed {removed} squad:: chunk(s) from the table.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
