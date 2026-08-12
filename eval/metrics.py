# Pure Recall@k computation against retrieve_chunks() output. No I/O here.

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def contains_answer(content: str, gold_answers: list[str]) -> bool:
    normalized_content = normalize_text(content)
    return any(normalize_text(a) in normalized_content for a in gold_answers)


def reciprocal_rank(ranked_source_docs: list[str], gold_source_doc: str) -> float:
    if gold_source_doc in ranked_source_docs:
        return 1.0 / (ranked_source_docs.index(gold_source_doc) + 1)
    return 0.0


def evaluate_question(
    ranked_source_docs: list[str],
    ranked_contents: list[str],
    gold_source_doc: str,
    gold_answers: list[str],
    is_impossible: bool,
    k_values: list[int],
) -> dict:
    """doc_recall and doc_mrr are defined for every question (answerable or
    not) since each question is authored against one specific paragraph.
    answer_recall is undefined (None) for unanswerable questions, which have
    no gold answer text to search for."""
    doc_recall = {}
    doc_mrr = {}
    answer_recall = {}
    for k in k_values:
        doc_recall[k] = gold_source_doc in ranked_source_docs[:k]
        doc_mrr[k] = reciprocal_rank(ranked_source_docs[:k], gold_source_doc)
        if is_impossible or not gold_answers:
            answer_recall[k] = None
        else:
            answer_recall[k] = any(contains_answer(c, gold_answers) for c in ranked_contents[:k])
    return {"doc_recall": doc_recall, "doc_mrr": doc_mrr, "answer_recall": answer_recall}


def aggregate(question_records: list[dict], k_values: list[int]) -> dict:
    """question_records: list of {"is_impossible": bool, "eval": evaluate_question(...) result}."""
    num_questions = len(question_records)
    answerable = [r for r in question_records if not r["is_impossible"]]
    unanswerable = [r for r in question_records if r["is_impossible"]]

    def doc_recall_rate(records, k):
        if not records:
            return None
        return sum(r["eval"]["doc_recall"][k] for r in records) / len(records)

    def doc_mrr_rate(records, k):
        if not records:
            return None
        return sum(r["eval"]["doc_mrr"][k] for r in records) / len(records)

    def answer_recall_rate(records, k):
        scored = [r["eval"]["answer_recall"][k] for r in records if r["eval"]["answer_recall"][k] is not None]
        if not scored:
            return None
        return sum(scored) / len(scored)

    per_k = {}
    for k in k_values:
        per_k[k] = {
            "doc_recall": doc_recall_rate(question_records, k),
            "doc_recall_answerable": doc_recall_rate(answerable, k),
            "doc_recall_unanswerable": doc_recall_rate(unanswerable, k),
            "doc_mrr": doc_mrr_rate(question_records, k),
            "doc_mrr_answerable": doc_mrr_rate(answerable, k),
            "doc_mrr_unanswerable": doc_mrr_rate(unanswerable, k),
            "answer_recall": answer_recall_rate(question_records, k),
        }

    return {
        "num_questions": num_questions,
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "per_k": per_k,
    }
