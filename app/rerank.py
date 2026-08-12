# 4. (Optional) Re-rank retrieved chunks with a cross-encoder, which scores
#    the (query, chunk) pair jointly and is more precise than the cosine
#    similarity used for initial candidate retrieval - at the cost of being
#    too slow to run over the whole corpus, hence "retrieve wide, rerank down".

from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def get_reranker_model() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def rerank_results(query: str, results: list[dict], top_n: int | None = None) -> list[dict]:
    """Re-score retrieved chunks against the query with a cross-encoder and
    return them sorted by rerank_score descending. Each returned dict is a
    shallow copy of the corresponding input dict with a `rerank_score` key
    added (existing keys, e.g. `similarity`, are preserved unchanged). If
    top_n is given, only the top_n highest-scoring results are kept."""
    if not results:
        return []

    model = get_reranker_model()
    pairs = [(query, r["content"]) for r in results]
    scores = model.predict(pairs)

    scored = [{**r, "rerank_score": float(score)} for r, score in zip(results, scores)]
    scored.sort(key=lambda r: r["rerank_score"], reverse=True)

    if top_n is not None:
        scored = scored[:top_n]
    return scored
