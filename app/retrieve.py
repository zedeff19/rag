# 1. Embed the incoming query with the SAME embedding model used at ingest time
from app.db import get_connection, create_table
from app.ingest import get_embedding_model
from app.rerank import rerank_results


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    query_embedding = model.encode(query)
    return query_embedding.tolist()

# 2. Run a similarity search against pgvector (cosine distance operator <=>)
# 3. Return the top-k chunks; apply a similarity threshold so irrelevant
#    chunks aren't passed along to generation
def similarity_search(conn, query_embedding: list[float], top_k: int = 5, min_similarity: float = 0.3):
    max_distance = 1 - min_similarity
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT id, source_doc, chunk_index, content, embedding <=> %s::vector AS distance
            FROM chunks
        ) sub
        WHERE distance <= %s
        ORDER BY distance
        LIMIT %s;
        """,
        (query_embedding, max_distance, top_k),
    ).fetchall()
    return rows


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    min_similarity: float | None = None,
    rerank: bool = False,
    rerank_candidates: int = 20,
) -> dict:
    """Single public entry point used by both the /query endpoint and the
    CLI. Owns the connection lifecycle end to end: embed -> search -> (optional
    rerank) -> format.

    top_k always means the final number of results returned. When rerank is
    True, a larger candidate pool (rerank_candidates) is pulled from pgvector
    first, then the cross-encoder trims it back down to top_k.

    min_similarity defaults to 0.0 when rerank=True and 0.3 when rerank=False,
    since cosine similarity and cross-encoder relevance don't always agree -
    a low pre-filter avoids discarding chunks the cross-encoder would rank
    highly. Pass an explicit value to override this."""
    if min_similarity is None:
        min_similarity = 0.0 if rerank else 0.3

    conn = get_connection()
    try:
        create_table(conn)
        query_embedding = embed_query(query)
        search_k = max(top_k, rerank_candidates) if rerank else top_k
        rows = similarity_search(conn, query_embedding, top_k=search_k, min_similarity=min_similarity)
    finally:
        conn.close()

    results = [
        {
            "id": id_,
            "source_doc": source_doc,
            "chunk_index": chunk_index,
            "content": content,
            "similarity": 1 - distance,
        }
        for id_, source_doc, chunk_index, content, distance in rows
    ]

    if rerank and results:
        results = rerank_results(query, results, top_n=top_k)

    return {"query": query, "results": results, "total_results": len(results), "reranked": rerank}


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Query the RAG chunk store.")
    parser.add_argument("query", help="Query text to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument("--min-similarity", type=float, default=None, help="Minimum cosine similarity (0-1). Defaults to 0.3, or 0.0 when --rerank is set.")
    parser.add_argument("--rerank", action="store_true", help="Re-rank candidates with a cross-encoder before returning")
    parser.add_argument("--rerank-candidates", type=int, default=20, help="Candidate pool size pulled from vector search before re-ranking (only used with --rerank)")
    args = parser.parse_args()

    summary = retrieve_chunks(
        args.query, top_k=args.top_k, min_similarity=args.min_similarity,
        rerank=args.rerank, rerank_candidates=args.rerank_candidates,
    )

    print(f"{summary['total_results']} chunk(s) matched \"{summary['query']}\" (reranked={summary['reranked']})\n")
    for r in summary["results"]:
        if "rerank_score" in r:
            print(f"[sim {r['similarity']:.3f} | rerank {r['rerank_score']:.3f}] {r['source_doc']} (chunk {r['chunk_index']}): {r['content'][:100]}...")
        else:
            print(f"[{r['similarity']:.3f}] {r['source_doc']} (chunk {r['chunk_index']}): {r['content'][:100]}...")
