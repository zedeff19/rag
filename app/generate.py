# 1. Take the retrieved chunks + the user's question
# 2. Build a grounded prompt (chunks as numbered context, ask the model to
#    cite which chunks it used)
# 3. Call the Gemini API and return the answer + which chunks it was grounded in

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import settings
from app.retrieve import retrieve_chunks

SYSTEM_PROMPT = """You are a question-answering assistant that must stay strictly grounded \
in the provided context.

Rules:
- Answer using ONLY information found in the numbered context chunks below. Do not use any \
outside knowledge, even if you know the answer.
- If the context does not contain enough information to answer, say so explicitly instead of \
guessing.
- In `cited_chunk_numbers`, list the numbers of every context chunk you actually relied on to \
form the answer. If you could not answer from the context, return an empty list.
"""


class GeneratedAnswer(BaseModel):
    answer: str
    cited_chunk_numbers: list[int]


def build_context_block(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] (source: {r['source_doc']}, chunk {r['chunk_index']}): {r['content']}")
    return "\n\n".join(lines)


def call_gemini(query: str, context_block: str) -> GeneratedAnswer:
    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=f"Context:\n{context_block}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GeneratedAnswer,
        ),
    )
    return GeneratedAnswer.model_validate_json(response.text)


def generate_answer(
    query: str,
    top_k: int = 5,
    min_similarity: float | None = None,
    rerank: bool = False,
    rerank_candidates: int = 20,
) -> dict:
    """Single public entry point used by both the /ask endpoint and the CLI.
    Owns retrieval -> prompt -> Gemini -> citation mapping end to end."""
    retrieval = retrieve_chunks(
        query, top_k=top_k, min_similarity=min_similarity,
        rerank=rerank, rerank_candidates=rerank_candidates,
    )
    results = retrieval["results"]

    if not results:
        return {
            "query": query,
            "answer": "No relevant information was found in the knowledge base to answer this question.",
            "sources": [],
        }

    context_block = build_context_block(results)
    parsed = call_gemini(query, context_block)

    sources = [
        results[n - 1]
        for n in parsed.cited_chunk_numbers
        if 1 <= n <= len(results)
    ]

    return {"query": query, "answer": parsed.answer, "sources": sources}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask a grounded question against the RAG chunk store.")
    parser.add_argument("query", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--min-similarity", type=float, default=None, help="Minimum cosine similarity (0-1). Defaults to 0.3, or 0.0 when --rerank is set.")
    parser.add_argument("--rerank", action="store_true", help="Re-rank retrieved candidates with a cross-encoder before use")
    parser.add_argument("--rerank-candidates", type=int, default=20, help="Candidate pool size pulled from vector search before re-ranking (only used with --rerank)")
    args = parser.parse_args()

    result = generate_answer(
        args.query, top_k=args.top_k, min_similarity=args.min_similarity,
        rerank=args.rerank, rerank_candidates=args.rerank_candidates,
    )

    print(f"Q: {result['query']}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            if "rerank_score" in s:
                print(f"  [sim {s['similarity']:.3f} | rerank {s['rerank_score']:.3f}] {s['source_doc']} (chunk {s['chunk_index']}): {s['content'][:100]}...")
            else:
                print(f"  [{s['similarity']:.3f}] {s['source_doc']} (chunk {s['chunk_index']}): {s['content'][:100]}...")
    else:
        print("Sources: none")
