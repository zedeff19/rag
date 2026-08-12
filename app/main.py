# Run with: uvicorn app.main:app --reload
#
# Build order (see rag-anchor-project-plan.md, Phase 2):
#   1. POST /ingest - ingest a document or directory -> ingest.py [done]
#   2. POST /query  - retrieve top-k chunks -> retrieve.py [done]
#   3. POST /ask    - retrieve + generate a grounded answer -> generate.py [done]

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.generate import generate_answer
from app.ingest import ingest_path
from app.retrieve import retrieve_chunks

app = FastAPI(title="RAG Anchor API")


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path to ingest (.txt/.md supported).")


class IngestedFile(BaseModel):
    path: str
    chunks_inserted: int


class SkippedFile(BaseModel):
    path: str
    reason: str


class IngestResponse(BaseModel):
    path: str
    mode: str
    files_ingested: list[IngestedFile]
    files_skipped: list[SkippedFile]
    total_files_ingested: int
    total_chunks_inserted: int


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        return ingest_path(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ingestion failed due to an internal error.") from exc


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text to search for.")
    top_k: int = Field(5, ge=1, description="Number of chunks to return.")
    min_similarity: float | None = Field(None, ge=0.0, le=1.0, description="Minimum cosine similarity (0-1). Defaults to 0.3, or 0.0 when rerank=true.")
    rerank: bool = Field(False, description="Re-rank retrieved candidates with a cross-encoder before use.")
    rerank_candidates: int = Field(20, ge=1, description="Candidate pool size pulled from vector search before re-ranking (only used when rerank=true).")


class ChunkResult(BaseModel):
    id: int
    source_doc: str
    chunk_index: int
    content: str
    similarity: float
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total_results: int
    reranked: bool


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return retrieve_chunks(
            request.query, top_k=request.top_k, min_similarity=request.min_similarity,
            rerank=request.rerank, rerank_candidates=request.rerank_candidates,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Retrieval failed due to an internal error.") from exc


class AskRequest(BaseModel):
    query: str = Field(..., description="Question to ask.")
    top_k: int = Field(5, ge=1, description="Number of chunks to retrieve as context.")
    min_similarity: float | None = Field(None, ge=0.0, le=1.0, description="Minimum cosine similarity (0-1). Defaults to 0.3, or 0.0 when rerank=true.")
    rerank: bool = Field(False, description="Re-rank retrieved candidates with a cross-encoder before use.")
    rerank_candidates: int = Field(20, ge=1, description="Candidate pool size pulled from vector search before re-ranking (only used when rerank=true).")


class CitedSource(BaseModel):
    id: int
    source_doc: str
    chunk_index: int
    content: str
    similarity: float
    rerank_score: float | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[CitedSource]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return generate_answer(
            request.query, top_k=request.top_k, min_similarity=request.min_similarity,
            rerank=request.rerank, rerank_candidates=request.rerank_candidates,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Generation failed due to an internal error.") from exc
