from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer

from app.db import get_connection, create_table, insert_chunk, delete_chunks_by_source

# Chunking knobs - tuned small for the current tiny sample docs; bump toward
# ~500/50 once ingesting real-sized documents.
CHUNK_SIZE = 50
OVERLAP = 10

SUPPORTED_EXTENSIONS = {".txt", ".md"}

_model: SentenceTransformer | None = None


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def chunk_tokens(tokens: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[list[str]]:
    chunks = []
    step = chunk_size - overlap
    i = 0
    while i < len(tokens):
        chunks.append(tokens[i:i + chunk_size])
        if i + chunk_size >= len(tokens):
            break
        i += step
    return chunks


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_chunks(chunk_texts: list[str]) -> list[list[float]]:
    if not chunk_texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(chunk_texts)
    return embeddings.tolist()


def ingest_text(conn, source_doc: str, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> dict:
    """Chunk, embed, and store already-loaded text under source_doc using the
    given connection. Shared core for both disk-based and upload-based ingestion."""
    tokens = text.split()
    chunks = chunk_tokens(tokens, chunk_size, overlap) if tokens else []
    chunk_texts = [" ".join(chunk) for chunk in chunks]
    embeddings = embed_chunks(chunk_texts)

    delete_chunks_by_source(conn, source_doc)
    for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
        insert_chunk(conn, source_doc=source_doc, chunk_index=i, content=chunk_text, embedding=embedding)
    conn.commit()

    return {"path": source_doc, "chunks_inserted": len(chunk_texts)}


def ingest_file(conn, path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> dict:
    """Ingest one already-validated file using the given connection.
    Caller (ingest_path) is responsible for confirming the file exists and
    has a supported extension."""
    source_doc = str(path.resolve())
    text = load_text(source_doc)
    return ingest_text(conn, source_doc, text, chunk_size, overlap)


def ingest_uploads(files: list[tuple[str, bytes]], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> dict:
    """Ingest a batch of in-memory uploads (filename, raw_bytes) pairs.
    Mirrors ingest_path's directory-mode response shape."""
    conn = get_connection()
    try:
        create_table(conn)

        ingested, skipped = [], []
        for filename, raw in files:
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                skipped.append({"path": filename, "reason": f"unsupported extension '{suffix or '(none)'}'"})
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                skipped.append({"path": filename, "reason": f"failed to decode as UTF-8: {exc}"})
                continue
            try:
                ingested.append(ingest_text(conn, filename, text, chunk_size, overlap))
            except Exception as exc:
                skipped.append({"path": filename, "reason": f"failed to ingest: {exc}"})

        return {
            "path": "<upload>",
            "mode": "upload",
            "files_ingested": ingested,
            "files_skipped": skipped,
            "total_files_ingested": len(ingested),
            "total_chunks_inserted": sum(r["chunks_inserted"] for r in ingested),
        }
    finally:
        conn.close()


def ingest_path(path_str: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> dict:
    """Single public entry point used by both the /ingest endpoint and the
    CLI. Auto-detects file vs directory and ingests accordingly.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: path exists but is unsupported (bad extension on a
            direct file path, or neither a file nor a directory).
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path_str}")

    conn = get_connection()
    try:
        create_table(conn)

        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type '{path.suffix or '(none)'}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            result = ingest_file(conn, path, chunk_size, overlap)
            return {
                "path": str(path.resolve()),
                "mode": "file",
                "files_ingested": [result],
                "files_skipped": [],
                "total_files_ingested": 1,
                "total_chunks_inserted": result["chunks_inserted"],
            }

        if path.is_dir():
            candidates = sorted(p for p in path.iterdir() if p.is_file())
            ingested, skipped = [], []
            for p in candidates:
                if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    skipped.append({"path": str(p.resolve()), "reason": f"unsupported extension '{p.suffix or '(none)'}'"})
                    continue
                try:
                    ingested.append(ingest_file(conn, p, chunk_size, overlap))
                except Exception as exc:
                    skipped.append({"path": str(p.resolve()), "reason": f"failed to ingest: {exc}"})

            return {
                "path": str(path.resolve()),
                "mode": "directory",
                "files_ingested": ingested,
                "files_skipped": skipped,
                "total_files_ingested": len(ingested),
                "total_chunks_inserted": sum(r["chunks_inserted"] for r in ingested),
            }

        raise ValueError(f"Path exists but is neither a file nor a directory: {path_str}")
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a file or directory into the RAG chunk store.")
    parser.add_argument("path", help="Path to a .txt/.md file, or a directory containing such files")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    args = parser.parse_args()

    try:
        summary = ingest_path(args.path, chunk_size=args.chunk_size, overlap=args.overlap)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Error: {e}")

    print(f"Mode: {summary['mode']} ({summary['path']})")
    print(f"Ingested {summary['total_files_ingested']} file(s), {summary['total_chunks_inserted']} chunk(s) total.")
    for f in summary["files_ingested"]:
        print(f"  [ok]      {f['path']} -> {f['chunks_inserted']} chunks")
    for f in summary["files_skipped"]:
        print(f"  [skipped] {f['path']} ({f['reason']})")
