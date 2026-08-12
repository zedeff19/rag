import os

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
load_dotenv()

def get_connection():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            source_doc TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL
        );
    """)
    conn.commit()


def insert_chunk(conn, source_doc: str, chunk_index: int, content: str, embedding: list[float]):
    conn.execute(
        "INSERT INTO chunks (source_doc, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
        (source_doc, chunk_index, content, embedding),
    )


def delete_chunks_by_source(conn, source_doc: str) -> int:
    cur = conn.execute("DELETE FROM chunks WHERE source_doc = %s", (source_doc,))
    return cur.rowcount


def delete_chunks_by_prefix(conn, prefix: str) -> int:
    cur = conn.execute("DELETE FROM chunks WHERE source_doc LIKE %s", (prefix + "%",))
    return cur.rowcount
