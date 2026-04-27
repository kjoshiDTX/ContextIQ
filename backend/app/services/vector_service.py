"""Vector operations using pgvector in Supabase Postgres."""

import logging
import numpy as np
from psycopg2.extras import RealDictCursor, execute_values

from app.core.database import get_pg_connection

logger = logging.getLogger(__name__)


def upsert_chunks(
    document_id: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    metadata: list[dict] | None = None,
) -> int:
    """Insert document chunks with their vector embeddings into pgvector.

    Args:
        document_id: UUID of the global document
        chunks: List of text chunks
        embeddings: List of numpy arrays (384-dim each)
        metadata: Optional list of metadata dicts per chunk

    Returns:
        Number of chunks inserted
    """
    if not chunks:
        return 0

    if metadata is None:
        metadata = [{}] * len(chunks)

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # Build values for batch insert
            import json

            values = []
            for i, (chunk, emb, meta) in enumerate(zip(chunks, embeddings, metadata)):
                values.append((
                    document_id,
                    i,
                    chunk,
                    emb.tolist(),
                    json.dumps(meta),
                ))

            execute_values(
                cur,
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata)
                VALUES %s
                """,
                values,
                template="(%s, %s, %s, %s::vector, %s::jsonb)",
            )

    logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")
    return len(chunks)


def search_similar(
    query_embedding: np.ndarray,
    top_k: int = 10,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Search for similar chunks using cosine similarity.

    Args:
        query_embedding: Query vector (384-dim)
        top_k: Number of results to return
        document_ids: Optional filter to specific documents

    Returns:
        List of dicts with id, document_id, content, similarity
    """
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if document_ids:
                cur.execute(
                    """
                    SELECT id, document_id, chunk_index, content, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM document_chunks
                    WHERE document_id = ANY(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), document_ids, query_embedding.tolist(), top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT id, document_id, chunk_index, content, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), top_k),
                )
            results = cur.fetchall()

    return [dict(r) for r in results]


def delete_document_vectors(document_id: str) -> int:
    """Delete all chunks/vectors for a document.

    Returns:
        Number of chunks deleted
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %s",
                (document_id,),
            )
            count = cur.rowcount
    logger.info(f"Deleted {count} chunks for document {document_id}")
    return count
