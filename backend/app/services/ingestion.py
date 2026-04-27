"""Document ingestion pipeline: extract → chunk → embed → store vectors → extract triplets → store graph."""

import asyncio
import hashlib
import json
import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from psycopg2.extras import RealDictCursor

from app.core.config import get_settings
from app.core.database import get_supabase, get_pg_connection
from app.services.text_extraction import extract_text
from app.services.vector_service import upsert_chunks
from app.services.graph_service import store_triplets

logger = logging.getLogger(__name__)

# Lazy-loaded ML models — cached for the lifetime of the process
_embedding_model: SentenceTransformer | None = None
_hf_embeddings: HuggingFaceEmbeddings | None = None
_openai_client: OpenAI | None = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the sentence transformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_hf_embeddings() -> HuggingFaceEmbeddings:
    """Get or initialize the HuggingFace embeddings wrapper (used by SemanticChunker)."""
    global _hf_embeddings
    if _hf_embeddings is None:
        settings = get_settings()
        _hf_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _hf_embeddings


def get_gemini_client() -> OpenAI:
    """Get or initialize the Gemini client (OpenAI-compatible endpoint)."""
    global _gemini_client
    if _gemini_client is None:
        settings = get_settings()
        _gemini_client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _openai_client


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content for deduplication."""
    return hashlib.sha256(content).hexdigest()


def semantic_chunk_text(text: str) -> list[str]:
    """Split text into semantic chunks using LangChain SemanticChunker."""
    chunker = SemanticChunker(
        get_hf_embeddings(),
        breakpoint_threshold_type="percentile",
    )
    docs = chunker.create_documents([text])
    chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
    logger.info(f"Created {len(chunks)} semantic chunks")
    return chunks


def select_high_value_chunks(chunks: list[str]) -> list[str]:
    """Select high-value chunks for graph extraction to reduce LLM costs.

    Strategy: first 2, last 2, and top 30% by word density (information-rich chunks).
    """
    if len(chunks) <= 6:
        return chunks

    selected_indices = {0, 1, len(chunks) - 2, len(chunks) - 1}

    remaining = [(i, len(c.split())) for i, c in enumerate(chunks) if i not in selected_indices]
    remaining.sort(key=lambda x: x[1], reverse=True)

    top_count = max(1, int(len(remaining) * 0.3))
    for i, _ in remaining[:top_count]:
        selected_indices.add(i)

    selected = [chunks[i] for i in sorted(selected_indices)]
    logger.info(f"Selected {len(selected)}/{len(chunks)} high-value chunks for graph extraction")
    return selected


def extract_triplets_from_chunks(chunks: list[str]) -> list[dict]:
    """Extract entity-relationship triplets from text chunks via a single batched OpenAI call.

    Sends all chunks in one request with [CHUNK N] delimiters instead of N individual calls.
    Falls back to per-chunk extraction if the batched call fails.

    Returns list of {head, tail, head_type, tail_type, relation}.
    """
    if not chunks:
        return []

    settings = get_settings()
    client = get_openai_client()

    system_prompt = """You are a knowledge graph extraction expert. Extract entity-relationship triplets from the given text chunks.

For each triplet, identify:
- head: The source entity name
- tail: The target entity name
- head_type: One of [Company, Product, Technology, Person, Organization, Topic, Concept]
- tail_type: One of [Company, Product, Technology, Person, Organization, Topic, Concept]
- relation: One of [USES, PRODUCED_BY, HAS_PART, IS_A, RELATED_TO, COMPETES_WITH, WORKS_WITH, BELONGS_TO, DEPENDS_ON]

Return a JSON object {"triplets": [...]} covering ALL chunks combined.
Only extract factual, meaningful relationships. Avoid trivial or overly generic triplets.
If no meaningful triplets can be extracted, return {"triplets": []}."""

    sections = "\n\n".join(f"[CHUNK {i + 1}]\n{chunk[:2000]}" for i, chunk in enumerate(chunks))
    max_tokens = min(4000 * len(chunks), 16000)

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract triplets from ALL of these text chunks:\n\n{sections}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        parsed = json.loads(response.choices[0].message.content)
        triplets = parsed.get("triplets", []) if isinstance(parsed, dict) else parsed
        valid = [t for t in triplets if isinstance(t, dict) and "head" in t and "tail" in t]
        logger.info(f"Extracted {len(valid)} triplets from {len(chunks)} chunks in 1 API call")
        return valid

    except Exception as e:
        logger.warning(f"Batched triplet extraction failed: {e}. Falling back to per-chunk.")

    # Fallback: per-chunk extraction
    all_triplets = []
    for chunk in chunks:
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract triplets from this text:\n\n{chunk[:3000]}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=2000,
            )
            parsed = json.loads(response.choices[0].message.content)
            triplets = parsed.get("triplets", []) if isinstance(parsed, dict) else parsed
            all_triplets.extend([t for t in triplets if isinstance(t, dict) and "head" in t and "tail" in t])
        except Exception as chunk_e:
            logger.warning(f"Per-chunk triplet extraction failed: {chunk_e}")
            continue

    logger.info(f"Extracted {len(all_triplets)} triplets via per-chunk fallback")
    return all_triplets


async def ingest_document(
    file_content: bytes,
    filename: str,
    file_type: str,
    user_id: str,
    category_id: Optional[str] = None,
) -> dict:
    """Full document ingestion pipeline.

    Steps:
    1. Deduplicate via SHA-256 hash
    2. Extract text
    3. Semantic chunking
    4. Generate embeddings
    5. Store chunks + vectors in pgvector
    6. Upload file to Supabase Storage
    7. Extract triplets from high-value chunks (single batched API call)
    8. Store triplets in Neo4j (single UNWIND query)
    9. Auto-trigger community detection in background
    """
    file_hash = compute_file_hash(file_content)

    # Step 1: Deduplication
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, filename FROM global_documents WHERE file_hash = %s",
                (file_hash,),
            )
            existing = cur.fetchone()

            if existing:
                document_id = str(existing["id"])
                cur.execute(
                    """
                    INSERT INTO user_documents (user_id, document_id, category_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, document_id) DO NOTHING
                    """,
                    (user_id, document_id, category_id),
                )
                logger.info(f"Document '{filename}' already exists (hash match), linked to user")
                return {
                    "status": "duplicate_linked",
                    "document_id": document_id,
                    "filename": existing["filename"],
                    "message": "Document already exists. Linked to your library.",
                }

    # Step 2: Extract text
    logger.info(f"Extracting text from {filename} ({file_type})")
    text = extract_text(file_content, file_type)
    logger.info(f"Extracted {len(text)} characters")

    # Step 3: Semantic chunking
    chunks = semantic_chunk_text(text)

    # Step 4: Generate embeddings
    model = get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

    # Step 5: Create document record and store vectors
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO global_documents (file_hash, filename, file_type, chunk_count)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (file_hash, filename, file_type, len(chunks)),
            )
            document_id = str(cur.fetchone()["id"])

            cur.execute(
                """
                INSERT INTO user_documents (user_id, document_id, category_id)
                VALUES (%s, %s, %s)
                """,
                (user_id, document_id, category_id),
            )

    upsert_chunks(document_id, chunks, embeddings)

    # Step 6: Upload to Supabase Storage
    storage_path = None
    try:
        supabase = get_supabase()
        path = f"documents/{user_id}/{document_id}/{filename}"
        supabase.storage.from_("documents").upload(
            path=path,
            file=file_content,
            file_options={"content-type": f"application/{file_type}"},
        )
        storage_path = path
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE global_documents SET storage_path = %s WHERE id = %s",
                    (storage_path, document_id),
                )
    except Exception as e:
        logger.warning(f"Storage upload failed (non-fatal): {e}")

    # Steps 7 & 8: Extract triplets and store in Neo4j
    triplet_count = 0
    try:
        high_value_chunks = select_high_value_chunks(chunks)
        triplets = extract_triplets_from_chunks(high_value_chunks)
        if triplets:
            triplet_count = store_triplets(triplets, document_id)
    except Exception as e:
        logger.warning(f"Graph extraction failed (non-fatal): {e}")

    # Step 9: Auto-trigger community detection in background when new graph data was added
    if triplet_count > 0:
        from app.services.community import trigger_community_detection_background
        asyncio.create_task(trigger_community_detection_background())

    result = {
        "status": "success",
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "chunks_created": len(chunks),
        "triplets_extracted": triplet_count,
        "storage_path": storage_path,
    }

    logger.info(f"Ingestion complete: {result}")
    return result
