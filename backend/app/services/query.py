"""Graph-aware RAG query pipeline: vector search + knowledge graph + personal context + LLM synthesis.

Searches three content layers:
  1. Document chunks (uploaded files)
  2. Personal content (journal entries + Claude conversation insights)
  3. Knowledge graph entities (Neo4j)
"""

import asyncio
import hashlib
import json
import logging
from typing import Optional

from google import genai
from google.genai import types
from psycopg2.extras import RealDictCursor

from app.core.config import get_settings
from app.core.database import get_pg_connection
from app.services.vector_service import search_similar
from app.services.graph_service import get_graph_context

logger = logging.getLogger(__name__)

# Lazy-loaded models
_embedding_model = None
_cross_encoder = None
_gemini_client: genai.Client | None = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        settings = get_settings()
        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    return _cross_encoder


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=get_settings().gemini_api_key)
    return _gemini_client


# ─── Query Cache ──────────────────────────────────────────────────

def _cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def _check_cache(query_hash: str) -> Optional[dict]:
    settings = get_settings()
    ttl = settings.query_cache_ttl_seconds

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT response, sources, created_at
                FROM query_cache
                WHERE query_hash = %s
                  AND created_at > NOW() - INTERVAL '%s seconds'
                """,
                (query_hash, ttl),
            )
            cached = cur.fetchone()
            if cached:
                return {
                    "answer": cached["response"],
                    "sources": json.loads(cached["sources"]) if isinstance(cached["sources"], str) else cached["sources"],
                    "graph_context": [],
                    "cached": True,
                }
    return None


def _store_cache(query_hash: str, query_text: str, response: str, sources: list):
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_cache (query_hash, query_text, response, sources)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (query_hash) DO UPDATE
                SET response = EXCLUDED.response,
                    sources = EXCLUDED.sources,
                    created_at = NOW()
                """,
                (query_hash, query_text, response, json.dumps(sources)),
            )


# ─── Entity Extraction ───────────────────────────────────────────

def extract_query_entities(query: str) -> list[str]:
    """Use Gemini to extract entity names from a query for graph search."""
    settings = get_settings()
    client = _get_gemini_client()

    try:
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Extract the key entity names from the user query. "
                    "Return a JSON object with a single key 'entities' containing an array of strings. "
                    "Focus on specific nouns: people, organizations, technologies, products, concepts. "
                    'If no specific entities are found, return {"entities": []}.'
                ),
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=200,
            ),
        )
        parsed = json.loads(response.text)
        entities = parsed.get("entities", [])
        logger.info(f"Extracted entities from query: {entities}")
        return entities
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return []


# ─── Cross-Encoder Reranking ─────────────────────────────────────

def rerank_results(query: str, results: list[dict], top_k: int = 3) -> list[dict]:
    """Rerank vector search results using a cross-encoder model."""
    if not results:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, r["content"]) for r in results]
    scores = cross_encoder.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ─── Personal Content Search ─────────────────────────────────────

def search_personal_content(query_embedding, user_id: str, top_k: int = 5) -> list[dict]:
    """Search user_content_chunks (journal + conversations) using vector similarity."""
    try:
        vec_str = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
        with get_pg_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, source_type, source_id, content, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM user_content_chunks
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec_str, user_id, vec_str, top_k),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.warning(f"Personal content search failed: {e}")
        return []


# ─── LLM Answer Synthesis ────────────────────────────────────────

def synthesize_answer(
    query: str,
    vector_context: list[dict],
    graph_context: list[dict],
    personal_context: list[dict] | None = None,
    task_context: list[dict] | None = None,
) -> str:
    """Combine all context sources and generate a personal, synthesized answer."""
    settings = get_settings()
    client = _get_gemini_client()

    context_parts = []

    if vector_context:
        doc_texts = [f"[Document Chunk {i}]\n{ctx['content']}" for i, ctx in enumerate(vector_context, 1)]
        context_parts.append("## Documents\n" + "\n\n".join(doc_texts))

    if personal_context:
        journal_chunks = [c for c in personal_context if c.get("source_type") == "journal"]
        conv_chunks = [c for c in personal_context if c.get("source_type") == "conversation"]
        if journal_chunks:
            journal_texts = [f"[Journal Entry {i}]\n{c['content']}" for i, c in enumerate(journal_chunks, 1)]
            context_parts.append("## Journal Reflections\n" + "\n\n".join(journal_texts))
        if conv_chunks:
            conv_texts = [f"[Conversation Insight {i}]\n{c['content']}" for i, c in enumerate(conv_chunks, 1)]
            context_parts.append("## Conversation Insights (from Claude chats)\n" + "\n\n".join(conv_texts))

    if graph_context:
        graph_facts = []
        for fact in graph_context:
            fact_str = f"- {fact['head']} ({fact.get('head_type', '?')}) —[{fact['relation']}]→ {fact['tail']} ({fact.get('tail_type', '?')})"
            if fact.get("user_context"):
                fact_str += f"\n  User note: {fact['user_context']}"
            graph_facts.append(fact_str)
        context_parts.append("## Knowledge Graph Connections\n" + "\n".join(graph_facts))

    if task_context:
        task_texts = [f"- [{t['status']}] {t['title']}: {t.get('description', '')}" for t in task_context]
        context_parts.append("## Active Tasks\n" + "\n".join(task_texts))

    full_context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    system_prompt = """You are the user's personal AI — a second brain that has access to everything they've shared with ContextIQ:
- Documents they've uploaded and read
- Journal reflections and guided answers to self-reflection questions
- Insights saved from their Claude conversations
- Their active tasks and goals
- Connections in their personal knowledge graph

Answer as if you truly know this person. Synthesize across all sources naturally — don't just list facts, but connect them into a coherent, personally relevant response.

Rules:
- Only use information from the provided context.
- If the context doesn't contain enough to answer, say so honestly and suggest what information would help.
- When the question is personal (about goals, emotions, decisions, growth), draw from journal and conversation context especially.
- Be concise but genuinely useful. Avoid generic advice — keep it grounded in what the user has actually shared."""

    response = client.models.generate_content(
        model=settings.llm_model,
        contents=f"Context:\n{full_context}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            max_output_tokens=1500,
        ),
    )

    return response.text


# ─── Main Query Pipeline ─────────────────────────────────────────

async def query_documents(
    query_text: str,
    user_id: Optional[str] = None,
) -> dict:
    """Execute the full Graph-aware RAG query pipeline.

    Steps:
    1. Check cache
    2. Embed query, then run vector search + entity extraction in parallel
    3. Rerank vector results, fetch graph context
    4. Fetch user tasks if query seems task-related
    5. LLM synthesis with combined context
    6. Cache result
    """
    settings = get_settings()

    # Step 1: Cache check
    query_hash = _cache_key(query_text)
    cached = _check_cache(query_hash)
    if cached:
        logger.info(f"Cache hit for query: {query_text[:50]}...")
        return cached

    # Compute embedding first (required by vector search)
    query_embedding = _get_embedding_model().encode(query_text, convert_to_numpy=True)

    # Step 2: Vector search + entity extraction + personal search run in parallel
    if user_id:
        vector_results, entities, personal_results = await asyncio.gather(
            asyncio.to_thread(search_similar, query_embedding, settings.vector_search_top_k),
            asyncio.to_thread(extract_query_entities, query_text),
            asyncio.to_thread(search_personal_content, query_embedding, user_id, 5),
        )
    else:
        vector_results, entities = await asyncio.gather(
            asyncio.to_thread(search_similar, query_embedding, settings.vector_search_top_k),
            asyncio.to_thread(extract_query_entities, query_text),
        )
        personal_results = []

    # Step 3: Rerank document results + graph lookup
    reranked = rerank_results(query_text, vector_results, top_k=settings.rerank_top_k)
    graph_facts = get_graph_context(entities, limit=settings.graph_search_limit) if entities else []
    logger.info(f"Query pipeline: {len(reranked)} doc chunks, {len(graph_facts)} graph triplets for entities={entities}")

    # Step 4: Task context (always include when user is authenticated)
    task_context = None
    if user_id:
        task_keywords = ["task", "todo", "assignment", "deadline", "due", "homework", "project"]
        if any(kw in query_text.lower() for kw in task_keywords):
            with get_pg_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT title, description, status, due_date FROM tasks WHERE user_id = %s AND status != 'done'",
                        (user_id,),
                    )
                    task_context = [dict(r) for r in cur.fetchall()]

    # Step 5: LLM synthesis with all context sources
    answer = synthesize_answer(
        query=query_text,
        vector_context=reranked,
        graph_context=graph_facts,
        personal_context=personal_results or None,
        task_context=task_context,
    )

    # Build sources list with source_type attribution
    sources = [
        {
            "source_type": "document",
            "document_id": str(r.get("document_id", "")),
            "chunk_index": r.get("chunk_index"),
            "similarity": r.get("similarity"),
            "rerank_score": r.get("rerank_score"),
            "content_preview": r["content"][:150],
        }
        for r in reranked
    ]
    for r in (personal_results or []):
        sources.append({
            "source_type": r.get("source_type", "personal"),
            "source_id": r.get("source_id", ""),
            "similarity": r.get("similarity"),
            "content_preview": r["content"][:150],
        })

    # Step 6: Cache result
    _store_cache(query_hash, query_text, answer, sources)

    return {
        "answer": answer,
        "sources": sources,
        "graph_context": graph_facts,
    }
