"""Conversation insight processing: extract concepts from Claude chat snippets → graph + vector store."""

import asyncio
import json
import logging
import uuid

from google import genai
from google.genai import types
from psycopg2.extras import RealDictCursor, execute_values

from app.core.config import get_settings
from app.core.database import get_pg_connection
from app.services.graph_service import store_conversation_triplets

logger = logging.getLogger(__name__)

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=get_settings().gemini_api_key)
    return _gemini_client


# ─── Triplet Extraction ──────────────────────────────────────────

def extract_conversation_triplets(content: str) -> list[dict]:
    """Extract entity-relationship triplets from a conversation insight via Gemini.

    Uses conversation-specific entity types to capture insights, decisions,
    and observations that emerge from thinking out loud with Claude.
    """
    client = _get_gemini_client()
    settings = get_settings()

    system_prompt = """You are a personal knowledge graph expert processing insights from a Claude conversation.
Extract entity-relationship triplets that represent ideas, decisions, and insights discussed.

For each triplet:
- head: The source concept, idea, or entity
- tail: The target concept, idea, or entity
- head_type: One of [Insight, Decision, Observation, Idea, Goal, Question, Concern, Concept, Person, Project, Topic]
- tail_type: One of [Insight, Decision, Observation, Idea, Goal, Question, Concern, Concept, Person, Project, Topic]
- relation: One of [LEADS_TO, INSPIRED_BY, CONTRADICTS, SUPPORTS, RELATED_TO, RESULTED_IN, BLOCKS, DEPENDS_ON, PART_OF, IS_A]

Focus on capturing the user's thinking: what they're planning, what they've decided, what they're exploring.
Return a JSON object {"triplets": [...]}. If nothing meaningful, return {"triplets": []}."""

    try:
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=f"Extract knowledge triplets from this conversation insight:\n\n{content[:4000]}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=2000,
            ),
        )
        parsed = json.loads(response.text)
        triplets = parsed.get("triplets", []) if isinstance(parsed, dict) else parsed
        valid = [t for t in triplets if isinstance(t, dict) and "head" in t and "tail" in t]
        logger.info(f"Extracted {len(valid)} conversation triplets")
        return valid
    except Exception as e:
        logger.warning(f"Conversation triplet extraction failed: {e}")
        return []


# ─── Vector Embedding ────────────────────────────────────────────

def _embed_conversation_content(session_id: str, user_id: str, content: str) -> None:
    """Embed conversation insight into user_content_chunks for unified vector search."""
    try:
        from app.services.ingestion import _get_embedding_model
        model = _get_embedding_model()
        embedding = model.encode([content])[0]
        vec = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_content_chunks WHERE source_id = %s AND user_id = %s AND source_type = 'conversation'",
                    (session_id, user_id),
                )
                execute_values(
                    cur,
                    """
                    INSERT INTO user_content_chunks (user_id, source_type, source_id, content, embedding, metadata)
                    VALUES %s
                    """,
                    [(user_id, 'conversation', session_id, content[:2000], vec, json.dumps({}))],
                    template="(%s, %s, %s, %s, %s::vector, %s::jsonb)",
                )
        logger.info(f"Embedded conversation insight {session_id} into user_content_chunks")
    except Exception as e:
        logger.warning(f"Failed to embed conversation insight {session_id}: {e}")


# ─── Background Processing ───────────────────────────────────────

async def _process_insight(session_id: str, user_id: str, content: str) -> None:
    """Background task: extract triplets → Neo4j + embed → user_content_chunks."""
    try:
        # Extract triplets from the conversation content
        triplets = await asyncio.to_thread(extract_conversation_triplets, content)

        if triplets:
            await asyncio.to_thread(store_conversation_triplets, triplets, session_id, user_id)

        # Always embed even if no triplets — content is still searchable
        await asyncio.to_thread(_embed_conversation_content, session_id, user_id, content)

        # Update status in DB
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE conversation_insights
                    SET extraction_status = 'extracted', concept_count = %s, updated_at = NOW()
                    WHERE session_id = %s AND user_id = %s
                    """,
                    (len(triplets), session_id, user_id),
                )
    except Exception as e:
        logger.error(f"Insight processing failed for session {session_id}: {e}", exc_info=True)
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversation_insights SET extraction_status = 'failed', updated_at = NOW() WHERE session_id = %s",
                    (session_id,),
                )


# ─── Public API ──────────────────────────────────────────────────

async def save_conversation_insight(
    user_id: str,
    content: str,
    title: str | None = None,
) -> dict:
    """Save a conversation insight and trigger async processing.

    Called from both the MCP server (via /api/mcp/insight) and directly
    from the frontend save button.
    """
    session_id = str(uuid.uuid4())

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO conversation_insights
                    (session_id, user_id, title, content, extraction_status)
                VALUES (%s, %s, %s, %s, 'pending')
                RETURNING session_id, title, content, extraction_status, created_at
                """,
                (session_id, user_id, title, content),
            )
            insight = dict(cur.fetchone())

    # Fire-and-forget background processing
    asyncio.create_task(_process_insight(session_id, user_id, content))

    return insight
