"""Journal ingestion pipeline: smart filter → extract concepts → store in user-scoped graph.

Also handles guided Q&A sessions: session_data is formatted as structured text,
then embedded into user_content_chunks for unified vector search.
"""

import asyncio
import json
import logging
from typing import Optional

import numpy as np
from google import genai
from google.genai import types
from psycopg2.extras import RealDictCursor, execute_values

from app.core.config import get_settings
from app.core.database import get_pg_connection
from app.services.graph_service import (
    store_journal_triplets,
    get_journal_graph,
    get_user_journal_concept_names,
)

logger = logging.getLogger(__name__)

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=get_settings().gemini_api_key)
    return _gemini_client


# ─── Smart Filter ────────────────────────────────────────────────

def smart_filter(content: str, user_id: str) -> tuple[bool, str]:
    """Three-stage filter to decide if a journal entry is worth graph-extracting.

    Returns (should_extract: bool, skip_reason: str).
    Stages run cheapest-first to avoid unnecessary LLM calls.

    Stage 1 (free): Minimum word count.
    Stage 2 (free): Novelty check — skip if content is overwhelmingly composed
                    of words already in the user's existing concept graph.
    Stage 3 (cheap LLM): Quality gate — a single yes/no call before full extraction.
    """
    words = content.split()

    # Stage 1: Length check
    if len(words) < 15:
        return False, "too_short"

    # Stage 2: Novelty check against existing JournalConcept names
    existing_concepts = get_user_journal_concept_names(user_id)
    if existing_concepts:
        content_words = {w.lower().strip(".,!?;:'\"") for w in words if len(w) > 2}
        if content_words:
            overlap = len(content_words & existing_concepts) / len(content_words)
            if overlap > 0.8:
                return False, "no_new_concepts"

    # Stage 3: LLM quality gate (~1 token output)
    try:
        client = _get_gemini_client()
        settings = get_settings()
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=(
                "Is this journal entry substantive enough to extract knowledge concepts from? "
                f"Reply only YES or NO.\n\n{content[:500]}"
            ),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
        )
        answer = response.text.strip().upper()
        if not answer.startswith("YES"):
            return False, "low_quality"
    except Exception as e:
        # If the gate fails, proceed — don't block extraction on a monitoring call
        logger.warning(f"LLM quality gate failed ({e}), proceeding with extraction")

    return True, ""


# ─── Triplet Extraction ──────────────────────────────────────────

def extract_journal_triplets(content: str) -> list[dict]:
    """Extract entity-relationship triplets from journal text via a single Gemini call.

    Uses journal-specific entity types (Concept, Idea, Goal, Emotion, Question, etc.)
    to better capture personal thought structures vs. factual document knowledge.

    Returns list of {head, tail, head_type, tail_type, relation}.
    """
    client = _get_gemini_client()
    settings = get_settings()

    system_prompt = """You are a personal knowledge graph expert. Extract entity-relationship triplets from this journal entry.

For each triplet, identify:
- head: The source concept, idea, or entity name
- tail: The target concept, idea, or entity name
- head_type: One of [Concept, Idea, Goal, Question, Emotion, Person, Topic, Technology, Project]
- tail_type: One of [Concept, Idea, Goal, Question, Emotion, Person, Topic, Technology, Project]
- relation: One of [LEADS_TO, RELATED_TO, BLOCKS, SUPPORTS, IS_A, DEPENDS_ON, CAUSES, CONTRASTS_WITH, INSPIRED_BY, PART_OF]

Return a JSON object {"triplets": [...]} covering the full entry.
Focus on meaningful personal connections: ideas linked to goals, questions connected to topics, emotions tied to events.
If no meaningful triplets can be extracted, return {"triplets": []}."""

    try:
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=f"Extract triplets from this journal entry:\n\n{content[:4000]}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=3000,
            ),
        )
        parsed = json.loads(response.text)
        triplets = parsed.get("triplets", []) if isinstance(parsed, dict) else parsed
        valid = [t for t in triplets if isinstance(t, dict) and "head" in t and "tail" in t]
        logger.info(f"Extracted {len(valid)} journal triplets")
        return valid
    except Exception as e:
        logger.warning(f"Journal triplet extraction failed: {e}")
        return []


# ─── Vector Embedding ────────────────────────────────────────────

def _embed_journal_content(entry_id: str, user_id: str, content: str) -> None:
    """Embed journal entry content into user_content_chunks for unified vector search."""
    try:
        from app.services.ingestion import _get_embedding_model
        model = _get_embedding_model()
        embedding = model.encode([content])[0]
        vec = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                # Remove previous embedding for this entry (in case of re-processing)
                cur.execute(
                    "DELETE FROM user_content_chunks WHERE source_id = %s AND user_id = %s AND source_type = 'journal'",
                    (entry_id, user_id),
                )
                execute_values(
                    cur,
                    """
                    INSERT INTO user_content_chunks (user_id, source_type, source_id, content, embedding, metadata)
                    VALUES %s
                    """,
                    [(user_id, 'journal', entry_id, content[:2000], vec, json.dumps({}))],
                    template="(%s, %s, %s, %s, %s::vector, %s::jsonb)",
                )
        logger.info(f"Embedded journal entry {entry_id} into user_content_chunks")
    except Exception as e:
        logger.warning(f"Failed to embed journal entry {entry_id}: {e}")


# ─── Background Processing ───────────────────────────────────────

async def _process_journal_entry(entry_id: str, user_id: str, content: str) -> None:
    """Background task: smart filter → extract → store graph concepts.

    Updates journal_entries.extraction_status as it progresses.
    Runs in a background asyncio task, not blocking the API response.
    """
    def _update_status(status: str, skip_reason: str = "", concept_count: int = 0):
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE journal_entries
                    SET extraction_status = %s, skip_reason = %s,
                        concept_count = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, skip_reason, concept_count, entry_id),
                )

    try:
        # Run smart filter in thread (Neo4j lookup + optional LLM call)
        should_extract, skip_reason = await asyncio.to_thread(smart_filter, content, user_id)

        if not should_extract:
            logger.info(f"Journal entry {entry_id} skipped: {skip_reason}")
            _update_status("skipped", skip_reason)
            return

        # Extract triplets
        triplets = await asyncio.to_thread(extract_journal_triplets, content)

        if not triplets:
            _update_status("skipped", "no_triplets_found")
            return

        # Store in Neo4j
        count = await asyncio.to_thread(store_journal_triplets, triplets, entry_id, user_id)
        _update_status("extracted", concept_count=count)
        logger.info(f"Journal entry {entry_id}: extracted {count} concepts into graph")

        # Embed into user_content_chunks for unified vector search
        await asyncio.to_thread(_embed_journal_content, entry_id, user_id, content)

    except Exception as e:
        logger.error(f"Journal processing failed for entry {entry_id}: {e}", exc_info=True)
        _update_status("failed", str(e)[:200])


# ─── Public API ──────────────────────────────────────────────────

def _format_session_content(session_data: list[dict]) -> str:
    """Format Q&A session data as structured text for triplet extraction."""
    lines = []
    for item in session_data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            lines.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(lines)


async def create_journal_entry(
    user_id: str,
    content: str,
    title: Optional[str] = None,
    session_data: Optional[list] = None,
) -> dict:
    """Create a journal entry and immediately return it.

    If session_data is provided (list of {question, answer} dicts from guided mode),
    the content is built from the Q&A pairs.

    Graph extraction and vector embedding run as background tasks.
    """
    # Build content from session data if provided
    if session_data:
        content = _format_session_content(session_data)
        if not title:
            # Use the first question as title (truncated)
            first_q = session_data[0].get("question", "") if session_data else ""
            title = first_q[:80] + "…" if len(first_q) > 80 else first_q

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO journal_entries (user_id, title, content, extraction_status)
                VALUES (%s, %s, %s, 'pending')
                RETURNING id, user_id, title, content, extraction_status, created_at
                """,
                (user_id, title, content),
            )
            entry = dict(cur.fetchone())

    # Fire-and-forget: extraction + embedding happens in background
    asyncio.create_task(_process_journal_entry(str(entry["id"]), user_id, content))

    return entry


def list_journal_entries(user_id: str, limit: int = 50) -> list[dict]:
    """List a user's journal entries, most recent first."""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, content, extraction_status, skip_reason,
                       concept_count, created_at, updated_at
                FROM journal_entries
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]


def delete_journal_entry(entry_id: str, user_id: str) -> bool:
    """Delete a journal entry and its associated graph concepts."""
    from app.services.graph_service import delete_journal_concepts

    # Remove graph concepts first (non-fatal if entry had no extraction)
    try:
        delete_journal_concepts(entry_id, user_id)
    except Exception as e:
        logger.warning(f"Graph cleanup failed for entry {entry_id}: {e}")

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM journal_entries WHERE id = %s AND user_id = %s RETURNING id",
                (entry_id, user_id),
            )
            return cur.fetchone() is not None


def get_user_journal_graph(user_id: str, limit: int = 50) -> dict:
    """Return the user's journal knowledge graph for visualization."""
    return get_journal_graph(user_id, limit=limit)
