"""ContextIQ API endpoints."""

import secrets
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Header
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.ingestion import ingest_document
from app.services.query import query_documents
from app.services.journal_service import (
    create_journal_entry,
    list_journal_entries,
    delete_journal_entry,
    get_user_journal_graph,
)
from app.services.graph_service import (
    get_graph_visualization,
    update_node_context,
)
from app.services.community import run_leiden_community_detection
from app.core.database import get_pg_connection
from app.core.auth import CurrentUser, OptionalUser
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── MCP Key Auth Helper ─────────────────────────────────────────

def _verify_mcp_key(x_mcp_key: str) -> str:
    """Verify the MCP API key and return the associated user_id.

    Raises HTTP 401 if key is missing or invalid.
    """
    if not x_mcp_key:
        raise HTTPException(status_code=401, detail="Missing X-MCP-Key header")
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id FROM mcp_api_keys WHERE api_key = %s",
                (x_mcp_key,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Invalid MCP API key")
            # Update last_used_at
            cur.execute(
                "UPDATE mcp_api_keys SET last_used_at = NOW() WHERE api_key = %s",
                (x_mcp_key,),
            )
    return str(row["user_id"])


# ─── Request / Response Models ────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    graph_context: list


class NodeContextRequest(BaseModel):
    node_name: str
    context: str


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: str = "todo"
    due_date: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[str] = None


class JournalCreate(BaseModel):
    content: str
    title: Optional[str] = None
    session_data: Optional[list] = None


class McpInsightRequest(BaseModel):
    content: str
    title: Optional[str] = None


class McpContextRequest(BaseModel):
    topic: str


class McpTaskRequest(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None


# ─── Document Endpoints ──────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    user_id: CurrentUser,
    file: UploadFile = File(...),
    category_id: Optional[str] = Form(None),
):
    """Upload and ingest a document (PDF, DOCX, or TXT)."""
    allowed_types = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "text/plain": "txt",
    }
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    ext_map = {"pdf": "pdf", "docx": "docx", "txt": "txt"}
    file_type = allowed_types.get(file.content_type) or ext_map.get(ext)
    if not file_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, TXT.",
        )

    content = await file.read()
    try:
        result = await ingest_document(
            file_content=content,
            filename=file.filename,
            file_type=file_type,
            user_id=user_id,
            category_id=category_id,
        )
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(user_id: CurrentUser):
    """List all documents for the authenticated user."""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT gd.id, gd.filename, gd.file_type, gd.chunk_count, gd.created_at,
                       ud.category_id, ud.added_at
                FROM user_documents ud
                JOIN global_documents gd ON ud.document_id = gd.id
                WHERE ud.user_id = %s
                ORDER BY ud.added_at DESC
                """,
                (user_id,),
            )
            docs = cur.fetchall()
    return {"documents": docs}


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, user_id: CurrentUser):
    """Remove a document from the user's library."""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "DELETE FROM user_documents WHERE document_id = %s AND user_id = %s RETURNING id",
                (document_id, user_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Document not found")

            cur.execute(
                "SELECT COUNT(*) as cnt FROM user_documents WHERE document_id = %s",
                (document_id,),
            )
            count = cur.fetchone()["cnt"]

            if count == 0:
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                cur.execute("DELETE FROM global_documents WHERE id = %s", (document_id,))

    return {"status": "deleted", "document_id": document_id}


# ─── Query Endpoint ──────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user_id: CurrentUser):
    """Query documents using combined vector search + knowledge graph."""
    try:
        result = await query_documents(
            query_text=request.query,
            user_id=user_id,
        )
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Graph Endpoints ─────────────────────────────────────────────

@router.get("/graph")
async def get_graph(limit: int = 50, user_id: OptionalUser = None):
    """Get knowledge graph visualization data — includes personal nodes when authenticated."""
    try:
        data = get_graph_visualization(limit=limit, user_id=user_id)
        return data
    except Exception as e:
        logger.error(f"Graph fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/node-context")
async def set_node_context(request: NodeContextRequest, user_id: CurrentUser):
    """Add user context/notes to a graph node."""
    try:
        update_node_context(
            node_name=request.node_name,
            context=request.context,
            user_id=user_id,
        )
        return {"status": "updated", "node": request.node_name}
    except Exception as e:
        logger.error(f"Node context update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/detect-communities")
async def detect_communities(resolution: float = 1.0):
    """Run Leiden community detection on the knowledge graph."""
    try:
        result = run_leiden_community_detection(resolution=resolution)
        return result
    except Exception as e:
        logger.error(f"Community detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Task Endpoints ──────────────────────────────────────────────

@router.post("/tasks")
async def create_task(task: TaskCreate, user_id: CurrentUser):
    """Create a new task for the authenticated user."""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO tasks (user_id, title, description, status, due_date)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (user_id, task.title, task.description, task.status, task.due_date),
            )
            new_task = cur.fetchone()
    return new_task


@router.get("/tasks")
async def list_tasks(user_id: CurrentUser):
    """List all tasks for the authenticated user."""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM tasks WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            tasks = cur.fetchall()
    return {"tasks": tasks}


@router.put("/tasks/{task_id}")
async def update_task(task_id: str, task: TaskUpdate):
    """Update a task."""
    updates = {k: v for k, v in task.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [task_id]

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"UPDATE tasks SET {set_clause}, updated_at = NOW() WHERE id = %s RETURNING *",
                values,
            )
            updated = cur.fetchone()
            if not updated:
                raise HTTPException(status_code=404, detail="Task not found")
    return updated


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("DELETE FROM tasks WHERE id = %s RETURNING id", (task_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "deleted", "task_id": task_id}


# ─── Journal Endpoints ───────────────────────────────────────────

@router.get("/journal/questions")
async def get_journal_questions(session_count: int = 0):
    """Return 3 guided journal questions for the current session."""
    from app.data.journal_questions import get_session_questions
    questions = get_session_questions(session_index=session_count, count=3)
    return {"questions": [{"id": q.id, "category": q.category, "text": q.text} for q in questions]}


@router.post("/journal")
async def create_journal(entry: JournalCreate, user_id: CurrentUser):
    """Create a journal entry and trigger async concept extraction.

    Accepts either free-form content or a guided session (session_data: [{question, answer}]).
    """
    try:
        result = await create_journal_entry(
            user_id=user_id,
            content=entry.content,
            title=entry.title,
            session_data=entry.session_data,
        )
        return result
    except Exception as e:
        logger.error(f"Journal creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/journal")
async def list_journal(user_id: CurrentUser, limit: int = 50):
    """List journal entries for the authenticated user, most recent first."""
    entries = list_journal_entries(user_id=user_id, limit=limit)
    return {"entries": entries}


@router.delete("/journal/{entry_id}")
async def delete_journal(entry_id: str, user_id: CurrentUser):
    """Delete a journal entry and its associated graph concepts."""
    deleted = delete_journal_entry(entry_id=entry_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    return {"status": "deleted", "entry_id": entry_id}


@router.get("/journal/graph")
async def get_journal_graph_view(user_id: CurrentUser, limit: int = 50):
    """Get the user's personal journal knowledge graph for visualization."""
    try:
        data = get_user_journal_graph(user_id=user_id, limit=limit)
        return data
    except Exception as e:
        logger.error(f"Journal graph fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── MCP Endpoints ───────────────────────────────────────────────

@router.get("/mcp/key")
async def get_mcp_key(user_id: CurrentUser):
    """Generate or return the user's MCP API key for Claude Desktop integration."""
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT api_key FROM mcp_api_keys WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                return {"api_key": row["api_key"]}
            # Generate new key
            new_key = "ciq_" + secrets.token_urlsafe(32)
            cur.execute(
                "INSERT INTO mcp_api_keys (user_id, api_key) VALUES (%s, %s) RETURNING api_key",
                (user_id, new_key),
            )
            row = cur.fetchone()
    return {"api_key": row["api_key"]}


@router.post("/mcp/insight")
async def mcp_save_insight(
    request: McpInsightRequest,
    x_mcp_key: Optional[str] = Header(None),
):
    """Save a conversation insight from Claude Desktop (MCP auth via X-MCP-Key header)."""
    user_id = _verify_mcp_key(x_mcp_key)
    try:
        from app.services.conversation_service import save_conversation_insight
        result = await save_conversation_insight(
            user_id=user_id,
            content=request.content,
            title=request.title,
        )
        return {"status": "saved", "session_id": result["session_id"]}
    except Exception as e:
        logger.error(f"MCP insight save failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/context")
async def mcp_get_context(
    request: McpContextRequest,
    x_mcp_key: Optional[str] = Header(None),
):
    """Retrieve user context for a topic — called by Claude to personalize conversations."""
    user_id = _verify_mcp_key(x_mcp_key)
    try:
        result = await query_documents(query_text=request.topic, user_id=user_id)
        return {
            "context": result["answer"],
            "sources_count": len(result.get("sources", [])),
        }
    except Exception as e:
        logger.error(f"MCP context fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/task")
async def mcp_create_task(
    request: McpTaskRequest,
    x_mcp_key: Optional[str] = Header(None),
):
    """Create a task from Claude Desktop (MCP auth)."""
    user_id = _verify_mcp_key(x_mcp_key)
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO tasks (user_id, title, description, status, due_date)
                VALUES (%s, %s, %s, 'todo', %s)
                RETURNING *
                """,
                (user_id, request.title, request.description, request.due_date),
            )
            task = cur.fetchone()
    return {"status": "created", "task": dict(task)}


@router.post("/mcp/journal")
async def mcp_add_journal(
    request: JournalCreate,
    x_mcp_key: Optional[str] = Header(None),
):
    """Add a journal entry from Claude Desktop (MCP auth)."""
    user_id = _verify_mcp_key(x_mcp_key)
    try:
        result = await create_journal_entry(
            user_id=user_id,
            content=request.content,
            title=request.title,
            session_data=request.session_data,
        )
        return {"status": "saved", "entry_id": str(result["id"])}
    except Exception as e:
        logger.error(f"MCP journal save failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
