# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContextIQ is a knowledge graph-powered RAG (Retrieval-Augmented Generation) backend. It ingests documents, chunks and embeds them into a pgvector store (Supabase/Postgres), extracts entity-relationship triplets into a Neo4j knowledge graph, and answers queries by combining vector search with graph traversal.

## Running the Backend

```bash
cd backend
# Install dependencies (use venv)
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server is at `http://localhost:8000`. API docs at `/docs`. Health check at `/health`.

## Environment Variables

Create `backend/.env` with:
```
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_SERVICE_KEY=
SUPABASE_DB_URL=          # Direct Postgres DSN for pgvector
NEO4J_URI=
NEO4J_USER=neo4j
NEO4J_PASSWORD=
OPENAI_API_KEY=
```

See `backend/app/core/config.py` for all settings and defaults.

## Database Setup

Run `backend/init_db.sql` in the Supabase SQL Editor to create all tables. Key tables:
- `global_documents` + `user_documents` — deduplication via SHA-256; multi-tenant via join table
- `document_chunks` — pgvector 384-dim embeddings (IVFFlat cosine index)
- `query_cache` — MD5-keyed response cache with TTL
- `tasks`, `document_categories` — user data in Postgres

Neo4j stores `Entity` nodes with `RELATES_TO` relationships (relation type stored as property, not edge label).

## Architecture

### Ingestion Pipeline (`app/services/ingestion.py`)
1. SHA-256 deduplication — if hash exists, just links to user
2. Text extraction (PDF/DOCX/TXT via `text_extraction.py`)
3. Semantic chunking via LangChain `SemanticChunker` + HuggingFace embeddings
4. Embed all chunks with `all-MiniLM-L6-v2` (384-dim)
5. Store chunks + vectors in pgvector
6. Upload raw file to Supabase Storage
7. Select high-value chunks (first 2, last 2, top 30% by word density)
8. Extract entity-relation triplets via OpenAI (`gpt-4o-mini`)
9. Store triplets in Neo4j

### Query Pipeline (`app/services/query.py`)
1. Check `query_cache` (MD5 hash, TTL-based)
2. Embed query → vector search (`match_chunks` SQL function) → cross-encoder rerank
3. Extract entities from query via OpenAI → Neo4j graph search (fuzzy CONTAINS match)
4. Optionally fetch open user tasks if query contains task-related keywords
5. Synthesize answer via OpenAI with combined vector + graph context
6. Cache result

### Knowledge Graph (`app/services/graph_service.py`)
All entities are stored as `:Entity` nodes. Relationships are stored as `RELATES_TO` edges with a `type` property (e.g., `"USES"`, `"IS_A"`). Community IDs are written back to nodes via Leiden algorithm (`app/services/community.py`) using Python-side igraph/cdlib since Neo4j AuraDB Free lacks GDS.

### API (`app/api/endpoints.py`)
Single router mounted at `/api`. Key endpoints:
- `POST /api/upload` — document ingestion
- `POST /api/query` — graph-aware RAG query
- `GET /api/graph` — visualization data (top nodes by degree)
- `POST /api/graph/detect-communities` — run Leiden detection
- CRUD for `/api/documents`, `/api/tasks`, `/api/categories`

### Database Connections (`app/core/database.py`)
Lazy-initialized singletons: Supabase client, `ThreadedConnectionPool` (psycopg2, 2–10 conns), Neo4j driver. All initialized at FastAPI startup via lifespan handler. Use `get_pg_connection()` context manager for Postgres — it handles pool checkout, pgvector registration, commit/rollback.

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy to keep main context window clean

- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -> then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
