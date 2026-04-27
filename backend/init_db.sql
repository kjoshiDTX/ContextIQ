-- ContextIQ: Supabase Database Schema
-- Run this in Supabase SQL Editor (Dashboard > SQL Editor > New Query)

-- =============================================================
-- 1. Enable pgvector extension
-- =============================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================
-- 2. Global documents (deduplicated by file hash)
-- =============================================================
CREATE TABLE IF NOT EXISTS global_documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    file_hash TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL DEFAULT 'pdf',
    storage_path TEXT,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================
-- 3. User-document junction (multi-tenant)
-- =============================================================
CREATE TABLE IF NOT EXISTS user_documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES global_documents(id) ON DELETE CASCADE,
    category_id UUID,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_user_documents_user_id ON user_documents(user_id);

-- =============================================================
-- 4. Document chunks with vector embeddings (384-dim)
-- =============================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES global_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for cosine similarity — rebuild after bulk inserts
-- lists = 100 is good for up to ~100k vectors; adjust as data grows
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);

-- =============================================================
-- 5. Document categories
-- =============================================================
CREATE TABLE IF NOT EXISTS document_categories (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    color TEXT DEFAULT '#6366f1',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add FK from user_documents to categories
ALTER TABLE user_documents
    ADD CONSTRAINT fk_user_documents_category
    FOREIGN KEY (category_id) REFERENCES document_categories(id) ON DELETE SET NULL;

-- =============================================================
-- 6. Tasks (moved from Neo4j to Supabase)
-- =============================================================
CREATE TABLE IF NOT EXISTS tasks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'todo' CHECK (status IN ('todo', 'in_progress', 'done')),
    due_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);

-- =============================================================
-- 7. Query cache
-- =============================================================
CREATE TABLE IF NOT EXISTS query_cache (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    query_hash TEXT UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    response TEXT NOT NULL,
    sources JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_cache_hash ON query_cache(query_hash);

-- =============================================================
-- 8. Journal entries (second brain / personal thought graph)
-- =============================================================
CREATE TABLE IF NOT EXISTS journal_entries (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT,
    content TEXT NOT NULL,
    extraction_status TEXT DEFAULT 'pending'
        CHECK (extraction_status IN ('pending', 'skipped', 'extracted', 'failed')),
    skip_reason TEXT,       -- populated when extraction_status = 'skipped' or 'failed'
    concept_count INTEGER DEFAULT 0,  -- number of JournalConcept nodes created in Neo4j
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_journal_entries_user_id ON journal_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_journal_entries_status ON journal_entries(extraction_status);

-- =============================================================
-- 9. User content chunks (journal + conversation embeddings)
--    Separate from document_chunks because these are user-scoped
--    and sourced from personal content, not uploaded documents.
-- =============================================================
CREATE TABLE IF NOT EXISTS user_content_chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK (source_type IN ('journal', 'conversation')),
    source_id TEXT NOT NULL,        -- journal_entry id or conversation session id
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_content_chunks_user_id ON user_content_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_user_content_chunks_source ON user_content_chunks(user_id, source_type);
CREATE INDEX IF NOT EXISTS idx_user_content_embedding
    ON user_content_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- =============================================================
-- 10. Conversation insights (from Claude Desktop MCP integration)
-- =============================================================
CREATE TABLE IF NOT EXISTS conversation_insights (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,    -- UUID generated per save_insight call
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT,
    content TEXT NOT NULL,
    extraction_status TEXT DEFAULT 'pending'
        CHECK (extraction_status IN ('pending', 'extracted', 'failed')),
    concept_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_insights_user ON conversation_insights(user_id);

-- =============================================================
-- 11. MCP API keys (one per user, for Claude Desktop integration)
-- =============================================================
CREATE TABLE IF NOT EXISTS mcp_api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
    api_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_mcp_api_keys_key ON mcp_api_keys(api_key);

-- =============================================================
-- 11. Helper function: cosine similarity search (documents)
-- =============================================================
CREATE OR REPLACE FUNCTION match_user_content(
    query_embedding vector(384),
    p_user_id UUID,
    match_count INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    source_type TEXT,
    source_id TEXT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        uc.id,
        uc.source_type,
        uc.source_id,
        uc.content,
        uc.metadata,
        1 - (uc.embedding <=> query_embedding) AS similarity
    FROM user_content_chunks uc
    WHERE uc.user_id = p_user_id
    ORDER BY uc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(384),
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
