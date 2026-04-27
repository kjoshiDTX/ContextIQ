"""ContextIQ backend configuration via environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # Supabase
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    supabase_db_url: str  # Direct Postgres connection for pgvector
    supabase_jwt_secret: str  # Project JWT secret for verifying auth tokens

    # Neo4j
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Gemini
    gemini_api_key: str

    # Model configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gemini-2.5-flash-preview-04-17"

    # App settings
    chunk_size_max: int = 1000
    query_cache_ttl_seconds: int = 3600
    vector_search_top_k: int = 10
    rerank_top_k: int = 3
    graph_search_limit: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
