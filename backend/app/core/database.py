"""Database connection management for Supabase (Postgres + pgvector) and Neo4j."""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector
from supabase import create_client, Client
from neo4j import GraphDatabase, Driver
from contextlib import contextmanager

from app.core.config import get_settings

# Global connection instances
_supabase_client: Client | None = None
_pg_pool: ThreadedConnectionPool | None = None
_neo4j_driver: Driver | None = None


def init_supabase() -> Client:
    """Initialize Supabase client for auth and storage."""
    global _supabase_client
    settings = get_settings()
    _supabase_client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _supabase_client


def get_supabase() -> Client:
    """Get the Supabase client instance."""
    if _supabase_client is None:
        return init_supabase()
    return _supabase_client


def init_pg_pool() -> ThreadedConnectionPool:
    """Initialize Postgres connection pool for pgvector operations."""
    global _pg_pool
    settings = get_settings()
    _pg_pool = ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        dsn=settings.supabase_db_url,
    )
    # Register pgvector type on a test connection
    conn = _pg_pool.getconn()
    try:
        register_vector(conn)
    finally:
        _pg_pool.putconn(conn)
    return _pg_pool


@contextmanager
def get_pg_connection():
    """Get a Postgres connection from the pool with pgvector support.
    
    Usage:
        with get_pg_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT ...")
    """
    global _pg_pool
    if _pg_pool is None:
        init_pg_pool()
    conn = _pg_pool.getconn()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pg_pool.putconn(conn)


def init_neo4j() -> Driver | None:
    """Initialize Neo4j driver. Returns None if the instance is unreachable."""
    global _neo4j_driver
    settings = get_settings()
    _neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        _neo4j_driver.verify_connectivity()
    except Exception as e:
        logging.warning("Neo4j unavailable at startup (graph features disabled): %s", e)
        _neo4j_driver.close()
        _neo4j_driver = None
    return _neo4j_driver


def get_neo4j() -> Driver:
    """Get the Neo4j driver instance."""
    if _neo4j_driver is None:
        return init_neo4j()
    return _neo4j_driver


async def startup_db():
    """Initialize all database connections on app startup."""
    init_supabase()
    init_pg_pool()
    init_neo4j()


async def shutdown_db():
    """Close all database connections on app shutdown."""
    global _pg_pool, _neo4j_driver
    if _pg_pool:
        _pg_pool.closeall()
        _pg_pool = None
    if _neo4j_driver:
        _neo4j_driver.close()
        _neo4j_driver = None
