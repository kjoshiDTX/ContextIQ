"""ContextIQ - Knowledge Graph-Powered RAG Application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import startup_db, shutdown_db
from app.api.endpoints import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle — init and teardown DB connections."""
    await startup_db()
    yield
    await shutdown_db()


app = FastAPI(
    title="ContextIQ",
    description="Knowledge Graph-Powered RAG for Document Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "contextiq"}
