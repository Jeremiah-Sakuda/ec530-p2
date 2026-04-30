"""FastAPI endpoints for the query service."""

from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.repos import VectorRepo, DocumentRepo, InMemoryDocumentRepo
from services.query.handlers import execute_query


# ============================================================================
# Request/Response Models
# ============================================================================


class QueryRequest(BaseModel):
    """Request model for query."""

    kind: Literal["text", "image"] = Field(..., description="Query type")
    value: str = Field(..., description="Query text or image path")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")


class QueryResultResponse(BaseModel):
    """Single query result."""

    image_id: str
    object_id: str
    score: float
    label: str


class QueryResponse(BaseModel):
    """Response model for query results."""

    results: List[QueryResultResponse]
    query_kind: str
    total_results: int


# ============================================================================
# Application State
# ============================================================================


class AppState:
    """Application state container for dependency injection."""

    vector_repo: Optional[VectorRepo] = None
    document_repo: Optional[DocumentRepo] = None


state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup: Initialize with defaults if not already set
    if state.vector_repo is None:
        state.vector_repo = VectorRepo(dim=128)
    if state.document_repo is None:
        state.document_repo = InMemoryDocumentRepo()

    yield

    # Shutdown: Nothing special needed


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="Query Service",
    description="Visual object similarity search",
    version="1.0.0",
    lifespan=lifespan,
)


def configure(vector_repo: VectorRepo, document_repo: DocumentRepo) -> None:
    """
    Configure the service with repositories.

    Call this before starting the application with custom dependencies.

    Args:
        vector_repo: Vector repository for search
        document_repo: Document repository for hydration
    """
    state.vector_repo = vector_repo
    state.document_repo = document_repo


# ============================================================================
# Endpoints
# ============================================================================


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Execute a similarity query.

    Searches for objects similar to the query text or image.

    Args:
        request: Query request with kind, value, and top_k

    Returns:
        Query results with hydrated labels
    """
    if state.vector_repo is None or state.document_repo is None:
        raise HTTPException(status_code=500, detail="Service not configured")

    results = await execute_query(
        kind=request.kind,
        value=request.value,
        top_k=request.top_k,
        vector_repo=state.vector_repo,
        document_repo=state.document_repo,
    )

    return QueryResponse(
        results=[
            QueryResultResponse(
                image_id=r["image_id"],
                object_id=r["object_id"],
                score=r["score"],
                label=r["label"],
            )
            for r in results
        ],
        query_kind=request.kind,
        total_results=len(results),
    )


@app.post("/query/text", response_model=QueryResponse)
async def query_text(
    text: str,
    top_k: int = 5,
) -> QueryResponse:
    """
    Execute a text-based similarity query.

    Convenience endpoint for text queries.

    Args:
        text: Query text
        top_k: Number of results

    Returns:
        Query results
    """
    if state.vector_repo is None or state.document_repo is None:
        raise HTTPException(status_code=500, detail="Service not configured")

    results = await execute_query(
        kind="text",
        value=text,
        top_k=top_k,
        vector_repo=state.vector_repo,
        document_repo=state.document_repo,
    )

    return QueryResponse(
        results=[
            QueryResultResponse(
                image_id=r["image_id"],
                object_id=r["object_id"],
                score=r["score"],
                label=r["label"],
            )
            for r in results
        ],
        query_kind="text",
        total_results=len(results),
    )


@app.post("/query/image", response_model=QueryResponse)
async def query_image(
    image_path: str,
    top_k: int = 5,
) -> QueryResponse:
    """
    Execute an image-based similarity query.

    Convenience endpoint for image queries.

    Args:
        image_path: Path to the query image
        top_k: Number of results

    Returns:
        Query results
    """
    if state.vector_repo is None or state.document_repo is None:
        raise HTTPException(status_code=500, detail="Service not configured")

    results = await execute_query(
        kind="image",
        value=image_path,
        top_k=top_k,
        vector_repo=state.vector_repo,
        document_repo=state.document_repo,
    )

    return QueryResponse(
        results=[
            QueryResultResponse(
                image_id=r["image_id"],
                object_id=r["object_id"],
                score=r["score"],
                label=r["label"],
            )
            for r in results
        ],
        query_kind="image",
        total_results=len(results),
    )


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "query"}
