"""FastAPI endpoints for the vector index service."""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.repos import VectorRepo


# ============================================================================
# Request/Response Models
# ============================================================================


class SearchRequest(BaseModel):
    """Request model for vector search."""

    vector: List[float] = Field(..., description="Query vector (128-dim)")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")


class SearchByIdsRequest(BaseModel):
    """Request model for search by existing vector IDs."""

    image_id: str = Field(..., description="Image ID of the query vector")
    object_id: str = Field(..., description="Object ID of the query vector")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")


class SearchResultResponse(BaseModel):
    """Single search result."""

    image_id: str
    object_id: str
    distance: float
    score: float


class SearchResponse(BaseModel):
    """Response model for search results."""

    results: List[SearchResultResponse]
    query_dim: int
    index_size: int


class IndexStatsResponse(BaseModel):
    """Response model for index statistics."""

    ntotal: int
    dim: int


# ============================================================================
# Application State
# ============================================================================


class AppState:
    """Application state container for dependency injection."""

    vector_repo: Optional[VectorRepo] = None


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

    yield

    # Shutdown: Nothing special needed


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="Vector Index Service",
    description="FAISS-based vector similarity search",
    version="1.0.0",
    lifespan=lifespan,
)


def configure(vector_repo: VectorRepo) -> None:
    """
    Configure the service with a vector repository.

    Call this before starting the application with custom dependencies.

    Args:
        vector_repo: Vector repository to use
    """
    state.vector_repo = vector_repo


# ============================================================================
# Endpoints
# ============================================================================


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search for similar vectors.

    Args:
        request: Search request with query vector and top_k

    Returns:
        Search results sorted by similarity
    """
    if state.vector_repo is None:
        raise HTTPException(status_code=500, detail="Vector repo not configured")

    if len(request.vector) != state.vector_repo.dim:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension mismatch: expected {state.vector_repo.dim}, got {len(request.vector)}",
        )

    results = state.vector_repo.search(request.vector, request.top_k)

    return SearchResponse(
        results=[
            SearchResultResponse(
                image_id=r.image_id,
                object_id=r.object_id,
                distance=r.distance,
                score=r.score,
            )
            for r in results
        ],
        query_dim=len(request.vector),
        index_size=state.vector_repo.ntotal,
    )


@app.post("/search/by-ids", response_model=SearchResponse)
async def search_by_ids(request: SearchByIdsRequest) -> SearchResponse:
    """
    Search for similar vectors using an existing vector's IDs.

    The query vector itself is excluded from results.

    Args:
        request: Search request with image_id, object_id, and top_k

    Returns:
        Search results sorted by similarity
    """
    if state.vector_repo is None:
        raise HTTPException(status_code=500, detail="Vector repo not configured")

    if not state.vector_repo.has(request.image_id, request.object_id):
        raise HTTPException(
            status_code=404,
            detail=f"Vector not found for {request.image_id}/{request.object_id}",
        )

    results = state.vector_repo.search_by_ids(
        request.image_id, request.object_id, request.top_k
    )

    return SearchResponse(
        results=[
            SearchResultResponse(
                image_id=r.image_id,
                object_id=r.object_id,
                distance=r.distance,
                score=r.score,
            )
            for r in results
        ],
        query_dim=state.vector_repo.dim,
        index_size=state.vector_repo.ntotal,
    )


@app.get("/stats", response_model=IndexStatsResponse)
async def get_stats() -> IndexStatsResponse:
    """
    Get vector index statistics.

    Returns:
        Index statistics including total vectors and dimension
    """
    if state.vector_repo is None:
        raise HTTPException(status_code=500, detail="Vector repo not configured")

    return IndexStatsResponse(
        ntotal=state.vector_repo.ntotal,
        dim=state.vector_repo.dim,
    )


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "vector_index"}
