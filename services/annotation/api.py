"""FastAPI endpoints for the annotation service."""

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from shared.events import create_envelope
from shared.events.topics import Topics
from shared.broker import BaseBroker, InMemoryBroker
from shared.repos import DocumentRepo, InMemoryDocumentRepo
from services.annotation.handlers import apply_patch


# ============================================================================
# Request/Response Models
# ============================================================================


class DetectedObjectResponse(BaseModel):
    """A detected object in an annotation."""

    object_id: str
    label: str
    bbox: list[int]
    conf: float


class AnnotationResponse(BaseModel):
    """Response model for annotation data."""

    image_id: str
    objects: list[DetectedObjectResponse]
    model_version: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class CorrectionPatch(BaseModel):
    """Request model for annotation corrections."""

    patch: dict[str, Any] = Field(
        ...,
        description="Patch to apply using dot notation (e.g., {'objects.0.label': 'truck'})",
    )
    reviewer: str = Field(..., description="Who is making the correction")


class CorrectionResponse(BaseModel):
    """Response model for correction results."""

    image_id: str
    status: str
    message: str


# ============================================================================
# Application State
# ============================================================================


class AppState:
    """Application state container for dependency injection."""

    repo: DocumentRepo
    broker: BaseBroker


state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup: Initialize with defaults if not already set
    if not hasattr(state, "repo") or state.repo is None:
        state.repo = InMemoryDocumentRepo()
    if not hasattr(state, "broker") or state.broker is None:
        state.broker = InMemoryBroker()
        await state.broker.start()

    yield

    # Shutdown: Stop broker
    if state.broker and state.broker.is_running:
        await state.broker.stop()


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="Annotation Service",
    description="Manages image annotations and corrections",
    version="1.0.0",
    lifespan=lifespan,
)


def configure(repo: DocumentRepo, broker: BaseBroker) -> None:
    """
    Configure the service with repository and broker.

    Call this before starting the application with custom dependencies.

    Args:
        repo: Document repository to use
        broker: Message broker to use
    """
    state.repo = repo
    state.broker = broker


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/annotations/{image_id}", response_model=AnnotationResponse)
async def get_annotation(image_id: str) -> AnnotationResponse:
    """
    Get annotation for a specific image.

    Args:
        image_id: The image identifier

    Returns:
        Annotation data for the image

    Raises:
        HTTPException: 404 if image not found
    """
    doc = await state.repo.get(image_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Annotation not found for image {image_id}")

    return AnnotationResponse(
        image_id=doc["image_id"],
        objects=[DetectedObjectResponse(**obj) for obj in doc.get("objects", [])],
        model_version=doc.get("model_version", "unknown"),
        status=doc.get("status", "pending"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        history=doc.get("history", []),
    )


@app.patch("/annotations/{image_id}", response_model=CorrectionResponse)
async def correct_annotation(image_id: str, correction: CorrectionPatch) -> CorrectionResponse:
    """
    Apply a correction to an annotation.

    This publishes an annotation.corrected event which will be
    processed to update the annotation and trigger re-embedding.

    Args:
        image_id: The image identifier
        correction: The correction patch and reviewer info

    Returns:
        Confirmation of correction submission

    Raises:
        HTTPException: 404 if image not found
    """
    # Verify annotation exists
    doc = await state.repo.get(image_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Annotation not found for image {image_id}")

    # Publish annotation.corrected event
    payload = {
        "image_id": image_id,
        "patch": correction.patch,
        "reviewer": correction.reviewer,
    }
    envelope = create_envelope(Topics.ANNOTATION_CORRECTED, payload)
    await state.broker.publish(Topics.ANNOTATION_CORRECTED, envelope)

    return CorrectionResponse(
        image_id=image_id,
        status="submitted",
        message=f"Correction submitted by {correction.reviewer}",
    )


@app.get("/annotations", response_model=list[AnnotationResponse])
async def query_annotations(
    label: Optional[str] = Query(None, description="Filter by object label"),
    min_conf: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
    status: Optional[str] = Query(None, description="Filter by status"),
) -> list[AnnotationResponse]:
    """
    Query annotations with optional filters.

    This is a metadata-only query, not vector search.

    Args:
        label: Filter by object label
        min_conf: Minimum confidence threshold
        status: Filter by annotation status

    Returns:
        List of matching annotations
    """
    filters = {}
    if label:
        filters["label"] = label
    if min_conf is not None:
        filters["min_conf"] = min_conf
    if status:
        filters["status"] = status

    docs = await state.repo.query(filters)

    return [
        AnnotationResponse(
            image_id=doc["image_id"],
            objects=[DetectedObjectResponse(**obj) for obj in doc.get("objects", [])],
            model_version=doc.get("model_version", "unknown"),
            status=doc.get("status", "pending"),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
            history=doc.get("history", []),
        )
        for doc in docs
    ]


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "annotation"}
