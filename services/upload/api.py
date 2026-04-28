"""FastAPI application for the upload service."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from shared.config import get_settings
from shared.broker import InMemoryBroker, BaseBroker

from .repo import InMemoryImageRegistry, ImageRegistry
from .handlers import handle_upload, get_image_status

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
_registry: Optional[ImageRegistry] = None
_broker: Optional[BaseBroker] = None


# ============================================================================
# Request/Response Models
# ============================================================================


class UploadRequest(BaseModel):
    """Request body for image upload."""

    path: str = Field(..., description="Path to the image file")
    source: str = Field(..., description="Source identifier (e.g., camera_A)")


class UploadResponse(BaseModel):
    """Response body for image upload."""

    image_id: str = Field(..., description="Unique image identifier")
    is_duplicate: bool = Field(..., description="True if image was already uploaded")


class ImageStatusResponse(BaseModel):
    """Response body for image status."""

    image_id: str
    path: str
    source: str
    status: str
    submitted_at: str


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    service: str


# ============================================================================
# Application Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global _registry, _broker

    logger.info("Starting upload service...")

    # Initialize registry
    _registry = InMemoryImageRegistry()

    # Initialize broker
    # TODO: Use RedisBroker in production based on settings
    _broker = InMemoryBroker()
    await _broker.start()

    logger.info("Upload service started")

    yield

    # Shutdown
    logger.info("Shutting down upload service...")
    if _broker:
        await _broker.stop()
    logger.info("Upload service stopped")


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="Upload Service",
    description="Service for uploading images to the annotation pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="upload")


@app.post(
    "/images",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Image already exists (duplicate)"},
        201: {"description": "Image uploaded successfully"},
    },
)
async def upload_image(request: UploadRequest) -> UploadResponse:
    """
    Upload an image to the pipeline.

    If the same image (path + source + content) was already uploaded,
    returns the existing image_id with is_duplicate=True.
    """
    if _registry is None or _broker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )

    result = await handle_upload(
        path=request.path,
        source=request.source,
        registry=_registry,
        broker=_broker,
    )

    return UploadResponse(
        image_id=result.image_id,
        is_duplicate=result.is_duplicate,
    )


@app.get(
    "/images/{image_id}",
    response_model=ImageStatusResponse,
    responses={
        404: {"description": "Image not found"},
    },
)
async def get_image(image_id: str) -> ImageStatusResponse:
    """Get the status of an uploaded image."""
    if _registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )

    record = await get_image_status(image_id, _registry)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found: {image_id}",
        )

    return ImageStatusResponse(
        image_id=record.image_id,
        path=record.path,
        source=record.source,
        status=record.status,
        submitted_at=record.submitted_at,
    )


# ============================================================================
# Dependency injection helpers (for testing)
# ============================================================================


def set_registry(registry: ImageRegistry) -> None:
    """Set the registry instance (for testing)."""
    global _registry
    _registry = registry


def set_broker(broker: BaseBroker) -> None:
    """Set the broker instance (for testing)."""
    global _broker
    _broker = broker


def get_registry() -> Optional[ImageRegistry]:
    """Get the current registry instance."""
    return _registry


def get_broker() -> Optional[BaseBroker]:
    """Get the current broker instance."""
    return _broker
