"""Pydantic schemas for event payloads and validation."""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Payload Schemas
# ============================================================================


class ImageSubmittedPayload(BaseModel):
    """Payload for image.submitted event."""

    image_id: str = Field(..., description="Unique image identifier")
    path: str = Field(..., description="Path to the image file")
    source: str = Field(..., description="Source identifier (e.g., camera_A)")


class DetectedObject(BaseModel):
    """A single detected object from inference."""

    object_id: str = Field(..., description="Unique object identifier within image")
    label: str = Field(..., description="Detected object class label")
    bbox: list[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 elements")
        return v


class InferenceCompletedPayload(BaseModel):
    """Payload for inference.completed event."""

    image_id: str = Field(..., description="Image that was processed")
    model_version: str = Field(..., description="Version of the inference model")
    objects: list[DetectedObject] = Field(..., description="List of detected objects")


class AnnotationStoredPayload(BaseModel):
    """Payload for annotation.stored event."""

    image_id: str = Field(..., description="Image ID")
    object_ids: list[str] = Field(..., description="IDs of objects in annotation")
    model_version: str = Field(..., description="Model version used for detection")


class AnnotationCorrectedPayload(BaseModel):
    """Payload for annotation.corrected event."""

    image_id: str = Field(..., description="Image being corrected")
    patch: dict[str, Any] = Field(..., description="Patch to apply (e.g., {'objects.0.label': 'truck'})")
    reviewer: str = Field(..., description="Who made the correction")


class EmbeddingInfo(BaseModel):
    """Information about a single embedding."""

    object_id: str = Field(..., description="Object this embedding represents")
    vector_ref: str = Field(..., description="Reference to vector storage location")
    dim: int = Field(..., description="Dimensionality of the vector")


class EmbeddingCreatedPayload(BaseModel):
    """Payload for embedding.created event."""

    image_id: str = Field(..., description="Image ID")
    embeddings: list[EmbeddingInfo] = Field(..., description="List of created embeddings")


class QuerySubmittedPayload(BaseModel):
    """Payload for query.submitted event."""

    query_id: str = Field(..., description="Unique query identifier")
    kind: str = Field(..., description="Query type: 'text' or 'image'")
    value: str = Field(..., description="Query text or image path")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        if v not in ("text", "image"):
            raise ValueError("kind must be 'text' or 'image'")
        return v


class QueryResult(BaseModel):
    """A single query result."""

    image_id: str = Field(..., description="Image containing the match")
    object_id: str = Field(..., description="Matched object ID")
    score: float = Field(..., description="Similarity score")


class QueryCompletedPayload(BaseModel):
    """Payload for query.completed event."""

    query_id: str = Field(..., description="Query this response is for")
    results: list[QueryResult] = Field(..., description="Ranked results")


# ============================================================================
# Envelope Schema
# ============================================================================


class EnvelopeSchema(BaseModel):
    """Full event envelope schema for validation."""

    type: str = Field(default="publish", description="Message type")
    topic: str = Field(..., description="Topic name")
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    schema_version: int = Field(default=1, description="Schema version")
    payload: dict[str, Any] = Field(..., description="Event payload")


# ============================================================================
# Validation Functions
# ============================================================================

# Mapping of topics to their payload schemas
PAYLOAD_SCHEMAS: dict[str, type[BaseModel]] = {
    "image.submitted": ImageSubmittedPayload,
    "inference.completed": InferenceCompletedPayload,
    "annotation.stored": AnnotationStoredPayload,
    "annotation.corrected": AnnotationCorrectedPayload,
    "embedding.created": EmbeddingCreatedPayload,
    "query.submitted": QuerySubmittedPayload,
    "query.completed": QueryCompletedPayload,
}


class ValidationError(Exception):
    """Raised when event validation fails."""

    def __init__(self, message: str, errors: Optional[list] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []


def validate_envelope(data: dict[str, Any]) -> EnvelopeSchema:
    """
    Validate the envelope structure.

    Args:
        data: Raw event data dictionary

    Returns:
        Validated EnvelopeSchema

    Raises:
        ValidationError: If envelope is malformed
    """
    try:
        return EnvelopeSchema.model_validate(data)
    except Exception as e:
        raise ValidationError(f"Invalid envelope: {e}", errors=[str(e)])


def validate_payload(topic: str, payload: dict[str, Any]) -> BaseModel:
    """
    Validate payload against the schema for the given topic.

    Args:
        topic: Event topic name
        payload: Event payload dictionary

    Returns:
        Validated payload model instance

    Raises:
        ValidationError: If payload is invalid or topic unknown
    """
    schema = PAYLOAD_SCHEMAS.get(topic)
    if schema is None:
        raise ValidationError(f"Unknown topic: {topic}")

    try:
        return schema.model_validate(payload)
    except Exception as e:
        raise ValidationError(f"Invalid payload for {topic}: {e}", errors=[str(e)])


def validate_event(data: dict[str, Any]) -> tuple[EnvelopeSchema, BaseModel]:
    """
    Validate both envelope and payload.

    Args:
        data: Raw event data dictionary

    Returns:
        Tuple of (validated envelope, validated payload)

    Raises:
        ValidationError: If validation fails
    """
    envelope = validate_envelope(data)
    payload = validate_payload(envelope.topic, envelope.payload)
    return envelope, payload


def is_valid_event(data: dict[str, Any]) -> bool:
    """
    Check if event is valid without raising exceptions.

    Args:
        data: Raw event data dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        validate_event(data)
        return True
    except ValidationError:
        return False
