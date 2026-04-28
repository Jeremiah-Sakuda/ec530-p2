"""Event system components."""

from .envelope import (
    EventEnvelope,
    create_envelope,
    generate_event_id,
    generate_timestamp,
)
from .topics import Topics
from .schema import (
    ImageSubmittedPayload,
    InferenceCompletedPayload,
    AnnotationStoredPayload,
    AnnotationCorrectedPayload,
    EmbeddingCreatedPayload,
    QuerySubmittedPayload,
    QueryCompletedPayload,
    DetectedObject,
    EmbeddingInfo,
    QueryResult,
    EnvelopeSchema,
    ValidationError,
    validate_envelope,
    validate_payload,
    validate_event,
    is_valid_event,
    PAYLOAD_SCHEMAS,
)

__all__ = [
    # Envelope
    "EventEnvelope",
    "create_envelope",
    "generate_event_id",
    "generate_timestamp",
    # Topics
    "Topics",
    # Payload schemas
    "ImageSubmittedPayload",
    "InferenceCompletedPayload",
    "AnnotationStoredPayload",
    "AnnotationCorrectedPayload",
    "EmbeddingCreatedPayload",
    "QuerySubmittedPayload",
    "QueryCompletedPayload",
    "DetectedObject",
    "EmbeddingInfo",
    "QueryResult",
    "EnvelopeSchema",
    # Validation
    "ValidationError",
    "validate_envelope",
    "validate_payload",
    "validate_event",
    "is_valid_event",
    "PAYLOAD_SCHEMAS",
]
