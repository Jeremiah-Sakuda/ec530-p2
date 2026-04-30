"""Event handlers for the annotation service."""

import logging
from datetime import datetime, timezone
from typing import Any

from shared.events import EventEnvelope, create_envelope
from shared.events.topics import Topics
from shared.events.schema import (
    InferenceCompletedPayload,
    AnnotationCorrectedPayload,
    validate_payload,
    ValidationError,
)
from shared.broker import BaseBroker
from shared.repos import DocumentRepo


logger = logging.getLogger(__name__)


def build_annotation_document(payload: InferenceCompletedPayload) -> dict[str, Any]:
    """
    Build an annotation document from inference results.

    Args:
        payload: Validated inference completed payload

    Returns:
        Document dictionary ready for storage
    """
    return {
        "image_id": payload.image_id,
        "objects": [obj.model_dump() for obj in payload.objects],
        "model_version": payload.model_version,
        "status": "pending",
        "history": [],
    }


async def handle_inference_completed(
    envelope: EventEnvelope,
    repo: DocumentRepo,
    broker: BaseBroker,
) -> None:
    """
    Handle inference.completed events by storing annotations.

    Processes inference results, stores them in the document repository,
    and publishes annotation.stored event.

    Args:
        envelope: The inference.completed event envelope
        repo: Document repository for storage
        broker: Broker to publish results to
    """
    # Validate payload
    try:
        payload = validate_payload(Topics.INFERENCE_COMPLETED, envelope.payload)
        if not isinstance(payload, InferenceCompletedPayload):
            raise ValidationError("Invalid payload type")
    except ValidationError as e:
        logger.error(f"Invalid inference.completed payload: {e}")
        return

    image_id = payload.image_id

    # Idempotency check
    if await repo.has_processed_event(image_id, envelope.event_id):
        logger.info(f"Duplicate event {envelope.event_id} for {image_id}, skipping")
        return

    logger.info(f"Storing annotation for image {image_id}")

    # Build annotation document
    doc = build_annotation_document(payload)

    # Preserve existing processed_event_ids from previous document
    existing_doc = await repo.get(image_id)
    if existing_doc:
        doc["processed_event_ids"] = existing_doc.get("processed_event_ids", [])

    await repo.upsert(image_id, doc)
    await repo.add_processed_event(image_id, envelope.event_id)

    # Publish annotation.stored event
    stored_payload = {
        "image_id": image_id,
        "object_ids": [obj.object_id for obj in payload.objects],
        "model_version": payload.model_version,
    }
    stored_envelope = create_envelope(Topics.ANNOTATION_STORED, stored_payload)
    await broker.publish(Topics.ANNOTATION_STORED, stored_envelope)

    logger.info(f"Published annotation.stored for image {image_id}")


async def handle_annotation_corrected(
    envelope: EventEnvelope,
    repo: DocumentRepo,
    broker: BaseBroker,
) -> None:
    """
    Handle annotation.corrected events by applying corrections.

    Applies the correction patch, adds to history, updates status,
    and re-publishes annotation.stored to trigger re-embedding.

    Args:
        envelope: The annotation.corrected event envelope
        repo: Document repository for storage
        broker: Broker to publish results to
    """
    # Validate payload
    try:
        payload = validate_payload(Topics.ANNOTATION_CORRECTED, envelope.payload)
        if not isinstance(payload, AnnotationCorrectedPayload):
            raise ValidationError("Invalid payload type")
    except ValidationError as e:
        logger.error(f"Invalid annotation.corrected payload: {e}")
        return

    image_id = payload.image_id

    # Idempotency check
    if await repo.has_processed_event(image_id, envelope.event_id):
        logger.info(f"Duplicate correction event {envelope.event_id}, skipping")
        return

    # Get existing annotation
    doc = await repo.get(image_id)
    if doc is None:
        logger.error(f"No annotation found for image {image_id}")
        return

    logger.info(f"Applying correction to image {image_id} by {payload.reviewer}")

    # Apply patch
    doc = apply_patch(doc, payload.patch)

    # Add to history
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": envelope.event_id,
        "reviewer": payload.reviewer,
        "patch": payload.patch,
    }
    doc.setdefault("history", []).append(history_entry)

    # Update status
    doc["status"] = "corrected"

    # Save updated document
    await repo.upsert(image_id, doc)
    await repo.add_processed_event(image_id, envelope.event_id)

    # Re-publish annotation.stored to trigger re-embedding
    object_ids = [obj["object_id"] for obj in doc.get("objects", [])]
    stored_payload = {
        "image_id": image_id,
        "object_ids": object_ids,
        "model_version": doc.get("model_version", "unknown"),
    }
    stored_envelope = create_envelope(Topics.ANNOTATION_STORED, stored_payload)
    await broker.publish(Topics.ANNOTATION_STORED, stored_envelope)

    logger.info(f"Re-published annotation.stored for corrected image {image_id}")


def apply_patch(doc: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """
    Apply a patch to an annotation document.

    Supports dot notation for nested updates (e.g., "objects.0.label").

    Args:
        doc: Original document
        patch: Patch dictionary with dot-notation keys

    Returns:
        Updated document
    """
    for key, value in patch.items():
        parts = key.split(".")
        target = doc

        # Navigate to the nested location
        for part in parts[:-1]:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = target.setdefault(part, {})

        # Apply the value
        final_key = parts[-1]
        if final_key.isdigit():
            target[int(final_key)] = value
        else:
            target[final_key] = value

    return doc


def create_inference_completed_handler(repo: DocumentRepo, broker: BaseBroker):
    """
    Create a handler function for inference.completed events.

    Args:
        repo: Document repository
        broker: Message broker

    Returns:
        Async handler function
    """

    async def handler(envelope: EventEnvelope) -> None:
        await handle_inference_completed(envelope, repo, broker)

    return handler


def create_annotation_corrected_handler(repo: DocumentRepo, broker: BaseBroker):
    """
    Create a handler function for annotation.corrected events.

    Args:
        repo: Document repository
        broker: Message broker

    Returns:
        Async handler function
    """

    async def handler(envelope: EventEnvelope) -> None:
        await handle_annotation_corrected(envelope, repo, broker)

    return handler
