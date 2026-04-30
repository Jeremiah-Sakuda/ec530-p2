"""Event handlers for the embedding service."""

import logging

from shared.events import EventEnvelope, create_envelope
from shared.events.topics import Topics
from shared.events.schema import AnnotationStoredPayload, validate_payload, ValidationError
from shared.broker import BaseBroker
from services.embedding.mock_embedder import embed_objects


logger = logging.getLogger(__name__)


async def handle_annotation_stored(
    envelope: EventEnvelope,
    broker: BaseBroker,
) -> None:
    """
    Handle annotation.stored events by generating embeddings.

    Creates embedding vectors for each object in the annotation
    and publishes embedding.created event.

    Args:
        envelope: The annotation.stored event envelope
        broker: Broker to publish results to
    """
    # Validate payload
    try:
        payload = validate_payload(Topics.ANNOTATION_STORED, envelope.payload)
        if not isinstance(payload, AnnotationStoredPayload):
            raise ValidationError("Invalid payload type")
    except ValidationError as e:
        logger.error(f"Invalid annotation.stored payload: {e}")
        return

    image_id = payload.image_id
    object_ids = payload.object_ids

    logger.info(f"Generating embeddings for image {image_id} ({len(object_ids)} objects)")

    # Generate embeddings for all objects
    embedding_results = embed_objects(image_id, object_ids)

    # Build embedding.created payload
    embeddings_payload = [result.to_dict() for result in embedding_results]

    result_payload = {
        "image_id": image_id,
        "embeddings": embeddings_payload,
    }

    # Publish embedding.created event
    result_envelope = create_envelope(Topics.EMBEDDING_CREATED, result_payload)
    await broker.publish(Topics.EMBEDDING_CREATED, result_envelope)

    logger.info(f"Published embedding.created for image {image_id}")


def create_annotation_stored_handler(broker: BaseBroker):
    """
    Create a handler function bound to a specific broker.

    Args:
        broker: Broker to use for publishing results

    Returns:
        Async handler function
    """

    async def handler(envelope: EventEnvelope) -> None:
        await handle_annotation_stored(envelope, broker)

    return handler
