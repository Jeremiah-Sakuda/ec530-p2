"""Event handlers for the vector index service."""

import logging

from shared.events import EventEnvelope
from shared.events.topics import Topics
from shared.events.schema import EmbeddingCreatedPayload, validate_payload, ValidationError
from shared.repos import VectorRepo
from services.embedding import mock_embed


logger = logging.getLogger(__name__)


async def handle_embedding_created(
    envelope: EventEnvelope,
    vector_repo: VectorRepo,
) -> None:
    """
    Handle embedding.created events by adding vectors to the index.

    For each embedding in the event, regenerates the vector deterministically
    and adds it to the FAISS index.

    Args:
        envelope: The embedding.created event envelope
        vector_repo: Vector repository to add embeddings to
    """
    # Validate payload
    try:
        payload = validate_payload(Topics.EMBEDDING_CREATED, envelope.payload)
        if not isinstance(payload, EmbeddingCreatedPayload):
            raise ValidationError("Invalid payload type")
    except ValidationError as e:
        logger.error(f"Invalid embedding.created payload: {e}")
        return

    image_id = payload.image_id
    embeddings = payload.embeddings

    logger.info(f"Indexing {len(embeddings)} embeddings for image {image_id}")

    for emb in embeddings:
        object_id = emb.object_id

        # Regenerate vector deterministically
        # In a real system, we would fetch from vector_ref storage
        vector = mock_embed(image_id, object_id)

        # Add to index (will replace if exists)
        vector_repo.add(image_id, object_id, vector)

    logger.info(f"Indexed {len(embeddings)} vectors for image {image_id}")


def create_embedding_created_handler(vector_repo: VectorRepo):
    """
    Create a handler function bound to a specific vector repo.

    Args:
        vector_repo: Vector repository to use

    Returns:
        Async handler function
    """

    async def handler(envelope: EventEnvelope) -> None:
        await handle_embedding_created(envelope, vector_repo)

    return handler
