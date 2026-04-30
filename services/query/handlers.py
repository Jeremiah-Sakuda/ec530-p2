"""Event handlers for the query service."""

import logging
from typing import Any, List, Optional

from shared.events import EventEnvelope, create_envelope
from shared.events.topics import Topics
from shared.events.schema import QuerySubmittedPayload, validate_payload, ValidationError
from shared.broker import BaseBroker
from shared.repos import VectorRepo, DocumentRepo
from services.embedding import mock_embed_text, mock_embed_image_query


logger = logging.getLogger(__name__)


def get_object_label(annotation: Optional[dict[str, Any]], object_id: str) -> str:
    """
    Extract object label from an annotation document.

    Args:
        annotation: Annotation document or None
        object_id: Object ID to find

    Returns:
        Object label or "unknown"
    """
    if annotation is None:
        return "unknown"

    for obj in annotation.get("objects", []):
        if obj.get("object_id") == object_id:
            return obj.get("label", "unknown")

    return "unknown"


async def handle_query_submitted(
    envelope: EventEnvelope,
    vector_repo: VectorRepo,
    document_repo: DocumentRepo,
    broker: BaseBroker,
) -> None:
    """
    Handle query.submitted events by performing similarity search.

    Processes the query, searches the vector index, hydrates results
    with annotation data, and publishes query.completed event.

    Args:
        envelope: The query.submitted event envelope
        vector_repo: Vector repository for similarity search
        document_repo: Document repository for hydration
        broker: Broker to publish results to
    """
    # Validate payload
    try:
        payload = validate_payload(Topics.QUERY_SUBMITTED, envelope.payload)
        if not isinstance(payload, QuerySubmittedPayload):
            raise ValidationError("Invalid payload type")
    except ValidationError as e:
        logger.error(f"Invalid query.submitted payload: {e}")
        return

    query_id = payload.query_id
    kind = payload.kind
    value = payload.value
    top_k = payload.top_k

    logger.info(f"Processing {kind} query {query_id}: '{value}' (top_k={top_k})")

    # Generate query vector
    if kind == "text":
        query_vector = mock_embed_text(value)
    else:  # kind == "image"
        query_vector = mock_embed_image_query(value)

    # Search vector index
    search_results = vector_repo.search(query_vector, top_k)

    # Hydrate results with annotation data
    hydrated_results: List[dict[str, Any]] = []
    for result in search_results:
        annotation = await document_repo.get(result.image_id)
        label = get_object_label(annotation, result.object_id)

        hydrated_results.append({
            "image_id": result.image_id,
            "object_id": result.object_id,
            "score": result.score,
            "label": label,
        })

    # Publish query.completed event
    result_payload = {
        "query_id": query_id,
        "results": hydrated_results,
    }
    result_envelope = create_envelope(Topics.QUERY_COMPLETED, result_payload)
    await broker.publish(Topics.QUERY_COMPLETED, result_envelope)

    logger.info(f"Published query.completed for query {query_id} with {len(hydrated_results)} results")


def create_query_submitted_handler(
    vector_repo: VectorRepo,
    document_repo: DocumentRepo,
    broker: BaseBroker,
):
    """
    Create a handler function bound to specific dependencies.

    Args:
        vector_repo: Vector repository for search
        document_repo: Document repository for hydration
        broker: Broker for publishing results

    Returns:
        Async handler function
    """

    async def handler(envelope: EventEnvelope) -> None:
        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

    return handler


async def execute_query(
    kind: str,
    value: str,
    top_k: int,
    vector_repo: VectorRepo,
    document_repo: DocumentRepo,
) -> List[dict[str, Any]]:
    """
    Execute a synchronous query.

    This is the synchronous API version, not going through events.

    Args:
        kind: Query type ('text' or 'image')
        value: Query value
        top_k: Number of results
        vector_repo: Vector repository for search
        document_repo: Document repository for hydration

    Returns:
        List of hydrated query results
    """
    logger.info(f"Executing {kind} query: '{value}' (top_k={top_k})")

    # Generate query vector
    if kind == "text":
        query_vector = mock_embed_text(value)
    else:  # kind == "image"
        query_vector = mock_embed_image_query(value)

    # Search vector index
    search_results = vector_repo.search(query_vector, top_k)

    # Hydrate results with annotation data
    hydrated_results: List[dict[str, Any]] = []
    for result in search_results:
        annotation = await document_repo.get(result.image_id)
        label = get_object_label(annotation, result.object_id)

        hydrated_results.append({
            "image_id": result.image_id,
            "object_id": result.object_id,
            "score": result.score,
            "label": label,
        })

    return hydrated_results
