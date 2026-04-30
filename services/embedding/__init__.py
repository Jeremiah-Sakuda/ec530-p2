"""Embedding service for generating vector embeddings."""

from services.embedding.mock_embedder import (
    mock_embed,
    mock_embed_text,
    mock_embed_image_query,
    embed_objects,
    EmbeddingResult,
    EMBEDDING_DIM,
)
from services.embedding.handlers import (
    handle_annotation_stored,
    create_annotation_stored_handler,
)

__all__ = [
    "mock_embed",
    "mock_embed_text",
    "mock_embed_image_query",
    "embed_objects",
    "EmbeddingResult",
    "EMBEDDING_DIM",
    "handle_annotation_stored",
    "create_annotation_stored_handler",
]
