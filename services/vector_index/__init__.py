"""Vector index service for similarity search."""

from services.vector_index.handlers import (
    handle_embedding_created,
    create_embedding_created_handler,
)
from services.vector_index.api import app, configure

__all__ = [
    "handle_embedding_created",
    "create_embedding_created_handler",
    "app",
    "configure",
]
