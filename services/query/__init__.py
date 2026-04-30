"""Query service for visual object similarity search."""

from services.query.handlers import (
    handle_query_submitted,
    create_query_submitted_handler,
    execute_query,
    get_object_label,
)
from services.query.api import app, configure

__all__ = [
    "handle_query_submitted",
    "create_query_submitted_handler",
    "execute_query",
    "get_object_label",
    "app",
    "configure",
]
