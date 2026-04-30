"""Annotation service for storing and managing annotations."""

from services.annotation.handlers import (
    handle_inference_completed,
    handle_annotation_corrected,
    create_inference_completed_handler,
    create_annotation_corrected_handler,
    apply_patch,
    build_annotation_document,
)
from services.annotation.api import app, configure

__all__ = [
    "handle_inference_completed",
    "handle_annotation_corrected",
    "create_inference_completed_handler",
    "create_annotation_corrected_handler",
    "apply_patch",
    "build_annotation_document",
    "app",
    "configure",
]
