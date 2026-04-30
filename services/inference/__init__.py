"""Inference service for object detection."""

from services.inference.mock_detector import mock_detect, DetectedObject, MOCK_MODEL_VERSION
from services.inference.handlers import handle_image_submitted, create_inference_handler

__all__ = [
    "mock_detect",
    "DetectedObject",
    "MOCK_MODEL_VERSION",
    "handle_image_submitted",
    "create_inference_handler",
]
