"""Event handlers for the inference service."""

import logging

from shared.events import EventEnvelope, create_envelope
from shared.events.topics import Topics
from shared.events.schema import ImageSubmittedPayload, validate_payload, ValidationError
from shared.broker import BaseBroker
from services.inference.mock_detector import mock_detect, MOCK_MODEL_VERSION


logger = logging.getLogger(__name__)


async def handle_image_submitted(envelope: EventEnvelope, broker: BaseBroker) -> None:
    """
    Handle image.submitted events by running mock detection.

    Processes the image with the mock detector and publishes
    inference.completed event with detected objects.

    Args:
        envelope: The image.submitted event envelope
        broker: Broker to publish results to
    """
    # Validate payload
    try:
        payload = validate_payload(Topics.IMAGE_SUBMITTED, envelope.payload)
        if not isinstance(payload, ImageSubmittedPayload):
            raise ValidationError("Invalid payload type")
    except ValidationError as e:
        logger.error(f"Invalid image.submitted payload: {e}")
        return

    image_id = payload.image_id
    logger.info(f"Processing image {image_id} from {payload.source}")

    # Run mock detection
    detected_objects = mock_detect(image_id)
    logger.info(f"Detected {len(detected_objects)} objects in image {image_id}")

    # Build result payload
    result_payload = {
        "image_id": image_id,
        "model_version": MOCK_MODEL_VERSION,
        "objects": [obj.to_dict() for obj in detected_objects],
    }

    # Publish inference.completed event
    result_envelope = create_envelope(Topics.INFERENCE_COMPLETED, result_payload)
    await broker.publish(Topics.INFERENCE_COMPLETED, result_envelope)

    logger.info(f"Published inference.completed for image {image_id}")


def create_inference_handler(broker: BaseBroker):
    """
    Create a handler function bound to a specific broker.

    This is useful for registering with the broker's subscribe method.

    Args:
        broker: Broker to use for publishing results

    Returns:
        Async handler function
    """

    async def handler(envelope: EventEnvelope) -> None:
        await handle_image_submitted(envelope, broker)

    return handler
