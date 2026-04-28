"""Upload service handlers."""

import logging
from dataclasses import dataclass
from typing import Optional

from shared.events import EventEnvelope, create_envelope, Topics
from shared.broker import BaseBroker

from .models import ImageRecord
from .repo import ImageRegistry, compute_content_hash, generate_image_id

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of an upload operation."""

    image_id: str
    is_duplicate: bool
    record: ImageRecord


async def handle_upload(
    path: str,
    source: str,
    registry: ImageRegistry,
    broker: BaseBroker,
) -> UploadResult:
    """
    Handle an image upload request.

    Implements idempotency: if the same path + source + content_hash
    is submitted twice, return the existing image_id and skip publishing.

    Args:
        path: Path to the image file
        source: Source identifier (e.g., "camera_A")
        registry: Image registry to store records
        broker: Broker to publish events

    Returns:
        UploadResult with image_id and duplicate status
    """
    # Compute content hash for idempotency check
    content_hash = compute_content_hash(path)

    # Check for existing record (idempotency)
    existing = await registry.get_by_hash(path, source, content_hash)
    if existing:
        logger.info(f"Duplicate upload detected: {existing.image_id}")
        return UploadResult(
            image_id=existing.image_id,
            is_duplicate=True,
            record=existing,
        )

    # Generate new image ID
    image_id = generate_image_id(path, source, content_hash)

    # Create record
    record = ImageRecord(
        image_id=image_id,
        path=path,
        source=source,
        content_hash=content_hash,
        status="submitted",
    )
    await registry.create(record)

    # Publish image.submitted event
    envelope = create_envelope(
        Topics.IMAGE_SUBMITTED,
        {
            "image_id": image_id,
            "path": path,
            "source": source,
        },
    )
    await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

    logger.info(f"Image uploaded: {image_id}")
    return UploadResult(
        image_id=image_id,
        is_duplicate=False,
        record=record,
    )


async def get_image_status(
    image_id: str,
    registry: ImageRegistry,
) -> Optional[ImageRecord]:
    """
    Get the status of an uploaded image.

    Args:
        image_id: The image ID to look up
        registry: Image registry to query

    Returns:
        ImageRecord if found, None otherwise
    """
    return await registry.get_by_id(image_id)


async def update_image_status(
    image_id: str,
    status: str,
    registry: ImageRegistry,
) -> bool:
    """
    Update the status of an image.

    Args:
        image_id: The image ID to update
        status: New status value
        registry: Image registry to update

    Returns:
        True if updated, False if not found
    """
    return await registry.update_status(image_id, status)
