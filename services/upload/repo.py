"""Image registry repository for the upload service."""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .models import ImageRecord

logger = logging.getLogger(__name__)


def compute_content_hash(path: str) -> str:
    """
    Compute a hash of the file content.

    For actual files, computes MD5 of content.
    For non-existent files (test mode), uses path hash.
    """
    file_path = Path(path)
    if file_path.exists():
        # Hash actual file content
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    else:
        # For testing: hash the path string
        return hashlib.md5(path.encode()).hexdigest()


def generate_image_id(path: str, source: str, content_hash: str) -> str:
    """Generate a deterministic image ID from path, source, and content hash."""
    combined = f"{path}:{source}:{content_hash}"
    hash_hex = hashlib.md5(combined.encode()).hexdigest()[:12]
    return f"img_{hash_hex}"


class ImageRegistry(ABC):
    """
    Abstract base class for image registry.

    The upload service owns this data store. Other services
    must not read from it directly.
    """

    @abstractmethod
    async def create(self, record: ImageRecord) -> None:
        """
        Create a new image record.

        Args:
            record: ImageRecord to store
        """
        pass

    @abstractmethod
    async def get_by_id(self, image_id: str) -> Optional[ImageRecord]:
        """
        Get an image record by ID.

        Args:
            image_id: The image ID to look up

        Returns:
            ImageRecord if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_hash(
        self, path: str, source: str, content_hash: str
    ) -> Optional[ImageRecord]:
        """
        Get an image record by path, source, and content hash.

        Used for idempotency check - if the same image was already
        submitted, return existing record.

        Args:
            path: Image file path
            source: Source identifier
            content_hash: Hash of image content

        Returns:
            ImageRecord if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_status(self, image_id: str, status: str) -> bool:
        """
        Update the status of an image record.

        Args:
            image_id: The image ID to update
            status: New status value

        Returns:
            True if record was found and updated, False otherwise
        """
        pass

    @abstractmethod
    async def list_all(self) -> list[ImageRecord]:
        """
        List all image records.

        Returns:
            List of all ImageRecord objects
        """
        pass


class InMemoryImageRegistry(ImageRegistry):
    """
    In-memory implementation of image registry for testing.
    """

    def __init__(self):
        self._records: dict[str, ImageRecord] = {}
        self._hash_index: dict[str, str] = {}  # hash_key -> image_id

    def _make_hash_key(self, path: str, source: str, content_hash: str) -> str:
        """Create a composite key for hash-based lookup."""
        return f"{path}:{source}:{content_hash}"

    async def create(self, record: ImageRecord) -> None:
        """Store a new image record."""
        self._records[record.image_id] = record
        hash_key = self._make_hash_key(record.path, record.source, record.content_hash)
        self._hash_index[hash_key] = record.image_id
        logger.debug(f"Created image record: {record.image_id}")

    async def get_by_id(self, image_id: str) -> Optional[ImageRecord]:
        """Get record by image ID."""
        return self._records.get(image_id)

    async def get_by_hash(
        self, path: str, source: str, content_hash: str
    ) -> Optional[ImageRecord]:
        """Get record by composite hash key."""
        hash_key = self._make_hash_key(path, source, content_hash)
        image_id = self._hash_index.get(hash_key)
        if image_id:
            return self._records.get(image_id)
        return None

    async def update_status(self, image_id: str, status: str) -> bool:
        """Update record status."""
        if image_id in self._records:
            self._records[image_id].status = status
            logger.debug(f"Updated status for {image_id}: {status}")
            return True
        return False

    async def list_all(self) -> list[ImageRecord]:
        """Return all records."""
        return list(self._records.values())

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._records.clear()
        self._hash_index.clear()
