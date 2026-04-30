"""Base document repository abstraction for annotation storage."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AnnotationDocument:
    """
    Annotation document stored in the repository.

    Represents all annotations for a single image, including
    detected objects, review status, and correction history.
    """

    image_id: str
    objects: list[dict[str, Any]]
    model_version: str
    status: str = "pending"  # pending, reviewed, corrected
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: list[dict[str, Any]] = field(default_factory=list)
    processed_event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "image_id": self.image_id,
            "objects": self.objects,
            "model_version": self.model_version,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history,
            "processed_event_ids": self.processed_event_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationDocument":
        """Create from dictionary."""
        return cls(
            image_id=data["image_id"],
            objects=data.get("objects", []),
            model_version=data.get("model_version", "unknown"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            history=data.get("history", []),
            processed_event_ids=data.get("processed_event_ids", []),
        )


class DocumentRepo(ABC):
    """
    Abstract base class for document repositories.

    Provides storage for annotation documents with support for:
    - CRUD operations
    - Query by filters
    - Idempotency tracking via event IDs
    """

    @abstractmethod
    async def upsert(self, image_id: str, data: dict[str, Any]) -> None:
        """
        Insert or update an annotation document.

        Args:
            image_id: Unique image identifier
            data: Document data to store
        """
        pass

    @abstractmethod
    async def get(self, image_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve an annotation document by image ID.

        Args:
            image_id: Unique image identifier

        Returns:
            Document data or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, image_id: str) -> bool:
        """
        Delete an annotation document.

        Args:
            image_id: Unique image identifier

        Returns:
            True if document was deleted, False if not found
        """
        pass

    @abstractmethod
    async def query(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Query documents by filters.

        Supported filter fields:
        - label: Filter by object label
        - min_conf: Minimum confidence threshold
        - status: Document status

        Args:
            filters: Query filter dictionary

        Returns:
            List of matching documents
        """
        pass

    @abstractmethod
    async def add_processed_event(self, image_id: str, event_id: str) -> None:
        """
        Record that an event has been processed for this image.

        Used for idempotency - prevents duplicate event processing.

        Args:
            image_id: Image identifier
            event_id: Event identifier to record
        """
        pass

    @abstractmethod
    async def has_processed_event(self, image_id: str, event_id: str) -> bool:
        """
        Check if an event has already been processed.

        Args:
            image_id: Image identifier
            event_id: Event identifier to check

        Returns:
            True if event was already processed
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Return total number of documents in the repository.

        Returns:
            Document count
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all documents from the repository."""
        pass
