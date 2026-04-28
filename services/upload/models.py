"""Data models for the upload service."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ImageRecord:
    """
    Record of an uploaded image in the registry.

    Represents the image metadata stored by the upload service.
    """

    image_id: str
    path: str
    source: str
    content_hash: str
    status: str = "submitted"
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "path": self.path,
            "source": self.source,
            "content_hash": self.content_hash,
            "status": self.status,
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageRecord":
        """Create from dictionary."""
        return cls(
            image_id=data["image_id"],
            path=data["path"],
            source=data["source"],
            content_hash=data["content_hash"],
            status=data.get("status", "submitted"),
            submitted_at=data.get("submitted_at", datetime.now(timezone.utc).isoformat()),
        )
