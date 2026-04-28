"""Event envelope for all pub-sub messages."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import json
import uuid


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def generate_timestamp() -> str:
    """Generate ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class EventEnvelope:
    """
    Standard envelope for all events in the system.

    Every message uses this envelope format. Schema validation
    lives in shared/events/schema.py.
    """
    topic: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=generate_event_id)
    timestamp: str = field(default_factory=generate_timestamp)
    schema_version: int = 1
    type: str = "publish"

    def to_dict(self) -> dict[str, Any]:
        """Convert envelope to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize envelope to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventEnvelope":
        """Create envelope from dictionary."""
        return cls(
            type=data.get("type", "publish"),
            topic=data["topic"],
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            schema_version=data.get("schema_version", 1),
            payload=data["payload"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EventEnvelope":
        """Deserialize envelope from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def create_envelope(topic: str, payload: dict[str, Any]) -> EventEnvelope:
    """Helper to create a new envelope with auto-generated id and timestamp."""
    return EventEnvelope(topic=topic, payload=payload)
