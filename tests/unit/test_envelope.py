"""Tests for event envelope creation and serialization."""

import json
import pytest
from shared.events import (
    EventEnvelope,
    create_envelope,
    generate_event_id,
    generate_timestamp,
    Topics,
)


class TestGenerateEventId:
    """Tests for event ID generation."""

    def test_generates_string(self):
        event_id = generate_event_id()
        assert isinstance(event_id, str)

    def test_starts_with_evt_prefix(self):
        event_id = generate_event_id()
        assert event_id.startswith("evt_")

    def test_generates_unique_ids(self):
        ids = [generate_event_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestGenerateTimestamp:
    """Tests for timestamp generation."""

    def test_generates_string(self):
        ts = generate_timestamp()
        assert isinstance(ts, str)

    def test_ends_with_z(self):
        ts = generate_timestamp()
        assert ts.endswith("Z")

    def test_is_iso_format(self):
        ts = generate_timestamp()
        # Basic ISO 8601 check
        assert "T" in ts


class TestEventEnvelope:
    """Tests for EventEnvelope dataclass."""

    def test_create_with_required_fields(self):
        envelope = EventEnvelope(
            topic=Topics.IMAGE_SUBMITTED,
            payload={"image_id": "img_001", "path": "/images/test.jpg", "source": "camera_A"},
        )
        assert envelope.topic == Topics.IMAGE_SUBMITTED
        assert envelope.payload["image_id"] == "img_001"
        assert envelope.type == "publish"
        assert envelope.schema_version == 1

    def test_auto_generates_event_id(self):
        envelope = EventEnvelope(
            topic=Topics.IMAGE_SUBMITTED,
            payload={"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        assert envelope.event_id.startswith("evt_")

    def test_auto_generates_timestamp(self):
        envelope = EventEnvelope(
            topic=Topics.IMAGE_SUBMITTED,
            payload={"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        assert envelope.timestamp.endswith("Z")

    def test_to_dict(self):
        envelope = EventEnvelope(
            topic=Topics.IMAGE_SUBMITTED,
            payload={"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
            event_id="evt_test123",
            timestamp="2026-04-07T14:33:00Z",
        )
        d = envelope.to_dict()
        assert d["type"] == "publish"
        assert d["topic"] == Topics.IMAGE_SUBMITTED
        assert d["event_id"] == "evt_test123"
        assert d["timestamp"] == "2026-04-07T14:33:00Z"
        assert d["schema_version"] == 1
        assert d["payload"]["image_id"] == "img_001"

    def test_to_json(self):
        envelope = EventEnvelope(
            topic=Topics.IMAGE_SUBMITTED,
            payload={"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
            event_id="evt_test123",
            timestamp="2026-04-07T14:33:00Z",
        )
        json_str = envelope.to_json()
        data = json.loads(json_str)
        assert data["topic"] == Topics.IMAGE_SUBMITTED
        assert data["event_id"] == "evt_test123"

    def test_from_dict(self):
        data = {
            "type": "publish",
            "topic": Topics.INFERENCE_COMPLETED,
            "event_id": "evt_abc123",
            "timestamp": "2026-04-07T14:33:00Z",
            "schema_version": 1,
            "payload": {"image_id": "img_002", "model_version": "v1", "objects": []},
        }
        envelope = EventEnvelope.from_dict(data)
        assert envelope.topic == Topics.INFERENCE_COMPLETED
        assert envelope.event_id == "evt_abc123"
        assert envelope.payload["image_id"] == "img_002"

    def test_from_json(self):
        json_str = json.dumps({
            "type": "publish",
            "topic": Topics.ANNOTATION_STORED,
            "event_id": "evt_xyz789",
            "timestamp": "2026-04-07T14:33:00Z",
            "schema_version": 1,
            "payload": {"image_id": "img_003", "object_ids": ["obj_0"], "model_version": "v1"},
        })
        envelope = EventEnvelope.from_json(json_str)
        assert envelope.topic == Topics.ANNOTATION_STORED
        assert envelope.event_id == "evt_xyz789"

    def test_roundtrip_serialization(self):
        original = EventEnvelope(
            topic=Topics.EMBEDDING_CREATED,
            payload={"image_id": "img_004", "embeddings": []},
            event_id="evt_roundtrip",
            timestamp="2026-04-07T14:33:00Z",
        )
        json_str = original.to_json()
        restored = EventEnvelope.from_json(json_str)
        assert original.topic == restored.topic
        assert original.event_id == restored.event_id
        assert original.payload == restored.payload


class TestCreateEnvelope:
    """Tests for the create_envelope helper."""

    def test_creates_envelope_with_topic_and_payload(self):
        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {"query_id": "q_001", "kind": "text", "value": "red car", "top_k": 5},
        )
        assert envelope.topic == Topics.QUERY_SUBMITTED
        assert envelope.payload["query_id"] == "q_001"

    def test_auto_generates_fields(self):
        envelope = create_envelope(Topics.QUERY_COMPLETED, {"query_id": "q_001", "results": []})
        assert envelope.event_id.startswith("evt_")
        assert envelope.timestamp.endswith("Z")
        assert envelope.type == "publish"
        assert envelope.schema_version == 1
