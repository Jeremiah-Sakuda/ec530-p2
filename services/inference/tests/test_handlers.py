"""Tests for inference service event handlers."""

import pytest
from shared.events import EventEnvelope, create_envelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from services.inference.handlers import (
    handle_image_submitted,
    create_inference_handler,
)
from services.inference.mock_detector import MOCK_MODEL_VERSION


@pytest.fixture
def broker():
    """Create a fresh in-memory broker for each test."""
    b = InMemoryBroker()
    return b


@pytest.fixture
def valid_image_submitted_envelope():
    """Create a valid image.submitted event envelope."""
    return create_envelope(
        Topics.IMAGE_SUBMITTED,
        {
            "image_id": "img_test123",
            "path": "/images/test.jpg",
            "source": "camera_A",
        },
    )


class TestHandleImageSubmitted:
    """Tests for handle_image_submitted handler."""

    @pytest.mark.asyncio
    async def test_publishes_inference_completed_event(
        self, broker, valid_image_submitted_envelope
    ):
        """Handler should publish inference.completed event."""
        await broker.start()

        await handle_image_submitted(valid_image_submitted_envelope, broker)

        # Check that inference.completed was published
        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_inference_completed_contains_image_id(
        self, broker, valid_image_submitted_envelope
    ):
        """Published event should contain the original image_id."""
        await broker.start()

        await handle_image_submitted(valid_image_submitted_envelope, broker)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert published[0].payload["image_id"] == "img_test123"

    @pytest.mark.asyncio
    async def test_inference_completed_contains_model_version(
        self, broker, valid_image_submitted_envelope
    ):
        """Published event should contain model version."""
        await broker.start()

        await handle_image_submitted(valid_image_submitted_envelope, broker)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert published[0].payload["model_version"] == MOCK_MODEL_VERSION

    @pytest.mark.asyncio
    async def test_inference_completed_contains_objects(
        self, broker, valid_image_submitted_envelope
    ):
        """Published event should contain detected objects."""
        await broker.start()

        await handle_image_submitted(valid_image_submitted_envelope, broker)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        objects = published[0].payload["objects"]

        assert isinstance(objects, list)
        assert len(objects) >= 1
        assert len(objects) <= 5

        # Check object structure
        for obj in objects:
            assert "object_id" in obj
            assert "label" in obj
            assert "bbox" in obj
            assert "conf" in obj

    @pytest.mark.asyncio
    async def test_deterministic_detection_results(self, broker):
        """Same image_id should produce same detection results."""
        await broker.start()

        # Process same image twice
        envelope1 = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_determinism", "path": "/test.jpg", "source": "test"},
        )
        envelope2 = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_determinism", "path": "/test.jpg", "source": "test"},
        )

        await handle_image_submitted(envelope1, broker)
        await handle_image_submitted(envelope2, broker)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 2

        # Results should be identical
        assert published[0].payload["objects"] == published[1].payload["objects"]

    @pytest.mark.asyncio
    async def test_invalid_payload_does_not_crash(self, broker):
        """Handler should gracefully handle invalid payloads."""
        await broker.start()

        invalid_envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"invalid_field": "value"},  # Missing required fields
        )

        # Should not raise
        await handle_image_submitted(invalid_envelope, broker)

        # Should not publish anything
        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 0

    @pytest.mark.asyncio
    async def test_handles_empty_path(self, broker):
        """Handler should work with empty path."""
        await broker.start()

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_empty_path", "path": "", "source": "test"},
        )

        await handle_image_submitted(envelope, broker)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 1


class TestCreateInferenceHandler:
    """Tests for the handler factory function."""

    @pytest.mark.asyncio
    async def test_returns_callable(self, broker):
        """Factory should return a callable handler."""
        handler = create_inference_handler(broker)
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_processes_events(self, broker, valid_image_submitted_envelope):
        """Created handler should process events correctly."""
        await broker.start()

        handler = create_inference_handler(broker)
        await handler(valid_image_submitted_envelope)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_handler_can_be_used_with_subscribe(
        self, broker, valid_image_submitted_envelope
    ):
        """Handler should work when registered with broker."""
        await broker.start()

        handler = create_inference_handler(broker)
        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)

        # Publish should trigger the handler
        await broker.publish(Topics.IMAGE_SUBMITTED, valid_image_submitted_envelope)

        # Should have published to both topics
        image_submitted = broker.get_published_for_topic(Topics.IMAGE_SUBMITTED)
        inference_completed = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)

        assert len(image_submitted) == 1
        assert len(inference_completed) == 1
