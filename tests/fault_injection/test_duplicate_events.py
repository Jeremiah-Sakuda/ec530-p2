"""Tests for duplicate event handling and idempotency."""

import pytest

from shared.events import create_envelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from shared.repos import VectorRepo, InMemoryDocumentRepo

from services.inference.handlers import handle_image_submitted
from services.annotation.handlers import handle_inference_completed, handle_annotation_corrected
from services.embedding.handlers import handle_annotation_stored
from services.vector_index.handlers import handle_embedding_created

from tools.event_generator import EventGenerator


@pytest.fixture
def broker():
    """Create a fresh in-memory broker."""
    return InMemoryBroker()


@pytest.fixture
def vector_repo():
    """Create a fresh vector repository."""
    return VectorRepo(dim=128)


@pytest.fixture
def document_repo():
    """Create a fresh document repository."""
    return InMemoryDocumentRepo()


class TestDuplicateEventIdempotency:
    """Test that duplicate events are handled idempotently."""

    @pytest.mark.asyncio
    async def test_duplicate_inference_completed_is_idempotent(
        self, broker, document_repo
    ):
        """Same inference.completed event should only be processed once."""
        await broker.start()

        # Create a single inference.completed event
        envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_dup_test",
                "model_version": "mock_v1",
                "objects": [
                    {"object_id": "obj_0", "label": "car", "bbox": [0, 0, 100, 100], "conf": 0.9},
                ],
            },
        )

        # Process the same event multiple times
        await handle_inference_completed(envelope, document_repo, broker)
        await handle_inference_completed(envelope, document_repo, broker)
        await handle_inference_completed(envelope, document_repo, broker)

        # Should only have one document
        assert await document_repo.count() == 1

        # Should only have published annotation.stored once
        published = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_duplicate_annotation_corrected_is_idempotent(
        self, broker, document_repo
    ):
        """Same correction event should only be applied once."""
        await broker.start()

        # First, create an annotation
        inference_envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_corr_dup",
                "model_version": "mock_v1",
                "objects": [
                    {"object_id": "obj_0", "label": "car", "bbox": [0, 0, 100, 100], "conf": 0.9},
                ],
            },
        )
        await handle_inference_completed(inference_envelope, document_repo, broker)

        # Create correction event
        correction_envelope = create_envelope(
            Topics.ANNOTATION_CORRECTED,
            {
                "image_id": "img_corr_dup",
                "patch": {"objects.0.label": "truck"},
                "reviewer": "test_user",
            },
        )

        # Apply correction multiple times
        await handle_annotation_corrected(correction_envelope, document_repo, broker)
        await handle_annotation_corrected(correction_envelope, document_repo, broker)
        await handle_annotation_corrected(correction_envelope, document_repo, broker)

        # Should only have one history entry
        doc = await document_repo.get("img_corr_dup")
        assert len(doc["history"]) == 1

    @pytest.mark.asyncio
    async def test_duplicate_events_in_full_pipeline(
        self, broker, vector_repo, document_repo
    ):
        """Duplicate events should not cause duplicate data in full pipeline."""
        await broker.start()

        generator = EventGenerator(seed=42)
        events = generator.generate_image_submitted(count=1)
        event = events[0]

        # Process the same image.submitted event 3 times
        for _ in range(3):
            await handle_image_submitted(event, broker)

        # Process all resulting events (each image submission creates new inference events)
        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        # Only one annotation should exist (others should be duplicates)
        assert await document_repo.count() == 1

        # Process embedding events
        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        # Vectors should exist (same vectors are replaced, not duplicated)
        assert vector_repo.ntotal >= 1


class TestEventIdTracking:
    """Test event ID tracking for idempotency."""

    @pytest.mark.asyncio
    async def test_processed_events_are_tracked(self, broker, document_repo):
        """Processed events should be recorded."""
        await broker.start()

        envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_track_test",
                "model_version": "mock_v1",
                "objects": [],
            },
        )

        await handle_inference_completed(envelope, document_repo, broker)

        # Event should be tracked
        assert await document_repo.has_processed_event(
            "img_track_test", envelope.event_id
        )

    @pytest.mark.asyncio
    async def test_different_events_for_same_image_both_processed(
        self, broker, document_repo
    ):
        """Different events for same image should all be processed."""
        await broker.start()

        # Create multiple correction events with different event_ids
        for i in range(3):
            envelope = create_envelope(
                Topics.INFERENCE_COMPLETED,
                {
                    "image_id": "img_multi_event",
                    "model_version": f"v{i}",
                    "objects": [
                        {"object_id": f"obj_{i}", "label": "car", "bbox": [0, 0, 100, 100], "conf": 0.9},
                    ],
                },
            )
            await handle_inference_completed(envelope, document_repo, broker)

        # Should have processed all events (last one wins for document content)
        assert await document_repo.count() == 1

        # All event IDs should be tracked
        doc = await document_repo.get("img_multi_event")
        assert len(doc.get("processed_event_ids", [])) == 3
