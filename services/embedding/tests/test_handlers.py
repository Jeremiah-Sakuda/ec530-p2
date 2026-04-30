"""Tests for embedding service event handlers."""

import pytest
from shared.events import create_envelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from services.embedding.handlers import (
    handle_annotation_stored,
    create_annotation_stored_handler,
)
from services.embedding.mock_embedder import EMBEDDING_DIM


@pytest.fixture
def broker():
    """Create a fresh in-memory broker for each test."""
    return InMemoryBroker()


@pytest.fixture
def valid_annotation_stored_envelope():
    """Create a valid annotation.stored event envelope."""
    return create_envelope(
        Topics.ANNOTATION_STORED,
        {
            "image_id": "img_test123",
            "object_ids": ["obj_0", "obj_1", "obj_2"],
            "model_version": "mock_v1",
        },
    )


class TestHandleAnnotationStored:
    """Tests for handle_annotation_stored handler."""

    @pytest.mark.asyncio
    async def test_publishes_embedding_created_event(
        self, broker, valid_annotation_stored_envelope
    ):
        """Handler should publish embedding.created event."""
        await broker.start()

        await handle_annotation_stored(valid_annotation_stored_envelope, broker)

        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_embedding_created_contains_image_id(
        self, broker, valid_annotation_stored_envelope
    ):
        """Published event should contain the original image_id."""
        await broker.start()

        await handle_annotation_stored(valid_annotation_stored_envelope, broker)

        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        assert published[0].payload["image_id"] == "img_test123"

    @pytest.mark.asyncio
    async def test_embedding_created_contains_all_embeddings(
        self, broker, valid_annotation_stored_envelope
    ):
        """Published event should contain embeddings for all objects."""
        await broker.start()

        await handle_annotation_stored(valid_annotation_stored_envelope, broker)

        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        embeddings = published[0].payload["embeddings"]

        assert len(embeddings) == 3

        object_ids = [e["object_id"] for e in embeddings]
        assert "obj_0" in object_ids
        assert "obj_1" in object_ids
        assert "obj_2" in object_ids

    @pytest.mark.asyncio
    async def test_embeddings_have_correct_structure(
        self, broker, valid_annotation_stored_envelope
    ):
        """Each embedding should have required fields."""
        await broker.start()

        await handle_annotation_stored(valid_annotation_stored_envelope, broker)

        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        embeddings = published[0].payload["embeddings"]

        for emb in embeddings:
            assert "object_id" in emb
            assert "vector_ref" in emb
            assert "dim" in emb
            assert emb["dim"] == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, broker):
        """Same input should produce same embeddings."""
        await broker.start()

        envelope1 = create_envelope(
            Topics.ANNOTATION_STORED,
            {
                "image_id": "img_determinism",
                "object_ids": ["obj_0"],
                "model_version": "v1",
            },
        )
        envelope2 = create_envelope(
            Topics.ANNOTATION_STORED,
            {
                "image_id": "img_determinism",
                "object_ids": ["obj_0"],
                "model_version": "v1",
            },
        )

        await handle_annotation_stored(envelope1, broker)
        await handle_annotation_stored(envelope2, broker)

        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        assert len(published) == 2

        # vector_refs should be the same for same image_id/object_id
        ref1 = published[0].payload["embeddings"][0]["vector_ref"]
        ref2 = published[1].payload["embeddings"][0]["vector_ref"]
        assert ref1 == ref2

    @pytest.mark.asyncio
    async def test_invalid_payload_does_not_crash(self, broker):
        """Handler should gracefully handle invalid payloads."""
        await broker.start()

        invalid_envelope = create_envelope(
            Topics.ANNOTATION_STORED,
            {"invalid_field": "value"},
        )

        # Should not raise
        await handle_annotation_stored(invalid_envelope, broker)

        # Should not publish anything
        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        assert len(published) == 0

    @pytest.mark.asyncio
    async def test_empty_object_ids(self, broker):
        """Handler should handle empty object_ids list."""
        await broker.start()

        envelope = create_envelope(
            Topics.ANNOTATION_STORED,
            {
                "image_id": "img_empty",
                "object_ids": [],
                "model_version": "v1",
            },
        )

        await handle_annotation_stored(envelope, broker)

        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        assert len(published) == 1
        assert published[0].payload["embeddings"] == []


class TestCreateAnnotationStoredHandler:
    """Tests for the handler factory function."""

    @pytest.mark.asyncio
    async def test_returns_callable(self, broker):
        """Factory should return a callable handler."""
        handler = create_annotation_stored_handler(broker)
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_processes_events(
        self, broker, valid_annotation_stored_envelope
    ):
        """Created handler should process events correctly."""
        await broker.start()

        handler = create_annotation_stored_handler(broker)
        await handler(valid_annotation_stored_envelope)

        published = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_handler_can_be_used_with_subscribe(
        self, broker, valid_annotation_stored_envelope
    ):
        """Handler should work when registered with broker."""
        await broker.start()

        handler = create_annotation_stored_handler(broker)
        await broker.subscribe(Topics.ANNOTATION_STORED, handler)

        # Publish should trigger the handler
        await broker.publish(Topics.ANNOTATION_STORED, valid_annotation_stored_envelope)

        # Should have published to both topics
        annotation_stored = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        embedding_created = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)

        assert len(annotation_stored) == 1
        assert len(embedding_created) == 1
