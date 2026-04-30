"""Tests for subscriber recovery scenarios.

Note: Full subscriber downtime recovery testing requires Redis Streams
for message persistence. These tests verify recovery behavior with the
in-memory broker, demonstrating the pattern without external dependencies.
"""

import pytest

from shared.events import create_envelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from shared.repos import VectorRepo, InMemoryDocumentRepo

from services.inference.handlers import handle_image_submitted
from services.annotation.handlers import handle_inference_completed
from services.embedding.handlers import handle_annotation_stored
from services.vector_index.handlers import handle_embedding_created


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


class TestSubscriberRecovery:
    """Test subscriber recovery after simulated downtime."""

    @pytest.mark.asyncio
    async def test_handler_processes_events_after_restart(
        self, broker, document_repo
    ):
        """Handler should process events normally after being re-registered."""
        await broker.start()

        # Simulate "downtime" by not processing events
        envelope1 = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_before",
                "model_version": "v1",
                "objects": [
                    {"object_id": "obj_0", "label": "car", "bbox": [0, 0, 100, 100], "conf": 0.9},
                ],
            },
        )

        # Process first event
        await handle_inference_completed(envelope1, document_repo, broker)
        assert await document_repo.count() == 1

        # Simulate "restart" - handler is called again for new events
        envelope2 = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_after",
                "model_version": "v1",
                "objects": [
                    {"object_id": "obj_0", "label": "bicycle", "bbox": [10, 10, 50, 50], "conf": 0.85},
                ],
            },
        )

        # Handler should work normally after "restart"
        await handle_inference_completed(envelope2, document_repo, broker)
        assert await document_repo.count() == 2

        # Both documents should exist
        doc1 = await document_repo.get("img_before")
        doc2 = await document_repo.get("img_after")
        assert doc1 is not None
        assert doc2 is not None

    @pytest.mark.asyncio
    async def test_vector_index_recovers_after_rebuild(
        self, broker, vector_repo
    ):
        """Vector index should accept vectors after being cleared and rebuilt."""
        await broker.start()

        # Add initial vectors
        envelope1 = create_envelope(
            Topics.EMBEDDING_CREATED,
            {
                "image_id": "img_initial",
                "embeddings": [
                    {"object_id": "obj_0", "vector_ref": "ref1", "dim": 128},
                ],
            },
        )
        await handle_embedding_created(envelope1, vector_repo)
        initial_count = vector_repo.ntotal

        # Simulate "recovery" - clear and rebuild
        vector_repo.clear()
        assert vector_repo.ntotal == 0

        # Add new vectors after recovery
        envelope2 = create_envelope(
            Topics.EMBEDDING_CREATED,
            {
                "image_id": "img_recovered",
                "embeddings": [
                    {"object_id": "obj_0", "vector_ref": "ref2", "dim": 128},
                    {"object_id": "obj_1", "vector_ref": "ref3", "dim": 128},
                ],
            },
        )
        await handle_embedding_created(envelope2, vector_repo)

        # Index should have new vectors
        assert vector_repo.ntotal == 2

    @pytest.mark.asyncio
    async def test_full_pipeline_recovery(
        self, broker, vector_repo, document_repo
    ):
        """Full pipeline should recover and process events after simulated restart."""
        await broker.start()

        # Process one image through pipeline
        image_envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_1", "path": "/test1.jpg", "source": "test"},
        )
        await handle_image_submitted(image_envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        state_before = {
            "docs": await document_repo.count(),
            "vectors": vector_repo.ntotal,
        }

        # Simulate "restart" - clear published events (simulating broker restart)
        broker.clear_published()

        # Process another image after "restart"
        image_envelope2 = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_2", "path": "/test2.jpg", "source": "test"},
        )
        await handle_image_submitted(image_envelope2, broker)

        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        # Both images should be in the system
        assert await document_repo.count() == state_before["docs"] + 1
        assert vector_repo.ntotal >= state_before["vectors"] + 1


class TestBrokerReconnection:
    """Test broker reconnection scenarios.

    Note: The InMemoryBroker preserves subscriptions across stop/start
    since it's designed for testing. Real broker implementations (Redis)
    would require re-subscription after connection loss.
    """

    @pytest.mark.asyncio
    async def test_broker_continues_after_stop_start(self, broker):
        """Broker should continue working after stop/start cycle."""
        await broker.start()

        received = []

        async def handler(envelope):
            received.append(envelope)

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)

        # Publish event
        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_test", "path": "/test.jpg", "source": "test"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        assert len(received) == 1

        # Stop and start broker
        await broker.stop()
        await broker.start()

        # Existing subscription should still work (in-memory behavior)
        envelope2 = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_test2", "path": "/test2.jpg", "source": "test"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope2)

        # Should have received second event via existing subscription
        assert len(received) == 2
        assert received[1].payload["image_id"] == "img_test2"

    @pytest.mark.asyncio
    async def test_multiple_handlers_continue_after_restart(self, broker):
        """Multiple handlers should continue working after restart."""
        await broker.start()

        handler1_received = []
        handler2_received = []

        async def handler1(envelope):
            handler1_received.append(envelope)

        async def handler2(envelope):
            handler2_received.append(envelope)

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler1)
        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler2)

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_multi", "path": "/test.jpg", "source": "test"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        assert len(handler1_received) == 1
        assert len(handler2_received) == 1

        # Restart
        await broker.stop()
        await broker.start()

        # Publish again - existing handlers should still receive
        envelope2 = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_multi2", "path": "/test2.jpg", "source": "test"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope2)

        # Both handlers should have received both events
        assert len(handler1_received) == 2
        assert len(handler2_received) == 2
