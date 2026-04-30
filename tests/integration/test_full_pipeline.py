"""Integration tests for the full image processing pipeline.

These tests verify the complete flow from image submission to query results,
using in-memory implementations for all dependencies.
"""

import pytest

from shared.events import create_envelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from shared.repos import VectorRepo, InMemoryDocumentRepo

from services.inference import mock_detect, MOCK_MODEL_VERSION
from services.inference.handlers import handle_image_submitted
from services.annotation.handlers import handle_inference_completed
from services.embedding.handlers import handle_annotation_stored
from services.vector_index.handlers import handle_embedding_created
from services.query.handlers import execute_query

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


class TestFullPipeline:
    """Test the complete image processing pipeline."""

    @pytest.mark.asyncio
    async def test_image_to_query_flow(self, broker, vector_repo, document_repo):
        """
        Test full flow: image submission -> inference -> annotation -> embedding -> query.

        This simulates the complete pipeline using in-memory implementations.
        """
        await broker.start()

        # Step 1: Create image.submitted event
        image_envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {
                "image_id": "img_pipeline_test",
                "path": "/images/test.jpg",
                "source": "integration_test",
            },
        )

        # Step 2: Handle image.submitted -> produces inference.completed
        await handle_image_submitted(image_envelope, broker)

        inference_events = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(inference_events) == 1

        # Step 3: Handle inference.completed -> produces annotation.stored
        await handle_inference_completed(inference_events[0], document_repo, broker)

        annotation_events = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        assert len(annotation_events) == 1

        # Verify annotation was stored
        annotation = await document_repo.get("img_pipeline_test")
        assert annotation is not None
        assert len(annotation["objects"]) >= 1

        # Step 4: Handle annotation.stored -> produces embedding.created
        await handle_annotation_stored(annotation_events[0], broker)

        embedding_events = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        assert len(embedding_events) == 1

        # Step 5: Handle embedding.created -> adds to vector index
        await handle_embedding_created(embedding_events[0], vector_repo)

        # Verify vectors were added
        assert vector_repo.ntotal >= 1

        # Step 6: Execute a query and verify results
        results = await execute_query(
            kind="text",
            value="find objects",
            top_k=5,
            vector_repo=vector_repo,
            document_repo=document_repo,
        )

        assert len(results) >= 1
        assert results[0]["image_id"] == "img_pipeline_test"
        assert "label" in results[0]

    @pytest.mark.asyncio
    async def test_multiple_images_pipeline(self, broker, vector_repo, document_repo):
        """Test pipeline with multiple images."""
        await broker.start()

        # Process 3 images
        for i in range(3):
            # Submit image
            image_envelope = create_envelope(
                Topics.IMAGE_SUBMITTED,
                {
                    "image_id": f"img_multi_{i}",
                    "path": f"/images/test_{i}.jpg",
                    "source": "integration_test",
                },
            )

            await handle_image_submitted(image_envelope, broker)

        # Process all inference events
        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        # Process all annotation events
        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        # Process all embedding events
        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        # Verify all images have annotations
        assert await document_repo.count() == 3

        # Verify vectors were added (at least 1 per image)
        assert vector_repo.ntotal >= 3

        # Query should return results
        results = await execute_query(
            kind="text",
            value="test query",
            top_k=10,
            vector_repo=vector_repo,
            document_repo=document_repo,
        )

        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_deterministic_pipeline(self, broker, vector_repo, document_repo):
        """Same input should produce same results."""
        await broker.start()

        image_id = "img_determinism_test"

        # Run pipeline
        image_envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": image_id, "path": "/test.jpg", "source": "test"},
        )

        await handle_image_submitted(image_envelope, broker)

        inference_events = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        await handle_inference_completed(inference_events[0], document_repo, broker)

        annotation_events = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        await handle_annotation_stored(annotation_events[0], broker)

        embedding_events = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        await handle_embedding_created(embedding_events[0], vector_repo)

        # Get results
        annotation1 = await document_repo.get(image_id)
        vector1 = vector_repo.get(image_id, "obj_0")

        # Clear and run again
        await document_repo.clear()
        vector_repo.clear()
        broker.clear_published()

        # Same input
        await handle_image_submitted(image_envelope, broker)

        inference_events = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        await handle_inference_completed(inference_events[0], document_repo, broker)

        annotation_events = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        await handle_annotation_stored(annotation_events[0], broker)

        embedding_events = broker.get_published_for_topic(Topics.EMBEDDING_CREATED)
        await handle_embedding_created(embedding_events[0], vector_repo)

        # Get results again
        annotation2 = await document_repo.get(image_id)
        vector2 = vector_repo.get(image_id, "obj_0")

        # Results should be identical
        assert annotation1["objects"] == annotation2["objects"]
        assert vector1 == vector2


class TestCorrectionFlow:
    """Test annotation correction flow."""

    @pytest.mark.asyncio
    async def test_correction_triggers_reindexing(self, broker, vector_repo, document_repo):
        """Correction should trigger re-embedding."""
        await broker.start()

        # First, process an image through the pipeline
        image_id = "img_correction_test"

        image_envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": image_id, "path": "/test.jpg", "source": "test"},
        )

        await handle_image_submitted(image_envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        initial_annotation_stored = len(
            broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        )

        # Process embeddings
        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        initial_vector_count = vector_repo.ntotal

        # Now apply a correction
        from services.annotation.handlers import handle_annotation_corrected

        correction_envelope = create_envelope(
            Topics.ANNOTATION_CORRECTED,
            {
                "image_id": image_id,
                "patch": {"objects.0.label": "corrected_label"},
                "reviewer": "test_reviewer",
            },
        )

        await handle_annotation_corrected(correction_envelope, document_repo, broker)

        # Correction should have published annotation.stored again
        final_annotation_stored = len(
            broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        )
        assert final_annotation_stored > initial_annotation_stored

        # Verify correction was applied
        annotation = await document_repo.get(image_id)
        assert annotation["status"] == "corrected"
        assert len(annotation["history"]) == 1

    @pytest.mark.asyncio
    async def test_query_returns_corrected_label(self, broker, vector_repo, document_repo):
        """Query should return corrected labels after correction."""
        await broker.start()

        # Process image
        image_id = "img_label_test"

        image_envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": image_id, "path": "/test.jpg", "source": "test"},
        )

        await handle_image_submitted(image_envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        # Apply correction
        from services.annotation.handlers import handle_annotation_corrected

        correction_envelope = create_envelope(
            Topics.ANNOTATION_CORRECTED,
            {
                "image_id": image_id,
                "patch": {"objects.0.label": "CORRECTED"},
                "reviewer": "tester",
            },
        )

        await handle_annotation_corrected(correction_envelope, document_repo, broker)

        # Query should return the corrected label
        results = await execute_query(
            kind="text",
            value="test",
            top_k=5,
            vector_repo=vector_repo,
            document_repo=document_repo,
        )

        # Find result for our image
        our_result = next(
            (r for r in results if r["image_id"] == image_id and r["object_id"] == "obj_0"),
            None,
        )

        assert our_result is not None
        assert our_result["label"] == "CORRECTED"


class TestEventGeneratorIntegration:
    """Test using event generator with the pipeline."""

    @pytest.mark.asyncio
    async def test_generated_events_through_pipeline(
        self, broker, vector_repo, document_repo
    ):
        """Generated events should process correctly through the pipeline."""
        await broker.start()

        generator = EventGenerator(seed=42)
        events = generator.generate_image_submitted(count=5)

        # Process all generated events
        for event in events:
            await handle_image_submitted(event, broker)

        # Process through pipeline
        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        # Verify all images were processed
        assert await document_repo.count() == 5
        assert vector_repo.ntotal >= 5

    @pytest.mark.asyncio
    async def test_replay_produces_identical_state(
        self, broker, vector_repo, document_repo
    ):
        """Replaying same events should produce identical state."""
        await broker.start()

        generator = EventGenerator(seed=12345)
        events = generator.generate_image_submitted(count=3)

        # First run
        for event in events:
            await handle_image_submitted(event, broker)

        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        # Capture state
        state1_count = await document_repo.count()
        state1_vectors = vector_repo.ntotal

        # Clear and replay
        await document_repo.clear()
        vector_repo.clear()
        broker.clear_published()

        # Regenerate with same seed
        generator2 = EventGenerator(seed=12345)
        events2 = generator2.generate_image_submitted(count=3)

        # Second run
        for event in events2:
            await handle_image_submitted(event, broker)

        for envelope in broker.get_published_for_topic(Topics.INFERENCE_COMPLETED):
            await handle_inference_completed(envelope, document_repo, broker)

        for envelope in broker.get_published_for_topic(Topics.ANNOTATION_STORED):
            await handle_annotation_stored(envelope, broker)

        for envelope in broker.get_published_for_topic(Topics.EMBEDDING_CREATED):
            await handle_embedding_created(envelope, vector_repo)

        # Verify identical state
        assert await document_repo.count() == state1_count
        assert vector_repo.ntotal == state1_vectors
