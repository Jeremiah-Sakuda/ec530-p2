"""Tests for malformed event handling."""

import pytest

from shared.events import create_envelope, EventEnvelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from shared.repos import VectorRepo, InMemoryDocumentRepo

from services.inference.handlers import handle_image_submitted
from services.annotation.handlers import handle_inference_completed
from services.embedding.handlers import handle_annotation_stored
from services.vector_index.handlers import handle_embedding_created
from services.query.handlers import handle_query_submitted


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


class TestMalformedPayloads:
    """Test handling of malformed event payloads."""

    @pytest.mark.asyncio
    async def test_missing_required_fields_does_not_crash(self, broker):
        """Missing required fields should not crash the handler."""
        await broker.start()

        # Missing image_id
        malformed = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"path": "/test.jpg", "source": "test"},  # Missing image_id
        )

        # Should not raise
        await handle_image_submitted(malformed, broker)

        # Should not have published anything
        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 0

    @pytest.mark.asyncio
    async def test_wrong_field_types_does_not_crash(self, broker, document_repo):
        """Wrong field types should not crash the handler."""
        await broker.start()

        # confidence should be float, not string
        malformed = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_test",
                "model_version": "v1",
                "objects": [
                    {
                        "object_id": "obj_0",
                        "label": "car",
                        "bbox": [0, 0, 100, 100],
                        "conf": "high",  # Should be float
                    },
                ],
            },
        )

        # Should not raise
        await handle_inference_completed(malformed, document_repo, broker)

        # Should not have stored anything
        assert await document_repo.count() == 0

    @pytest.mark.asyncio
    async def test_extra_fields_are_tolerated(self, broker):
        """Extra fields should not cause errors."""
        await broker.start()

        # Extra fields that aren't in the schema
        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {
                "image_id": "img_extra",
                "path": "/test.jpg",
                "source": "test",
                "extra_field": "should be ignored",
                "another_extra": 12345,
            },
        )

        await handle_image_submitted(envelope, broker)

        # Should have processed successfully
        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_empty_payload_does_not_crash(self, broker):
        """Empty payload should not crash."""
        await broker.start()

        malformed = create_envelope(Topics.IMAGE_SUBMITTED, {})

        await handle_image_submitted(malformed, broker)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 0

    @pytest.mark.asyncio
    async def test_null_values_do_not_crash(self, broker):
        """Null values should not crash."""
        await broker.start()

        malformed = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": None, "path": None, "source": None},
        )

        await handle_image_submitted(malformed, broker)

        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 0


class TestInvalidEnvelopes:
    """Test handling of invalid envelope structures."""

    @pytest.mark.asyncio
    async def test_valid_events_after_invalid_still_process(self, broker):
        """Valid events should process after invalid ones."""
        await broker.start()

        # First, send invalid event
        invalid = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"bad": "data"},
        )
        await handle_image_submitted(invalid, broker)

        # Then send valid event
        valid = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_valid", "path": "/test.jpg", "source": "test"},
        )
        await handle_image_submitted(valid, broker)

        # Valid event should have been processed
        published = broker.get_published_for_topic(Topics.INFERENCE_COMPLETED)
        assert len(published) == 1
        assert published[0].payload["image_id"] == "img_valid"


class TestMalformedObjectArrays:
    """Test handling of malformed object arrays in payloads."""

    @pytest.mark.asyncio
    async def test_empty_objects_array(self, broker, document_repo):
        """Empty objects array should be handled."""
        await broker.start()

        envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_empty_objs",
                "model_version": "v1",
                "objects": [],
            },
        )

        await handle_inference_completed(envelope, document_repo, broker)

        # Should be stored with empty objects
        doc = await document_repo.get("img_empty_objs")
        assert doc is not None
        assert doc["objects"] == []

    @pytest.mark.asyncio
    async def test_invalid_bbox_length(self, broker, document_repo):
        """Invalid bbox length should not crash."""
        await broker.start()

        envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_bad_bbox",
                "model_version": "v1",
                "objects": [
                    {
                        "object_id": "obj_0",
                        "label": "car",
                        "bbox": [0, 0],  # Should be 4 elements
                        "conf": 0.9,
                    },
                ],
            },
        )

        await handle_inference_completed(envelope, document_repo, broker)

        # Should not have stored (validation fails)
        assert await document_repo.count() == 0

    @pytest.mark.asyncio
    async def test_confidence_out_of_range(self, broker, document_repo):
        """Confidence out of [0, 1] range should not crash."""
        await broker.start()

        envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {
                "image_id": "img_bad_conf",
                "model_version": "v1",
                "objects": [
                    {
                        "object_id": "obj_0",
                        "label": "car",
                        "bbox": [0, 0, 100, 100],
                        "conf": 1.5,  # Should be <= 1.0
                    },
                ],
            },
        )

        await handle_inference_completed(envelope, document_repo, broker)

        # Should not have stored (validation fails)
        assert await document_repo.count() == 0


class TestQueryMalformedInput:
    """Test query handling of malformed input."""

    @pytest.mark.asyncio
    async def test_invalid_query_kind(self, broker, vector_repo, document_repo):
        """Invalid query kind should not crash."""
        await broker.start()

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_bad",
                "kind": "invalid_kind",  # Should be "text" or "image"
                "value": "test",
                "top_k": 5,
            },
        )

        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

        # Should not have published query.completed
        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert len(published) == 0

    @pytest.mark.asyncio
    async def test_missing_query_value(self, broker, vector_repo, document_repo):
        """Missing query value should not crash."""
        await broker.start()

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_missing",
                "kind": "text",
                # Missing "value"
                "top_k": 5,
            },
        )

        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert len(published) == 0
