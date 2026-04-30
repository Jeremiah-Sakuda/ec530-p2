"""Tests for annotation service event handlers."""

import pytest
from shared.events import EventEnvelope, create_envelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from shared.repos import InMemoryDocumentRepo
from services.annotation.handlers import (
    handle_inference_completed,
    handle_annotation_corrected,
    create_inference_completed_handler,
    create_annotation_corrected_handler,
    apply_patch,
    build_annotation_document,
)
from shared.events.schema import InferenceCompletedPayload, DetectedObject


@pytest.fixture
def broker():
    """Create a fresh in-memory broker for each test."""
    return InMemoryBroker()


@pytest.fixture
def repo():
    """Create a fresh in-memory document repo for each test."""
    return InMemoryDocumentRepo()


@pytest.fixture
def valid_inference_completed_envelope():
    """Create a valid inference.completed event envelope."""
    return create_envelope(
        Topics.INFERENCE_COMPLETED,
        {
            "image_id": "img_test123",
            "model_version": "mock_v1",
            "objects": [
                {
                    "object_id": "obj_0",
                    "label": "car",
                    "bbox": [10, 20, 100, 200],
                    "conf": 0.95,
                },
                {
                    "object_id": "obj_1",
                    "label": "person",
                    "bbox": [50, 60, 150, 250],
                    "conf": 0.88,
                },
            ],
        },
    )


class TestHandleInferenceCompleted:
    """Tests for handle_inference_completed handler."""

    @pytest.mark.asyncio
    async def test_stores_annotation_document(
        self, broker, repo, valid_inference_completed_envelope
    ):
        """Handler should store annotation in repository."""
        await broker.start()

        await handle_inference_completed(valid_inference_completed_envelope, repo, broker)

        doc = await repo.get("img_test123")
        assert doc is not None
        assert doc["image_id"] == "img_test123"
        assert len(doc["objects"]) == 2

    @pytest.mark.asyncio
    async def test_publishes_annotation_stored_event(
        self, broker, repo, valid_inference_completed_envelope
    ):
        """Handler should publish annotation.stored event."""
        await broker.start()

        await handle_inference_completed(valid_inference_completed_envelope, repo, broker)

        published = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        assert len(published) == 1
        assert published[0].payload["image_id"] == "img_test123"
        assert published[0].payload["object_ids"] == ["obj_0", "obj_1"]

    @pytest.mark.asyncio
    async def test_idempotency_prevents_duplicate_processing(
        self, broker, repo, valid_inference_completed_envelope
    ):
        """Same event should not be processed twice."""
        await broker.start()

        # Process same event twice
        await handle_inference_completed(valid_inference_completed_envelope, repo, broker)
        await handle_inference_completed(valid_inference_completed_envelope, repo, broker)

        # Should only have one published event
        published = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_records_processed_event(
        self, broker, repo, valid_inference_completed_envelope
    ):
        """Handler should record event as processed."""
        await broker.start()

        await handle_inference_completed(valid_inference_completed_envelope, repo, broker)

        event_id = valid_inference_completed_envelope.event_id
        assert await repo.has_processed_event("img_test123", event_id) is True

    @pytest.mark.asyncio
    async def test_invalid_payload_does_not_crash(self, broker, repo):
        """Handler should gracefully handle invalid payloads."""
        await broker.start()

        invalid_envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {"invalid_field": "value"},
        )

        # Should not raise
        await handle_inference_completed(invalid_envelope, repo, broker)

        # Should not store anything
        assert await repo.count() == 0


class TestHandleAnnotationCorrected:
    """Tests for handle_annotation_corrected handler."""

    @pytest.fixture
    def existing_annotation(self):
        """Pre-existing annotation document."""
        return {
            "image_id": "img_existing",
            "objects": [
                {"object_id": "obj_0", "label": "car", "bbox": [10, 20, 100, 200], "conf": 0.95},
            ],
            "model_version": "mock_v1",
            "status": "pending",
            "history": [],
        }

    @pytest.fixture
    def correction_envelope(self):
        """Create a correction event envelope."""
        return create_envelope(
            Topics.ANNOTATION_CORRECTED,
            {
                "image_id": "img_existing",
                "patch": {"objects.0.label": "truck"},
                "reviewer": "test_user",
            },
        )

    @pytest.mark.asyncio
    async def test_applies_correction_patch(
        self, broker, repo, existing_annotation, correction_envelope
    ):
        """Handler should apply the correction patch."""
        await broker.start()
        await repo.upsert("img_existing", existing_annotation)

        await handle_annotation_corrected(correction_envelope, repo, broker)

        doc = await repo.get("img_existing")
        assert doc["objects"][0]["label"] == "truck"

    @pytest.mark.asyncio
    async def test_updates_status_to_corrected(
        self, broker, repo, existing_annotation, correction_envelope
    ):
        """Handler should update status to corrected."""
        await broker.start()
        await repo.upsert("img_existing", existing_annotation)

        await handle_annotation_corrected(correction_envelope, repo, broker)

        doc = await repo.get("img_existing")
        assert doc["status"] == "corrected"

    @pytest.mark.asyncio
    async def test_adds_history_entry(
        self, broker, repo, existing_annotation, correction_envelope
    ):
        """Handler should add correction to history."""
        await broker.start()
        await repo.upsert("img_existing", existing_annotation)

        await handle_annotation_corrected(correction_envelope, repo, broker)

        doc = await repo.get("img_existing")
        assert len(doc["history"]) == 1
        assert doc["history"][0]["reviewer"] == "test_user"
        assert doc["history"][0]["patch"] == {"objects.0.label": "truck"}

    @pytest.mark.asyncio
    async def test_republishes_annotation_stored(
        self, broker, repo, existing_annotation, correction_envelope
    ):
        """Handler should re-publish annotation.stored for re-embedding."""
        await broker.start()
        await repo.upsert("img_existing", existing_annotation)

        await handle_annotation_corrected(correction_envelope, repo, broker)

        published = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        assert len(published) == 1
        assert published[0].payload["image_id"] == "img_existing"

    @pytest.mark.asyncio
    async def test_idempotency_prevents_duplicate_correction(
        self, broker, repo, existing_annotation, correction_envelope
    ):
        """Same correction should not be applied twice."""
        await broker.start()
        await repo.upsert("img_existing", existing_annotation)

        # Apply same correction twice
        await handle_annotation_corrected(correction_envelope, repo, broker)
        await handle_annotation_corrected(correction_envelope, repo, broker)

        # Should only have one history entry
        doc = await repo.get("img_existing")
        assert len(doc["history"]) == 1

    @pytest.mark.asyncio
    async def test_missing_annotation_does_not_crash(self, broker, repo, correction_envelope):
        """Handler should handle missing annotations gracefully."""
        await broker.start()

        # No annotation exists
        await handle_annotation_corrected(correction_envelope, repo, broker)

        # Should not publish anything
        published = broker.get_published_for_topic(Topics.ANNOTATION_STORED)
        assert len(published) == 0


class TestApplyPatch:
    """Tests for apply_patch utility function."""

    def test_simple_update(self):
        """Should update a simple field."""
        doc = {"status": "pending"}
        result = apply_patch(doc, {"status": "reviewed"})
        assert result["status"] == "reviewed"

    def test_nested_update_with_dot_notation(self):
        """Should update nested fields using dot notation."""
        doc = {"objects": [{"label": "car"}, {"label": "person"}]}
        result = apply_patch(doc, {"objects.0.label": "truck"})
        assert result["objects"][0]["label"] == "truck"
        assert result["objects"][1]["label"] == "person"

    def test_deeply_nested_update(self):
        """Should update deeply nested fields."""
        doc = {"data": {"nested": {"value": 10}}}
        result = apply_patch(doc, {"data.nested.value": 20})
        assert result["data"]["nested"]["value"] == 20

    def test_multiple_patches(self):
        """Should apply multiple patches."""
        doc = {"a": 1, "b": {"c": 2}}
        result = apply_patch(doc, {"a": 10, "b.c": 20})
        assert result["a"] == 10
        assert result["b"]["c"] == 20


class TestBuildAnnotationDocument:
    """Tests for build_annotation_document function."""

    def test_builds_correct_structure(self):
        """Should build document with correct structure."""
        payload = InferenceCompletedPayload(
            image_id="img_test",
            model_version="v1",
            objects=[
                DetectedObject(
                    object_id="obj_0",
                    label="car",
                    bbox=[10, 20, 100, 200],
                    conf=0.95,
                )
            ],
        )

        doc = build_annotation_document(payload)

        assert doc["image_id"] == "img_test"
        assert doc["model_version"] == "v1"
        assert doc["status"] == "pending"
        assert len(doc["objects"]) == 1
        assert doc["history"] == []


class TestHandlerFactories:
    """Tests for handler factory functions."""

    @pytest.mark.asyncio
    async def test_inference_completed_handler_factory(
        self, broker, repo, valid_inference_completed_envelope
    ):
        """Factory should create working handler."""
        await broker.start()

        handler = create_inference_completed_handler(repo, broker)
        await handler(valid_inference_completed_envelope)

        assert await repo.count() == 1

    @pytest.mark.asyncio
    async def test_annotation_corrected_handler_factory(self, broker, repo):
        """Factory should create working handler."""
        await broker.start()
        await repo.upsert("img_test", {
            "objects": [{"object_id": "obj_0", "label": "car"}],
            "status": "pending",
        })

        handler = create_annotation_corrected_handler(repo, broker)
        envelope = create_envelope(
            Topics.ANNOTATION_CORRECTED,
            {"image_id": "img_test", "patch": {"status": "reviewed"}, "reviewer": "tester"},
        )
        await handler(envelope)

        doc = await repo.get("img_test")
        assert doc["status"] == "corrected"
