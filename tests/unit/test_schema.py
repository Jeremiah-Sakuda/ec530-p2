"""Tests for event schema validation."""

import pytest
from shared.events import (
    Topics,
    ValidationError,
    validate_envelope,
    validate_payload,
    validate_event,
    is_valid_event,
    ImageSubmittedPayload,
    InferenceCompletedPayload,
    DetectedObject,
    AnnotationStoredPayload,
    AnnotationCorrectedPayload,
    EmbeddingCreatedPayload,
    EmbeddingInfo,
    QuerySubmittedPayload,
    QueryCompletedPayload,
    QueryResult,
)


class TestValidateEnvelope:
    """Tests for envelope validation."""

    def test_valid_envelope(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
            "schema_version": 1,
            "payload": {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        }
        envelope = validate_envelope(data)
        assert envelope.topic == Topics.IMAGE_SUBMITTED
        assert envelope.event_id == "evt_001"

    def test_missing_topic_raises(self):
        data = {
            "type": "publish",
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
            "payload": {},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_envelope(data)
        assert "Invalid envelope" in str(exc_info.value)

    def test_missing_event_id_raises(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "timestamp": "2026-04-07T14:33:00Z",
            "payload": {},
        }
        with pytest.raises(ValidationError):
            validate_envelope(data)

    def test_missing_timestamp_raises(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "event_id": "evt_001",
            "payload": {},
        }
        with pytest.raises(ValidationError):
            validate_envelope(data)

    def test_missing_payload_raises(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
        }
        with pytest.raises(ValidationError):
            validate_envelope(data)


class TestImageSubmittedPayload:
    """Tests for image.submitted payload validation."""

    def test_valid_payload(self):
        payload = {"image_id": "img_001", "path": "/images/test.jpg", "source": "camera_A"}
        result = validate_payload(Topics.IMAGE_SUBMITTED, payload)
        assert isinstance(result, ImageSubmittedPayload)
        assert result.image_id == "img_001"

    def test_missing_image_id_raises(self):
        payload = {"path": "/images/test.jpg", "source": "camera_A"}
        with pytest.raises(ValidationError):
            validate_payload(Topics.IMAGE_SUBMITTED, payload)

    def test_missing_path_raises(self):
        payload = {"image_id": "img_001", "source": "camera_A"}
        with pytest.raises(ValidationError):
            validate_payload(Topics.IMAGE_SUBMITTED, payload)

    def test_missing_source_raises(self):
        payload = {"image_id": "img_001", "path": "/images/test.jpg"}
        with pytest.raises(ValidationError):
            validate_payload(Topics.IMAGE_SUBMITTED, payload)

    def test_wrong_type_raises(self):
        payload = {"image_id": 123, "path": "/images/test.jpg", "source": "camera_A"}
        with pytest.raises(ValidationError):
            validate_payload(Topics.IMAGE_SUBMITTED, payload)


class TestInferenceCompletedPayload:
    """Tests for inference.completed payload validation."""

    def test_valid_payload_with_objects(self):
        payload = {
            "image_id": "img_001",
            "model_version": "mock_v1",
            "objects": [
                {"object_id": "obj_0", "label": "car", "bbox": [10, 20, 100, 200], "conf": 0.95}
            ],
        }
        result = validate_payload(Topics.INFERENCE_COMPLETED, payload)
        assert isinstance(result, InferenceCompletedPayload)
        assert len(result.objects) == 1
        assert result.objects[0].label == "car"

    def test_valid_payload_empty_objects(self):
        payload = {"image_id": "img_001", "model_version": "mock_v1", "objects": []}
        result = validate_payload(Topics.INFERENCE_COMPLETED, payload)
        assert len(result.objects) == 0

    def test_invalid_bbox_length_raises(self):
        payload = {
            "image_id": "img_001",
            "model_version": "mock_v1",
            "objects": [
                {"object_id": "obj_0", "label": "car", "bbox": [10, 20, 100], "conf": 0.95}
            ],
        }
        with pytest.raises(ValidationError):
            validate_payload(Topics.INFERENCE_COMPLETED, payload)

    def test_invalid_confidence_raises(self):
        payload = {
            "image_id": "img_001",
            "model_version": "mock_v1",
            "objects": [
                {"object_id": "obj_0", "label": "car", "bbox": [10, 20, 100, 200], "conf": 1.5}
            ],
        }
        with pytest.raises(ValidationError):
            validate_payload(Topics.INFERENCE_COMPLETED, payload)

    def test_negative_confidence_raises(self):
        payload = {
            "image_id": "img_001",
            "model_version": "mock_v1",
            "objects": [
                {"object_id": "obj_0", "label": "car", "bbox": [10, 20, 100, 200], "conf": -0.1}
            ],
        }
        with pytest.raises(ValidationError):
            validate_payload(Topics.INFERENCE_COMPLETED, payload)


class TestAnnotationStoredPayload:
    """Tests for annotation.stored payload validation."""

    def test_valid_payload(self):
        payload = {"image_id": "img_001", "object_ids": ["obj_0", "obj_1"], "model_version": "mock_v1"}
        result = validate_payload(Topics.ANNOTATION_STORED, payload)
        assert isinstance(result, AnnotationStoredPayload)
        assert len(result.object_ids) == 2


class TestAnnotationCorrectedPayload:
    """Tests for annotation.corrected payload validation."""

    def test_valid_payload(self):
        payload = {
            "image_id": "img_001",
            "patch": {"objects.0.label": "truck"},
            "reviewer": "cli_user",
        }
        result = validate_payload(Topics.ANNOTATION_CORRECTED, payload)
        assert isinstance(result, AnnotationCorrectedPayload)
        assert result.reviewer == "cli_user"


class TestEmbeddingCreatedPayload:
    """Tests for embedding.created payload validation."""

    def test_valid_payload(self):
        payload = {
            "image_id": "img_001",
            "embeddings": [
                {"object_id": "obj_0", "vector_ref": "vec_store/img_001_obj_0", "dim": 128}
            ],
        }
        result = validate_payload(Topics.EMBEDDING_CREATED, payload)
        assert isinstance(result, EmbeddingCreatedPayload)
        assert result.embeddings[0].dim == 128


class TestQuerySubmittedPayload:
    """Tests for query.submitted payload validation."""

    def test_valid_text_query(self):
        payload = {"query_id": "q_001", "kind": "text", "value": "red car at night", "top_k": 5}
        result = validate_payload(Topics.QUERY_SUBMITTED, payload)
        assert isinstance(result, QuerySubmittedPayload)
        assert result.kind == "text"

    def test_valid_image_query(self):
        payload = {"query_id": "q_002", "kind": "image", "value": "/images/query.jpg", "top_k": 10}
        result = validate_payload(Topics.QUERY_SUBMITTED, payload)
        assert result.kind == "image"

    def test_invalid_kind_raises(self):
        payload = {"query_id": "q_001", "kind": "audio", "value": "test", "top_k": 5}
        with pytest.raises(ValidationError):
            validate_payload(Topics.QUERY_SUBMITTED, payload)

    def test_top_k_too_large_raises(self):
        payload = {"query_id": "q_001", "kind": "text", "value": "test", "top_k": 200}
        with pytest.raises(ValidationError):
            validate_payload(Topics.QUERY_SUBMITTED, payload)

    def test_top_k_zero_raises(self):
        payload = {"query_id": "q_001", "kind": "text", "value": "test", "top_k": 0}
        with pytest.raises(ValidationError):
            validate_payload(Topics.QUERY_SUBMITTED, payload)


class TestQueryCompletedPayload:
    """Tests for query.completed payload validation."""

    def test_valid_payload(self):
        payload = {
            "query_id": "q_001",
            "results": [{"image_id": "img_001", "object_id": "obj_0", "score": 0.95}],
        }
        result = validate_payload(Topics.QUERY_COMPLETED, payload)
        assert isinstance(result, QueryCompletedPayload)
        assert len(result.results) == 1


class TestValidateEvent:
    """Tests for full event validation."""

    def test_valid_event(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
            "schema_version": 1,
            "payload": {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        }
        envelope, payload = validate_event(data)
        assert envelope.event_id == "evt_001"
        assert isinstance(payload, ImageSubmittedPayload)

    def test_invalid_envelope_raises(self):
        data = {"topic": Topics.IMAGE_SUBMITTED, "payload": {}}
        with pytest.raises(ValidationError):
            validate_event(data)

    def test_invalid_payload_raises(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
            "payload": {"wrong_field": "value"},
        }
        with pytest.raises(ValidationError):
            validate_event(data)

    def test_unknown_topic_raises(self):
        data = {
            "type": "publish",
            "topic": "unknown.topic",
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
            "payload": {},
        }
        with pytest.raises(ValidationError):
            validate_event(data)


class TestIsValidEvent:
    """Tests for is_valid_event helper."""

    def test_returns_true_for_valid(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
            "schema_version": 1,
            "payload": {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        }
        assert is_valid_event(data) is True

    def test_returns_false_for_invalid_envelope(self):
        data = {"topic": Topics.IMAGE_SUBMITTED}
        assert is_valid_event(data) is False

    def test_returns_false_for_invalid_payload(self):
        data = {
            "type": "publish",
            "topic": Topics.IMAGE_SUBMITTED,
            "event_id": "evt_001",
            "timestamp": "2026-04-07T14:33:00Z",
            "payload": {},
        }
        assert is_valid_event(data) is False
