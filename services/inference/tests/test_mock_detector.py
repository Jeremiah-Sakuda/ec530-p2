"""Tests for mock detector determinism and output validity."""

import pytest
from services.inference.mock_detector import (
    mock_detect,
    DetectedObject,
    MOCK_LABELS,
    MOCK_MODEL_VERSION,
)


class TestMockDetectorDeterminism:
    """Test that mock detector produces deterministic results."""

    def test_same_image_id_produces_same_objects(self):
        """Same image_id should always produce identical results."""
        image_id = "img_abc123"

        result1 = mock_detect(image_id)
        result2 = mock_detect(image_id)

        assert len(result1) == len(result2)
        for obj1, obj2 in zip(result1, result2):
            assert obj1.object_id == obj2.object_id
            assert obj1.label == obj2.label
            assert obj1.bbox == obj2.bbox
            assert obj1.conf == obj2.conf

    def test_different_image_ids_produce_different_results(self):
        """Different image_ids should produce different results."""
        result1 = mock_detect("img_001")
        result2 = mock_detect("img_002")

        # At least one property should differ
        # (very unlikely to be identical by chance)
        labels1 = [obj.label for obj in result1]
        labels2 = [obj.label for obj in result2]

        # Either count differs or labels differ
        if len(result1) == len(result2):
            assert labels1 != labels2 or any(
                obj1.bbox != obj2.bbox for obj1, obj2 in zip(result1, result2)
            )

    def test_determinism_across_multiple_calls(self):
        """Multiple calls with same input should be identical."""
        image_id = "img_determinism_test"

        results = [mock_detect(image_id) for _ in range(10)]

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert len(result) == len(first)
            for obj1, obj2 in zip(first, result):
                assert obj1.to_dict() == obj2.to_dict()


class TestMockDetectorOutput:
    """Test mock detector output validity."""

    def test_returns_list_of_detected_objects(self):
        """Should return a list of DetectedObject instances."""
        result = mock_detect("img_test")

        assert isinstance(result, list)
        assert all(isinstance(obj, DetectedObject) for obj in result)

    def test_returns_one_to_five_objects(self):
        """Should return between 1 and 5 objects."""
        # Test with many different image IDs
        for i in range(100):
            result = mock_detect(f"img_count_test_{i}")
            assert 1 <= len(result) <= 5

    def test_object_ids_are_sequential(self):
        """Object IDs should be obj_0, obj_1, etc."""
        result = mock_detect("img_id_test")

        for i, obj in enumerate(result):
            assert obj.object_id == f"obj_{i}"

    def test_labels_are_from_allowed_set(self):
        """Labels should come from the predefined set."""
        for i in range(50):
            result = mock_detect(f"img_label_test_{i}")
            for obj in result:
                assert obj.label in MOCK_LABELS

    def test_confidence_in_valid_range(self):
        """Confidence should be between 0.5 and 0.99."""
        for i in range(50):
            result = mock_detect(f"img_conf_test_{i}")
            for obj in result:
                assert 0.5 <= obj.conf <= 0.99

    def test_bbox_has_four_elements(self):
        """Bounding box should have exactly 4 elements."""
        result = mock_detect("img_bbox_test")

        for obj in result:
            assert len(obj.bbox) == 4
            assert all(isinstance(coord, int) for coord in obj.bbox)

    def test_bbox_coordinates_are_valid(self):
        """x2 should be > x1, y2 should be > y1."""
        for i in range(50):
            result = mock_detect(f"img_bbox_valid_test_{i}")
            for obj in result:
                x1, y1, x2, y2 = obj.bbox
                assert x2 > x1, "x2 should be greater than x1"
                assert y2 > y1, "y2 should be greater than y1"


class TestDetectedObjectModel:
    """Test DetectedObject dataclass."""

    def test_to_dict_conversion(self):
        """to_dict should return proper dictionary."""
        obj = DetectedObject(
            object_id="obj_0",
            label="car",
            bbox=[10, 20, 100, 200],
            conf=0.95,
        )

        result = obj.to_dict()

        assert result == {
            "object_id": "obj_0",
            "label": "car",
            "bbox": [10, 20, 100, 200],
            "conf": 0.95,
        }

    def test_model_version_constant(self):
        """Model version constant should be defined."""
        assert MOCK_MODEL_VERSION == "mock_v1"
