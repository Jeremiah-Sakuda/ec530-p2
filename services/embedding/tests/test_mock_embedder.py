"""Tests for mock embedder determinism and output validity."""

import pytest
from services.embedding.mock_embedder import (
    mock_embed,
    mock_embed_text,
    mock_embed_image_query,
    embed_objects,
    EmbeddingResult,
    EMBEDDING_DIM,
)


class TestMockEmbedDeterminism:
    """Test that mock embedder produces deterministic results."""

    def test_same_inputs_produce_same_vector(self):
        """Same image_id and object_id should produce identical vectors."""
        vector1 = mock_embed("img_abc", "obj_0")
        vector2 = mock_embed("img_abc", "obj_0")

        assert vector1 == vector2

    def test_different_object_ids_produce_different_vectors(self):
        """Different object_ids should produce different vectors."""
        vector1 = mock_embed("img_abc", "obj_0")
        vector2 = mock_embed("img_abc", "obj_1")

        assert vector1 != vector2

    def test_different_image_ids_produce_different_vectors(self):
        """Different image_ids should produce different vectors."""
        vector1 = mock_embed("img_001", "obj_0")
        vector2 = mock_embed("img_002", "obj_0")

        assert vector1 != vector2

    def test_determinism_across_multiple_calls(self):
        """Multiple calls should produce identical results."""
        results = [mock_embed("img_test", "obj_test") for _ in range(10)]

        for result in results[1:]:
            assert result == results[0]


class TestMockEmbedOutput:
    """Test mock embedder output validity."""

    def test_returns_list_of_floats(self):
        """Should return a list of floats."""
        vector = mock_embed("img_test", "obj_0")

        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_default_dimension_is_128(self):
        """Should return 128-dimensional vector by default."""
        vector = mock_embed("img_test", "obj_0")

        assert len(vector) == 128
        assert len(vector) == EMBEDDING_DIM

    def test_custom_dimension(self):
        """Should support custom dimensions."""
        vector = mock_embed("img_test", "obj_0", dim=64)

        assert len(vector) == 64

    def test_values_are_gaussian_distributed(self):
        """Values should be approximately normally distributed."""
        vector = mock_embed("img_test", "obj_0")

        # Check that values are roughly centered around 0
        mean = sum(vector) / len(vector)
        assert -1.0 < mean < 1.0

        # Check that there's variance
        min_val = min(vector)
        max_val = max(vector)
        assert max_val - min_val > 1.0


class TestMockEmbedText:
    """Tests for text embedding function."""

    def test_same_text_produces_same_vector(self):
        """Same text should produce identical vectors."""
        vector1 = mock_embed_text("hello world")
        vector2 = mock_embed_text("hello world")

        assert vector1 == vector2

    def test_different_text_produces_different_vectors(self):
        """Different text should produce different vectors."""
        vector1 = mock_embed_text("hello")
        vector2 = mock_embed_text("world")

        assert vector1 != vector2

    def test_default_dimension(self):
        """Should return 128-dimensional vector by default."""
        vector = mock_embed_text("test query")

        assert len(vector) == 128


class TestMockEmbedImageQuery:
    """Tests for image query embedding function."""

    def test_same_path_produces_same_vector(self):
        """Same image path should produce identical vectors."""
        vector1 = mock_embed_image_query("/images/test.jpg")
        vector2 = mock_embed_image_query("/images/test.jpg")

        assert vector1 == vector2

    def test_different_paths_produce_different_vectors(self):
        """Different paths should produce different vectors."""
        vector1 = mock_embed_image_query("/images/a.jpg")
        vector2 = mock_embed_image_query("/images/b.jpg")

        assert vector1 != vector2


class TestEmbedObjects:
    """Tests for embed_objects batch function."""

    def test_embeds_multiple_objects(self):
        """Should return embeddings for all objects."""
        results = embed_objects("img_test", ["obj_0", "obj_1", "obj_2"])

        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)

    def test_embedding_results_have_correct_ids(self):
        """Each result should have correct image_id and object_id."""
        results = embed_objects("img_test", ["obj_0", "obj_1"])

        assert results[0].image_id == "img_test"
        assert results[0].object_id == "obj_0"
        assert results[1].image_id == "img_test"
        assert results[1].object_id == "obj_1"

    def test_embedding_results_have_vectors(self):
        """Each result should have a vector."""
        results = embed_objects("img_test", ["obj_0"])

        assert len(results[0].vector) == EMBEDDING_DIM

    def test_deterministic_results(self):
        """Same inputs should produce same results."""
        results1 = embed_objects("img_test", ["obj_0", "obj_1"])
        results2 = embed_objects("img_test", ["obj_0", "obj_1"])

        for r1, r2 in zip(results1, results2):
            assert r1.vector == r2.vector


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_to_dict_conversion(self):
        """to_dict should return proper dictionary."""
        result = EmbeddingResult(
            object_id="obj_0",
            image_id="img_test",
            vector=[0.1, 0.2, 0.3],
            dim=3,
        )

        d = result.to_dict()

        assert d["object_id"] == "obj_0"
        assert d["vector_ref"] == "vec_store/img_test_obj_0"
        assert d["dim"] == 3

    def test_default_dimension(self):
        """Default dimension should be EMBEDDING_DIM."""
        result = EmbeddingResult(
            object_id="obj_0",
            image_id="img_test",
            vector=[],
        )

        assert result.dim == EMBEDDING_DIM
