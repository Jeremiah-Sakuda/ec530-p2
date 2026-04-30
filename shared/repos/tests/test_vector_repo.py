"""Tests for FAISS vector repository."""

import os
import tempfile
import pytest
from shared.repos.vector_repo import VectorRepo, SearchResult


@pytest.fixture
def repo():
    """Create a fresh vector repository for each test."""
    return VectorRepo(dim=128)


@pytest.fixture
def sample_vector():
    """Sample 128-dimensional vector."""
    return [float(i) / 128.0 for i in range(128)]


@pytest.fixture
def another_vector():
    """Another sample vector (different from sample_vector)."""
    return [float(128 - i) / 128.0 for i in range(128)]


class TestVectorRepoBasics:
    """Basic CRUD tests for VectorRepo."""

    def test_add_and_get(self, repo, sample_vector):
        """Should store and retrieve a vector."""
        repo.add("img_1", "obj_0", sample_vector)

        result = repo.get("img_1", "obj_0")

        assert result is not None
        assert len(result) == 128
        # Check values are approximately equal (float comparison)
        for v1, v2 in zip(result, sample_vector):
            assert abs(v1 - v2) < 1e-6

    def test_get_nonexistent_returns_none(self, repo):
        """Should return None for nonexistent vector."""
        result = repo.get("nonexistent", "obj_0")
        assert result is None

    def test_has(self, repo, sample_vector):
        """Should correctly check existence."""
        assert repo.has("img_1", "obj_0") is False

        repo.add("img_1", "obj_0", sample_vector)

        assert repo.has("img_1", "obj_0") is True
        assert repo.has("img_1", "obj_1") is False

    def test_ntotal(self, repo, sample_vector, another_vector):
        """Should track vector count correctly."""
        assert repo.ntotal == 0

        repo.add("img_1", "obj_0", sample_vector)
        assert repo.ntotal == 1

        repo.add("img_1", "obj_1", another_vector)
        assert repo.ntotal == 2

    def test_remove(self, repo, sample_vector):
        """Should remove vector and update count."""
        repo.add("img_1", "obj_0", sample_vector)
        assert repo.ntotal == 1

        removed = repo.remove("img_1", "obj_0")

        assert removed is True
        assert repo.ntotal == 0
        assert repo.has("img_1", "obj_0") is False

    def test_remove_nonexistent_returns_false(self, repo):
        """Should return False when removing nonexistent vector."""
        removed = repo.remove("nonexistent", "obj_0")
        assert removed is False

    def test_replace_existing(self, repo, sample_vector, another_vector):
        """Adding with same IDs should replace the vector."""
        repo.add("img_1", "obj_0", sample_vector)
        repo.add("img_1", "obj_0", another_vector)

        assert repo.ntotal == 1

        result = repo.get("img_1", "obj_0")
        # Should be the new vector
        for v1, v2 in zip(result, another_vector):
            assert abs(v1 - v2) < 1e-6

    def test_clear(self, repo, sample_vector, another_vector):
        """Should remove all vectors."""
        repo.add("img_1", "obj_0", sample_vector)
        repo.add("img_2", "obj_0", another_vector)

        repo.clear()

        assert repo.ntotal == 0
        assert repo.has("img_1", "obj_0") is False
        assert repo.has("img_2", "obj_0") is False

    def test_get_all_ids(self, repo, sample_vector, another_vector):
        """Should return all ID pairs."""
        repo.add("img_1", "obj_0", sample_vector)
        repo.add("img_2", "obj_1", another_vector)

        ids = repo.get_all_ids()

        assert len(ids) == 2
        assert ("img_1", "obj_0") in ids
        assert ("img_2", "obj_1") in ids


class TestVectorRepoSearch:
    """Search functionality tests for VectorRepo."""

    def test_search_empty_index(self, repo, sample_vector):
        """Search on empty index should return empty list."""
        results = repo.search(sample_vector, top_k=5)
        assert results == []

    def test_search_returns_results(self, repo, sample_vector, another_vector):
        """Should return search results."""
        repo.add("img_1", "obj_0", sample_vector)
        repo.add("img_2", "obj_0", another_vector)

        results = repo.search(sample_vector, top_k=5)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_sorted_by_distance(self, repo, sample_vector, another_vector):
        """Results should be sorted by distance (ascending)."""
        repo.add("img_1", "obj_0", sample_vector)
        repo.add("img_2", "obj_0", another_vector)

        # Search for sample_vector - it should be the closest match
        results = repo.search(sample_vector, top_k=5)

        assert results[0].image_id == "img_1"
        assert results[0].distance < results[1].distance

    def test_search_respects_top_k(self, repo):
        """Should respect top_k limit."""
        # Add 10 vectors
        for i in range(10):
            vec = [float(i + j) / 100.0 for j in range(128)]
            repo.add(f"img_{i}", "obj_0", vec)

        results = repo.search([0.0] * 128, top_k=3)

        assert len(results) == 3

    def test_search_result_fields(self, repo, sample_vector):
        """SearchResult should have all required fields."""
        repo.add("img_1", "obj_0", sample_vector)

        results = repo.search(sample_vector, top_k=1)

        result = results[0]
        assert result.image_id == "img_1"
        assert result.object_id == "obj_0"
        assert isinstance(result.distance, float)
        assert isinstance(result.score, float)
        assert result.distance >= 0
        assert 0 < result.score <= 1

    def test_search_excludes_deleted(self, repo, sample_vector, another_vector):
        """Search should not return deleted vectors."""
        repo.add("img_1", "obj_0", sample_vector)
        repo.add("img_2", "obj_0", another_vector)

        repo.remove("img_1", "obj_0")

        results = repo.search(sample_vector, top_k=5)

        assert len(results) == 1
        assert results[0].image_id == "img_2"

    def test_search_by_ids(self, repo, sample_vector, another_vector):
        """search_by_ids should exclude the query vector."""
        repo.add("img_1", "obj_0", sample_vector)
        repo.add("img_2", "obj_0", another_vector)

        results = repo.search_by_ids("img_1", "obj_0", top_k=5)

        assert len(results) == 1
        assert results[0].image_id == "img_2"

    def test_search_by_ids_nonexistent(self, repo):
        """search_by_ids with nonexistent IDs should return empty."""
        results = repo.search_by_ids("nonexistent", "obj_0", top_k=5)
        assert results == []


class TestVectorRepoPersistence:
    """Persistence tests for VectorRepo."""

    def test_save_and_load(self, sample_vector, another_vector):
        """Should save and load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index")

            # Create and save
            repo1 = VectorRepo(dim=128, index_path=path)
            repo1.add("img_1", "obj_0", sample_vector)
            repo1.add("img_2", "obj_1", another_vector)
            repo1.save()

            # Load in new instance
            repo2 = VectorRepo(dim=128, index_path=path)
            repo2.load()

            assert repo2.ntotal == 2
            assert repo2.has("img_1", "obj_0")
            assert repo2.has("img_2", "obj_1")

            # Verify vectors are correct
            vec1 = repo2.get("img_1", "obj_0")
            for v1, v2 in zip(vec1, sample_vector):
                assert abs(v1 - v2) < 1e-6

    def test_save_with_custom_path(self, sample_vector):
        """Should save to custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "custom_index")

            repo = VectorRepo(dim=128)
            repo.add("img_1", "obj_0", sample_vector)
            repo.save(path)

            assert os.path.exists(f"{path}.faiss")
            assert os.path.exists(f"{path}.mappings")

    def test_load_without_path_raises(self):
        """Should raise error when loading without path."""
        repo = VectorRepo(dim=128)

        with pytest.raises(ValueError):
            repo.load()


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_to_dict(self):
        """to_dict should return correct structure."""
        result = SearchResult(
            image_id="img_1",
            object_id="obj_0",
            distance=0.5,
            score=0.667,
        )

        d = result.to_dict()

        assert d["image_id"] == "img_1"
        assert d["object_id"] == "obj_0"
        assert d["distance"] == 0.5
        assert d["score"] == 0.667
