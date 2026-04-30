"""Tests for vector index service API endpoints."""

import pytest
from fastapi.testclient import TestClient

from shared.repos import VectorRepo
from services.vector_index.api import app, configure, state


@pytest.fixture
def vector_repo():
    """Create a fresh vector repository."""
    return VectorRepo(dim=128)


@pytest.fixture
def client(vector_repo):
    """Create a test client with configured dependencies."""
    configure(vector_repo)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_vector():
    """Sample 128-dimensional vector."""
    return [float(i) / 128.0 for i in range(128)]


@pytest.fixture
def populated_repo(vector_repo, sample_vector):
    """Vector repo with some data."""
    vector_repo.add("img_1", "obj_0", sample_vector)
    vector_repo.add("img_1", "obj_1", [float(128 - i) / 128.0 for i in range(128)])
    vector_repo.add("img_2", "obj_0", [float(i + 64) / 192.0 for i in range(128)])
    return vector_repo


class TestSearch:
    """Tests for POST /search."""

    def test_search_empty_index(self, client, sample_vector):
        """Search on empty index should return empty results."""
        response = client.post(
            "/search",
            json={"vector": sample_vector, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["index_size"] == 0

    def test_search_returns_results(self, client, populated_repo, sample_vector):
        """Search should return results."""
        configure(populated_repo)

        response = client.post(
            "/search",
            json={"vector": sample_vector, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["index_size"] == 3

    def test_search_sorted_by_similarity(self, client, populated_repo, sample_vector):
        """Results should be sorted by similarity."""
        configure(populated_repo)

        response = client.post(
            "/search",
            json={"vector": sample_vector, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        # First result should be img_1/obj_0 (exact match)
        assert data["results"][0]["image_id"] == "img_1"
        assert data["results"][0]["object_id"] == "obj_0"

        # Scores should be descending
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, client, populated_repo, sample_vector):
        """Should respect top_k limit."""
        configure(populated_repo)

        response = client.post(
            "/search",
            json={"vector": sample_vector, "top_k": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_search_wrong_dimension(self, client, vector_repo):
        """Should reject vectors with wrong dimension."""
        configure(vector_repo)

        response = client.post(
            "/search",
            json={"vector": [1.0, 2.0, 3.0], "top_k": 5},  # Wrong dimension
        )

        assert response.status_code == 400
        assert "dimension mismatch" in response.json()["detail"].lower()


class TestSearchByIds:
    """Tests for POST /search/by-ids."""

    def test_search_by_ids(self, client, populated_repo):
        """Should find similar vectors excluding query."""
        configure(populated_repo)

        response = client.post(
            "/search/by-ids",
            json={"image_id": "img_1", "object_id": "obj_0", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        # Should not include the query vector itself
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert not (r["image_id"] == "img_1" and r["object_id"] == "obj_0")

    def test_search_by_ids_not_found(self, client, vector_repo):
        """Should return 404 for nonexistent vector."""
        configure(vector_repo)

        response = client.post(
            "/search/by-ids",
            json={"image_id": "nonexistent", "object_id": "obj_0", "top_k": 5},
        )

        assert response.status_code == 404


class TestStats:
    """Tests for GET /stats."""

    def test_stats_empty_index(self, client, vector_repo):
        """Should return stats for empty index."""
        configure(vector_repo)

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["ntotal"] == 0
        assert data["dim"] == 128

    def test_stats_populated_index(self, client, populated_repo):
        """Should return stats for populated index."""
        configure(populated_repo)

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["ntotal"] == 3
        assert data["dim"] == 128


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_check(self, client):
        """Should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "vector_index"
