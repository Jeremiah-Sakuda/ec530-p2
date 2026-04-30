"""Tests for query service API endpoints."""

import pytest
from fastapi.testclient import TestClient

from shared.repos import VectorRepo, InMemoryDocumentRepo
from services.query.api import app, configure
from services.embedding import mock_embed


@pytest.fixture
def vector_repo():
    """Create a fresh vector repository."""
    return VectorRepo(dim=128)


@pytest.fixture
def document_repo():
    """Create a fresh document repository."""
    return InMemoryDocumentRepo()


@pytest.fixture
async def populated_repos(vector_repo, document_repo):
    """Set up repos with sample data."""
    for i, label in enumerate(["car", "person", "bicycle"]):
        image_id = f"img_{i}"
        object_id = "obj_0"

        vec = mock_embed(image_id, object_id)
        vector_repo.add(image_id, object_id, vec)

        await document_repo.upsert(image_id, {
            "image_id": image_id,
            "objects": [{"object_id": object_id, "label": label}],
            "model_version": "mock_v1",
            "status": "pending",
        })

    return vector_repo, document_repo


@pytest.fixture
def client(vector_repo, document_repo):
    """Create a test client with configured dependencies."""
    configure(vector_repo, document_repo)
    with TestClient(app) as client:
        yield client


class TestQuery:
    """Tests for POST /query."""

    @pytest.mark.asyncio
    async def test_text_query(self, client, populated_repos):
        """Should execute text query."""
        vector_repo, document_repo = populated_repos
        configure(vector_repo, document_repo)

        response = client.post(
            "/query",
            json={"kind": "text", "value": "find cars", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query_kind"] == "text"
        assert data["total_results"] == 3

    @pytest.mark.asyncio
    async def test_image_query(self, client, populated_repos):
        """Should execute image query."""
        vector_repo, document_repo = populated_repos
        configure(vector_repo, document_repo)

        response = client.post(
            "/query",
            json={"kind": "image", "value": "/path/to/query.jpg", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query_kind"] == "image"

    @pytest.mark.asyncio
    async def test_results_have_labels(self, client, populated_repos):
        """Results should include labels."""
        vector_repo, document_repo = populated_repos
        configure(vector_repo, document_repo)

        response = client.post(
            "/query",
            json={"kind": "text", "value": "test", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            assert "label" in result
            assert result["label"] in ["car", "person", "bicycle"]

    @pytest.mark.asyncio
    async def test_respects_top_k(self, client, populated_repos):
        """Should respect top_k limit."""
        vector_repo, document_repo = populated_repos
        configure(vector_repo, document_repo)

        response = client.post(
            "/query",
            json={"kind": "text", "value": "test", "top_k": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 2

    def test_empty_index(self, client, vector_repo, document_repo):
        """Should return empty results for empty index."""
        configure(vector_repo, document_repo)

        response = client.post(
            "/query",
            json={"kind": "text", "value": "test", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total_results"] == 0

    def test_invalid_kind(self, client, vector_repo, document_repo):
        """Should reject invalid query kind."""
        configure(vector_repo, document_repo)

        response = client.post(
            "/query",
            json={"kind": "invalid", "value": "test", "top_k": 5},
        )

        assert response.status_code == 422  # Validation error


class TestQueryText:
    """Tests for POST /query/text."""

    @pytest.mark.asyncio
    async def test_text_query(self, client, populated_repos):
        """Should execute text query via convenience endpoint."""
        vector_repo, document_repo = populated_repos
        configure(vector_repo, document_repo)

        response = client.post(
            "/query/text",
            params={"text": "find cars", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query_kind"] == "text"


class TestQueryImage:
    """Tests for POST /query/image."""

    @pytest.mark.asyncio
    async def test_image_query(self, client, populated_repos):
        """Should execute image query via convenience endpoint."""
        vector_repo, document_repo = populated_repos
        configure(vector_repo, document_repo)

        response = client.post(
            "/query/image",
            params={"image_path": "/path/to/query.jpg", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query_kind"] == "image"


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_check(self, client):
        """Should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "query"
