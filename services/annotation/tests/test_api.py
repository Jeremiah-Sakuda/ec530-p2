"""Tests for annotation service API endpoints."""

import pytest
from fastapi.testclient import TestClient

from shared.broker import InMemoryBroker
from shared.repos import InMemoryDocumentRepo
from shared.events.topics import Topics
from services.annotation import api
from services.annotation.api import app, configure, state


@pytest.fixture
def repo():
    """Create a fresh in-memory document repo."""
    return InMemoryDocumentRepo()


@pytest.fixture
def broker():
    """Create a fresh in-memory broker."""
    return InMemoryBroker()


@pytest.fixture
def client(repo, broker):
    """Create a test client with configured dependencies."""
    configure(repo, broker)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_annotation():
    """Sample annotation document."""
    return {
        "image_id": "img_test123",
        "objects": [
            {"object_id": "obj_0", "label": "car", "bbox": [10, 20, 100, 200], "conf": 0.95},
            {"object_id": "obj_1", "label": "person", "bbox": [50, 60, 150, 250], "conf": 0.88},
        ],
        "model_version": "mock_v1",
        "status": "pending",
        "history": [],
    }


class TestGetAnnotation:
    """Tests for GET /annotations/{image_id}."""

    @pytest.mark.asyncio
    async def test_get_existing_annotation(self, client, repo, sample_annotation):
        """Should return annotation for existing image."""
        await repo.upsert("img_test123", sample_annotation)

        response = client.get("/annotations/img_test123")

        assert response.status_code == 200
        data = response.json()
        assert data["image_id"] == "img_test123"
        assert len(data["objects"]) == 2
        assert data["model_version"] == "mock_v1"
        assert data["status"] == "pending"

    def test_get_nonexistent_annotation(self, client):
        """Should return 404 for nonexistent image."""
        response = client.get("/annotations/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestCorrectAnnotation:
    """Tests for PATCH /annotations/{image_id}."""

    @pytest.mark.asyncio
    async def test_correct_annotation_success(self, client, repo, broker, sample_annotation):
        """Should accept correction and publish event."""
        await repo.upsert("img_test123", sample_annotation)

        response = client.patch(
            "/annotations/img_test123",
            json={
                "patch": {"objects.0.label": "truck"},
                "reviewer": "test_user",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"
        assert "test_user" in data["message"]

        # Check that event was published
        published = broker.get_published_for_topic(Topics.ANNOTATION_CORRECTED)
        assert len(published) == 1
        assert published[0].payload["image_id"] == "img_test123"

    def test_correct_nonexistent_annotation(self, client):
        """Should return 404 for nonexistent image."""
        response = client.patch(
            "/annotations/nonexistent",
            json={
                "patch": {"status": "reviewed"},
                "reviewer": "test_user",
            },
        )

        assert response.status_code == 404

    def test_correct_with_invalid_body(self, client, repo, sample_annotation):
        """Should return 422 for invalid request body."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            repo.upsert("img_test123", sample_annotation)
        )

        response = client.patch(
            "/annotations/img_test123",
            json={"invalid": "body"},
        )

        assert response.status_code == 422


class TestQueryAnnotations:
    """Tests for GET /annotations."""

    @pytest.mark.asyncio
    async def test_query_all(self, client, repo):
        """Should return all annotations when no filters."""
        await repo.upsert("img_1", {"objects": [], "status": "pending"})
        await repo.upsert("img_2", {"objects": [], "status": "reviewed"})

        response = client.get("/annotations")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_query_by_label(self, client, repo):
        """Should filter by object label."""
        await repo.upsert("img_1", {
            "objects": [{"object_id": "obj_0", "label": "car", "bbox": [0, 0, 10, 10], "conf": 0.9}],
            "status": "pending",
        })
        await repo.upsert("img_2", {
            "objects": [{"object_id": "obj_0", "label": "person", "bbox": [0, 0, 10, 10], "conf": 0.8}],
            "status": "pending",
        })

        response = client.get("/annotations?label=car")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["image_id"] == "img_1"

    @pytest.mark.asyncio
    async def test_query_by_status(self, client, repo):
        """Should filter by status."""
        await repo.upsert("img_1", {"objects": [], "status": "pending"})
        await repo.upsert("img_2", {"objects": [], "status": "reviewed"})

        response = client.get("/annotations?status=reviewed")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["image_id"] == "img_2"

    @pytest.mark.asyncio
    async def test_query_by_min_conf(self, client, repo):
        """Should filter by minimum confidence."""
        await repo.upsert("img_1", {
            "objects": [{"object_id": "obj_0", "label": "car", "bbox": [0, 0, 10, 10], "conf": 0.95}],
            "status": "pending",
        })
        await repo.upsert("img_2", {
            "objects": [{"object_id": "obj_0", "label": "car", "bbox": [0, 0, 10, 10], "conf": 0.50}],
            "status": "pending",
        })

        response = client.get("/annotations?min_conf=0.8")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["image_id"] == "img_1"

    @pytest.mark.asyncio
    async def test_query_empty_result(self, client, repo):
        """Should return empty list when no matches."""
        await repo.upsert("img_1", {
            "objects": [{"object_id": "obj_0", "label": "car", "bbox": [0, 0, 10, 10], "conf": 0.9}],
            "status": "pending",
        })

        response = client.get("/annotations?label=nonexistent")

        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_check(self, client):
        """Should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "annotation"
