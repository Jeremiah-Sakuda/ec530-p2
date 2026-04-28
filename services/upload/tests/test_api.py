"""Tests for upload service API endpoints."""

import pytest
from fastapi.testclient import TestClient

from services.upload import (
    app,
    set_registry,
    set_broker,
    InMemoryImageRegistry,
)
from shared.broker import InMemoryBroker


class TestUploadAPI:
    """Tests for upload API endpoints."""

    @pytest.fixture
    def registry(self):
        return InMemoryImageRegistry()

    @pytest.fixture
    def broker(self):
        broker = InMemoryBroker()
        return broker

    @pytest.fixture
    def client(self, registry, broker):
        # Inject test dependencies
        set_registry(registry)
        set_broker(broker)
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "upload"

    def test_upload_image_success(self, client):
        response = client.post(
            "/images",
            json={"path": "/images/test.jpg", "source": "camera_A"},
        )
        assert response.status_code == 201
        data = response.json()
        assert "image_id" in data
        assert data["image_id"].startswith("img_")
        assert data["is_duplicate"] is False

    def test_upload_image_duplicate(self, client):
        # First upload
        response1 = client.post(
            "/images",
            json={"path": "/images/test.jpg", "source": "camera_A"},
        )
        image_id = response1.json()["image_id"]

        # Second upload (duplicate)
        response2 = client.post(
            "/images",
            json={"path": "/images/test.jpg", "source": "camera_A"},
        )
        assert response2.status_code == 201
        data = response2.json()
        assert data["image_id"] == image_id
        assert data["is_duplicate"] is True

    def test_upload_missing_path(self, client):
        response = client.post(
            "/images",
            json={"source": "camera_A"},
        )
        assert response.status_code == 422  # Validation error

    def test_upload_missing_source(self, client):
        response = client.post(
            "/images",
            json={"path": "/images/test.jpg"},
        )
        assert response.status_code == 422

    def test_get_image_after_upload(self, client):
        # Upload first
        upload_response = client.post(
            "/images",
            json={"path": "/images/test.jpg", "source": "camera_A"},
        )
        image_id = upload_response.json()["image_id"]

        # Get image
        get_response = client.get(f"/images/{image_id}")
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["image_id"] == image_id
        assert data["path"] == "/images/test.jpg"
        assert data["source"] == "camera_A"
        assert data["status"] == "submitted"

    def test_get_image_not_found(self, client):
        response = client.get("/images/nonexistent")
        assert response.status_code == 404

    def test_multiple_uploads_different_sources(self, client):
        response1 = client.post(
            "/images",
            json={"path": "/images/test.jpg", "source": "camera_A"},
        )
        response2 = client.post(
            "/images",
            json={"path": "/images/test.jpg", "source": "camera_B"},
        )

        assert response1.json()["image_id"] != response2.json()["image_id"]
        assert response1.json()["is_duplicate"] is False
        assert response2.json()["is_duplicate"] is False
