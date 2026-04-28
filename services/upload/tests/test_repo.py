"""Tests for upload service repository."""

import pytest
from services.upload import (
    ImageRecord,
    InMemoryImageRegistry,
    compute_content_hash,
    generate_image_id,
)


class TestComputeContentHash:
    """Tests for content hash computation."""

    def test_same_path_produces_same_hash(self):
        hash1 = compute_content_hash("/test/image.jpg")
        hash2 = compute_content_hash("/test/image.jpg")
        assert hash1 == hash2

    def test_different_paths_produce_different_hashes(self):
        hash1 = compute_content_hash("/test/image1.jpg")
        hash2 = compute_content_hash("/test/image2.jpg")
        assert hash1 != hash2

    def test_returns_string(self):
        result = compute_content_hash("/test/image.jpg")
        assert isinstance(result, str)


class TestGenerateImageId:
    """Tests for image ID generation."""

    def test_starts_with_img_prefix(self):
        image_id = generate_image_id("/test.jpg", "camera_A", "abc123")
        assert image_id.startswith("img_")

    def test_deterministic_for_same_inputs(self):
        id1 = generate_image_id("/test.jpg", "camera_A", "abc123")
        id2 = generate_image_id("/test.jpg", "camera_A", "abc123")
        assert id1 == id2

    def test_different_for_different_inputs(self):
        id1 = generate_image_id("/test1.jpg", "camera_A", "abc123")
        id2 = generate_image_id("/test2.jpg", "camera_A", "abc123")
        assert id1 != id2


class TestImageRecord:
    """Tests for ImageRecord model."""

    def test_create_with_required_fields(self):
        record = ImageRecord(
            image_id="img_001",
            path="/test.jpg",
            source="camera_A",
            content_hash="abc123",
        )
        assert record.image_id == "img_001"
        assert record.status == "submitted"

    def test_to_dict(self):
        record = ImageRecord(
            image_id="img_001",
            path="/test.jpg",
            source="camera_A",
            content_hash="abc123",
        )
        d = record.to_dict()
        assert d["image_id"] == "img_001"
        assert d["path"] == "/test.jpg"
        assert "submitted_at" in d

    def test_from_dict(self):
        data = {
            "image_id": "img_001",
            "path": "/test.jpg",
            "source": "camera_A",
            "content_hash": "abc123",
            "status": "processed",
            "submitted_at": "2026-04-07T14:33:00Z",
        }
        record = ImageRecord.from_dict(data)
        assert record.image_id == "img_001"
        assert record.status == "processed"


class TestInMemoryImageRegistry:
    """Tests for in-memory image registry."""

    @pytest.fixture
    def registry(self):
        return InMemoryImageRegistry()

    @pytest.fixture
    def sample_record(self):
        return ImageRecord(
            image_id="img_001",
            path="/test.jpg",
            source="camera_A",
            content_hash="abc123",
        )

    @pytest.mark.asyncio
    async def test_create_and_get_by_id(self, registry, sample_record):
        await registry.create(sample_record)
        result = await registry.get_by_id("img_001")
        assert result is not None
        assert result.image_id == "img_001"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, registry):
        result = await registry.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_hash(self, registry, sample_record):
        await registry.create(sample_record)
        result = await registry.get_by_hash("/test.jpg", "camera_A", "abc123")
        assert result is not None
        assert result.image_id == "img_001"

    @pytest.mark.asyncio
    async def test_get_by_hash_not_found(self, registry):
        result = await registry.get_by_hash("/nonexistent.jpg", "cam", "xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_status(self, registry, sample_record):
        await registry.create(sample_record)
        success = await registry.update_status("img_001", "processed")
        assert success is True

        record = await registry.get_by_id("img_001")
        assert record.status == "processed"

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, registry):
        success = await registry.update_status("nonexistent", "processed")
        assert success is False

    @pytest.mark.asyncio
    async def test_list_all(self, registry):
        record1 = ImageRecord("img_001", "/test1.jpg", "cam_A", "hash1")
        record2 = ImageRecord("img_002", "/test2.jpg", "cam_B", "hash2")

        await registry.create(record1)
        await registry.create(record2)

        all_records = await registry.list_all()
        assert len(all_records) == 2

    @pytest.mark.asyncio
    async def test_clear(self, registry, sample_record):
        await registry.create(sample_record)
        registry.clear()
        result = await registry.get_by_id("img_001")
        assert result is None
