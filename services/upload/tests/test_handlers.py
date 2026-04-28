"""Tests for upload service handlers."""

import pytest
from services.upload import (
    InMemoryImageRegistry,
    handle_upload,
    get_image_status,
    update_image_status,
)
from shared.broker import InMemoryBroker
from shared.events import Topics


class TestHandleUpload:
    """Tests for the upload handler."""

    @pytest.fixture
    def registry(self):
        return InMemoryImageRegistry()

    @pytest.fixture
    async def broker(self):
        broker = InMemoryBroker()
        await broker.start()
        return broker

    @pytest.mark.asyncio
    async def test_creates_new_image(self, registry, broker):
        result = await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        assert result.image_id.startswith("img_")
        assert result.is_duplicate is False
        assert result.record.status == "submitted"

    @pytest.mark.asyncio
    async def test_publishes_event(self, registry, broker):
        await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        messages = broker.get_published_for_topic(Topics.IMAGE_SUBMITTED)
        assert len(messages) == 1
        assert messages[0].payload["path"] == "/images/test.jpg"
        assert messages[0].payload["source"] == "camera_A"

    @pytest.mark.asyncio
    async def test_idempotency_same_image(self, registry, broker):
        # First upload
        result1 = await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        # Clear published messages
        broker.clear_published()

        # Second upload of same image
        result2 = await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        # Should return same image_id
        assert result1.image_id == result2.image_id
        assert result2.is_duplicate is True

        # Should NOT publish another event
        messages = broker.get_published_for_topic(Topics.IMAGE_SUBMITTED)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_different_source_creates_new(self, registry, broker):
        result1 = await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        result2 = await handle_upload(
            path="/images/test.jpg",
            source="camera_B",
            registry=registry,
            broker=broker,
        )

        # Different source = different image
        assert result1.image_id != result2.image_id
        assert result2.is_duplicate is False

    @pytest.mark.asyncio
    async def test_different_path_creates_new(self, registry, broker):
        result1 = await handle_upload(
            path="/images/test1.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        result2 = await handle_upload(
            path="/images/test2.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        assert result1.image_id != result2.image_id

    @pytest.mark.asyncio
    async def test_event_contains_correct_payload(self, registry, broker):
        result = await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        messages = broker.get_published_for_topic(Topics.IMAGE_SUBMITTED)
        payload = messages[0].payload

        assert payload["image_id"] == result.image_id
        assert payload["path"] == "/images/test.jpg"
        assert payload["source"] == "camera_A"


class TestGetImageStatus:
    """Tests for getting image status."""

    @pytest.fixture
    def registry(self):
        return InMemoryImageRegistry()

    @pytest.fixture
    async def broker(self):
        broker = InMemoryBroker()
        await broker.start()
        return broker

    @pytest.mark.asyncio
    async def test_returns_record_after_upload(self, registry, broker):
        result = await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        record = await get_image_status(result.image_id, registry)
        assert record is not None
        assert record.image_id == result.image_id
        assert record.path == "/images/test.jpg"

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent(self, registry):
        record = await get_image_status("nonexistent", registry)
        assert record is None


class TestUpdateImageStatus:
    """Tests for updating image status."""

    @pytest.fixture
    def registry(self):
        return InMemoryImageRegistry()

    @pytest.fixture
    async def broker(self):
        broker = InMemoryBroker()
        await broker.start()
        return broker

    @pytest.mark.asyncio
    async def test_updates_status(self, registry, broker):
        result = await handle_upload(
            path="/images/test.jpg",
            source="camera_A",
            registry=registry,
            broker=broker,
        )

        success = await update_image_status(result.image_id, "processed", registry)
        assert success is True

        record = await get_image_status(result.image_id, registry)
        assert record.status == "processed"

    @pytest.mark.asyncio
    async def test_returns_false_for_nonexistent(self, registry):
        success = await update_image_status("nonexistent", "processed", registry)
        assert success is False
