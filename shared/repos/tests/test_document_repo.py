"""Tests for document repository implementations."""

import pytest
import os
import tempfile
from shared.repos import (
    DocumentRepo,
    AnnotationDocument,
    InMemoryDocumentRepo,
    TinyDBRepo,
)


class DocumentRepoTestSuite:
    """
    Abstract test suite for DocumentRepo implementations.

    Inherit from this class and implement get_repo() to test
    any DocumentRepo implementation.
    """

    @pytest.fixture
    def repo(self) -> DocumentRepo:
        """Override this to return the repo implementation to test."""
        raise NotImplementedError

    @pytest.fixture
    def sample_document(self) -> dict:
        """Sample annotation document for testing."""
        return {
            "image_id": "img_test123",
            "objects": [
                {"object_id": "obj_0", "label": "car", "bbox": [10, 20, 100, 200], "conf": 0.95},
                {"object_id": "obj_1", "label": "person", "bbox": [50, 60, 150, 250], "conf": 0.88},
            ],
            "model_version": "mock_v1",
            "status": "pending",
        }

    # ========================================================================
    # Basic CRUD Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_upsert_and_get(self, repo, sample_document):
        """Should store and retrieve a document."""
        await repo.upsert("img_test123", sample_document)

        result = await repo.get("img_test123")

        assert result is not None
        assert result["image_id"] == "img_test123"
        assert len(result["objects"]) == 2
        assert result["model_version"] == "mock_v1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, repo):
        """Should return None for nonexistent document."""
        result = await repo.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, repo, sample_document):
        """Should update existing document."""
        await repo.upsert("img_test123", sample_document)

        # Update with new data
        updated = sample_document.copy()
        updated["status"] = "reviewed"
        await repo.upsert("img_test123", updated)

        result = await repo.get("img_test123")
        assert result["status"] == "reviewed"
        # Should still only have one document
        assert await repo.count() == 1

    @pytest.mark.asyncio
    async def test_delete_existing(self, repo, sample_document):
        """Should delete existing document and return True."""
        await repo.upsert("img_test123", sample_document)

        deleted = await repo.delete("img_test123")

        assert deleted is True
        assert await repo.get("img_test123") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, repo):
        """Should return False when deleting nonexistent document."""
        deleted = await repo.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_count(self, repo, sample_document):
        """Should return correct document count."""
        assert await repo.count() == 0

        await repo.upsert("img_1", sample_document)
        assert await repo.count() == 1

        await repo.upsert("img_2", sample_document)
        assert await repo.count() == 2

    @pytest.mark.asyncio
    async def test_clear(self, repo, sample_document):
        """Should remove all documents."""
        await repo.upsert("img_1", sample_document)
        await repo.upsert("img_2", sample_document)

        await repo.clear()

        assert await repo.count() == 0

    # ========================================================================
    # Query Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_query_by_status(self, repo):
        """Should filter by status."""
        await repo.upsert("img_1", {"status": "pending", "objects": []})
        await repo.upsert("img_2", {"status": "reviewed", "objects": []})
        await repo.upsert("img_3", {"status": "pending", "objects": []})

        results = await repo.query({"status": "pending"})

        assert len(results) == 2
        assert all(r["status"] == "pending" for r in results)

    @pytest.mark.asyncio
    async def test_query_by_label(self, repo):
        """Should filter by object label."""
        await repo.upsert("img_1", {
            "objects": [{"label": "car", "conf": 0.9}],
            "status": "pending",
        })
        await repo.upsert("img_2", {
            "objects": [{"label": "person", "conf": 0.8}],
            "status": "pending",
        })
        await repo.upsert("img_3", {
            "objects": [{"label": "car", "conf": 0.7}, {"label": "truck", "conf": 0.6}],
            "status": "pending",
        })

        results = await repo.query({"label": "car"})

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_min_conf(self, repo):
        """Should filter by minimum confidence."""
        await repo.upsert("img_1", {
            "objects": [{"label": "car", "conf": 0.9}],
            "status": "pending",
        })
        await repo.upsert("img_2", {
            "objects": [{"label": "car", "conf": 0.5}],
            "status": "pending",
        })
        await repo.upsert("img_3", {
            "objects": [{"label": "car", "conf": 0.8}],
            "status": "pending",
        })

        results = await repo.query({"min_conf": 0.75})

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_empty_result(self, repo, sample_document):
        """Should return empty list when no matches."""
        await repo.upsert("img_1", sample_document)

        results = await repo.query({"label": "nonexistent"})

        assert results == []

    # ========================================================================
    # Idempotency Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_add_processed_event(self, repo, sample_document):
        """Should record processed event."""
        await repo.upsert("img_test123", sample_document)

        await repo.add_processed_event("img_test123", "evt_abc123")

        assert await repo.has_processed_event("img_test123", "evt_abc123") is True

    @pytest.mark.asyncio
    async def test_has_processed_event_false_for_new(self, repo, sample_document):
        """Should return False for unprocessed event."""
        await repo.upsert("img_test123", sample_document)

        assert await repo.has_processed_event("img_test123", "evt_new") is False

    @pytest.mark.asyncio
    async def test_add_processed_event_creates_document_if_needed(self, repo):
        """Should create minimal document if it doesn't exist."""
        await repo.add_processed_event("img_new", "evt_first")

        assert await repo.has_processed_event("img_new", "evt_first") is True

    @pytest.mark.asyncio
    async def test_add_processed_event_is_idempotent(self, repo, sample_document):
        """Adding same event multiple times should not create duplicates."""
        await repo.upsert("img_test123", sample_document)

        await repo.add_processed_event("img_test123", "evt_abc123")
        await repo.add_processed_event("img_test123", "evt_abc123")
        await repo.add_processed_event("img_test123", "evt_abc123")

        doc = await repo.get("img_test123")
        # Should only appear once
        assert doc["processed_event_ids"].count("evt_abc123") == 1

    @pytest.mark.asyncio
    async def test_multiple_processed_events(self, repo, sample_document):
        """Should track multiple processed events."""
        await repo.upsert("img_test123", sample_document)

        await repo.add_processed_event("img_test123", "evt_1")
        await repo.add_processed_event("img_test123", "evt_2")
        await repo.add_processed_event("img_test123", "evt_3")

        assert await repo.has_processed_event("img_test123", "evt_1") is True
        assert await repo.has_processed_event("img_test123", "evt_2") is True
        assert await repo.has_processed_event("img_test123", "evt_3") is True
        assert await repo.has_processed_event("img_test123", "evt_4") is False


class TestAnnotationDocument:
    """Tests for AnnotationDocument dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        doc = AnnotationDocument(
            image_id="img_123",
            objects=[{"object_id": "obj_0", "label": "car"}],
            model_version="v1",
            status="pending",
        )

        result = doc.to_dict()

        assert result["image_id"] == "img_123"
        assert result["objects"] == [{"object_id": "obj_0", "label": "car"}]
        assert result["model_version"] == "v1"
        assert result["status"] == "pending"
        assert "created_at" in result
        assert "updated_at" in result

    def test_from_dict(self):
        """Should create from dictionary correctly."""
        data = {
            "image_id": "img_456",
            "objects": [{"object_id": "obj_1", "label": "person"}],
            "model_version": "v2",
            "status": "reviewed",
            "history": [{"action": "created"}],
        }

        doc = AnnotationDocument.from_dict(data)

        assert doc.image_id == "img_456"
        assert doc.model_version == "v2"
        assert doc.status == "reviewed"
        assert doc.history == [{"action": "created"}]

    def test_from_dict_with_defaults(self):
        """Should use defaults for missing fields."""
        data = {"image_id": "img_789"}

        doc = AnnotationDocument.from_dict(data)

        assert doc.image_id == "img_789"
        assert doc.objects == []
        assert doc.model_version == "unknown"
        assert doc.status == "pending"


class TestInMemoryDocumentRepo(DocumentRepoTestSuite):
    """Test suite for InMemoryDocumentRepo."""

    @pytest.fixture
    def repo(self) -> InMemoryDocumentRepo:
        return InMemoryDocumentRepo()


class TestTinyDBRepo(DocumentRepoTestSuite):
    """Test suite for TinyDBRepo."""

    @pytest.fixture
    def repo(self) -> TinyDBRepo:
        # Create a temporary file for the database
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        repo = TinyDBRepo(db_path=path)
        yield repo

        # Cleanup
        repo.delete_db_file()

    @pytest.mark.asyncio
    async def test_persistence(self):
        """Should persist data to file."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        try:
            # Write data
            repo1 = TinyDBRepo(db_path=path)
            await repo1.upsert("img_persist", {
                "objects": [],
                "status": "pending",
            })
            repo1.close()

            # Read with new instance
            repo2 = TinyDBRepo(db_path=path)
            result = await repo2.get("img_persist")
            repo2.close()

            assert result is not None
            assert result["image_id"] == "img_persist"
        finally:
            os.remove(path)
