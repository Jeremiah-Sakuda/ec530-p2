"""Tests for query service event handlers."""

import pytest
from shared.events import create_envelope
from shared.events.topics import Topics
from shared.broker import InMemoryBroker
from shared.repos import VectorRepo, InMemoryDocumentRepo
from services.query.handlers import (
    handle_query_submitted,
    create_query_submitted_handler,
    execute_query,
    get_object_label,
)
from services.embedding import mock_embed


@pytest.fixture
def broker():
    """Create a fresh in-memory broker for each test."""
    return InMemoryBroker()


@pytest.fixture
def vector_repo():
    """Create a fresh vector repository for each test."""
    return VectorRepo(dim=128)


@pytest.fixture
def document_repo():
    """Create a fresh document repository for each test."""
    return InMemoryDocumentRepo()


@pytest.fixture
async def populated_repos(vector_repo, document_repo):
    """Set up repos with sample data."""
    # Add vectors
    for i, label in enumerate(["car", "person", "bicycle"]):
        image_id = f"img_{i}"
        object_id = "obj_0"

        # Add vector
        vec = mock_embed(image_id, object_id)
        vector_repo.add(image_id, object_id, vec)

        # Add annotation
        await document_repo.upsert(image_id, {
            "image_id": image_id,
            "objects": [
                {"object_id": object_id, "label": label, "conf": 0.95},
            ],
            "model_version": "mock_v1",
            "status": "pending",
        })

    return vector_repo, document_repo


class TestHandleQuerySubmitted:
    """Tests for handle_query_submitted handler."""

    @pytest.mark.asyncio
    async def test_publishes_query_completed(self, broker, populated_repos):
        """Handler should publish query.completed event."""
        await broker.start()
        vector_repo, document_repo = populated_repos

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_123",
                "kind": "text",
                "value": "car",
                "top_k": 5,
            },
        )

        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_query_completed_contains_query_id(self, broker, populated_repos):
        """Published event should contain the query_id."""
        await broker.start()
        vector_repo, document_repo = populated_repos

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_456",
                "kind": "text",
                "value": "car",
                "top_k": 5,
            },
        )

        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert published[0].payload["query_id"] == "qry_456"

    @pytest.mark.asyncio
    async def test_results_contain_labels(self, broker, populated_repos):
        """Results should include hydrated labels."""
        await broker.start()
        vector_repo, document_repo = populated_repos

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_labels",
                "kind": "text",
                "value": "test",
                "top_k": 5,
            },
        )

        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        results = published[0].payload["results"]

        # All results should have labels from annotations
        for result in results:
            assert "label" in result
            assert result["label"] in ["car", "person", "bicycle"]

    @pytest.mark.asyncio
    async def test_handles_text_query(self, broker, populated_repos):
        """Should handle text queries."""
        await broker.start()
        vector_repo, document_repo = populated_repos

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_text",
                "kind": "text",
                "value": "find cars",
                "top_k": 3,
            },
        )

        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert len(published[0].payload["results"]) == 3

    @pytest.mark.asyncio
    async def test_handles_image_query(self, broker, populated_repos):
        """Should handle image queries."""
        await broker.start()
        vector_repo, document_repo = populated_repos

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_image",
                "kind": "image",
                "value": "/path/to/query.jpg",
                "top_k": 3,
            },
        )

        await handle_query_submitted(envelope, vector_repo, document_repo, broker)

        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_invalid_payload_does_not_crash(self, broker, vector_repo, document_repo):
        """Handler should gracefully handle invalid payloads."""
        await broker.start()

        invalid_envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {"invalid_field": "value"},
        )

        # Should not raise
        await handle_query_submitted(invalid_envelope, vector_repo, document_repo, broker)

        # Should not publish anything
        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert len(published) == 0


class TestExecuteQuery:
    """Tests for execute_query function."""

    @pytest.mark.asyncio
    async def test_returns_results(self, populated_repos):
        """Should return query results."""
        vector_repo, document_repo = populated_repos

        results = await execute_query(
            kind="text",
            value="test query",
            top_k=5,
            vector_repo=vector_repo,
            document_repo=document_repo,
        )

        assert len(results) == 3
        for r in results:
            assert "image_id" in r
            assert "object_id" in r
            assert "score" in r
            assert "label" in r

    @pytest.mark.asyncio
    async def test_empty_index(self, vector_repo, document_repo):
        """Should return empty results for empty index."""
        results = await execute_query(
            kind="text",
            value="test",
            top_k=5,
            vector_repo=vector_repo,
            document_repo=document_repo,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_respects_top_k(self, populated_repos):
        """Should respect top_k limit."""
        vector_repo, document_repo = populated_repos

        results = await execute_query(
            kind="text",
            value="test",
            top_k=2,
            vector_repo=vector_repo,
            document_repo=document_repo,
        )

        assert len(results) == 2


class TestGetObjectLabel:
    """Tests for get_object_label function."""

    def test_finds_label(self):
        """Should find label in annotation."""
        annotation = {
            "objects": [
                {"object_id": "obj_0", "label": "car"},
                {"object_id": "obj_1", "label": "person"},
            ],
        }

        assert get_object_label(annotation, "obj_0") == "car"
        assert get_object_label(annotation, "obj_1") == "person"

    def test_returns_unknown_for_missing(self):
        """Should return 'unknown' for missing object."""
        annotation = {
            "objects": [{"object_id": "obj_0", "label": "car"}],
        }

        assert get_object_label(annotation, "obj_99") == "unknown"

    def test_returns_unknown_for_none(self):
        """Should return 'unknown' for None annotation."""
        assert get_object_label(None, "obj_0") == "unknown"


class TestCreateQuerySubmittedHandler:
    """Tests for the handler factory function."""

    @pytest.mark.asyncio
    async def test_returns_callable(self, vector_repo, document_repo, broker):
        """Factory should return a callable handler."""
        handler = create_query_submitted_handler(vector_repo, document_repo, broker)
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_processes_events(self, broker, populated_repos):
        """Created handler should process events correctly."""
        await broker.start()
        vector_repo, document_repo = populated_repos

        handler = create_query_submitted_handler(vector_repo, document_repo, broker)

        envelope = create_envelope(
            Topics.QUERY_SUBMITTED,
            {
                "query_id": "qry_factory",
                "kind": "text",
                "value": "test",
                "top_k": 5,
            },
        )

        await handler(envelope)

        published = broker.get_published_for_topic(Topics.QUERY_COMPLETED)
        assert len(published) == 1
