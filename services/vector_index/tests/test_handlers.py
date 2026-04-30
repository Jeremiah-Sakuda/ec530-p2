"""Tests for vector index service event handlers."""

import pytest
from shared.events import create_envelope
from shared.events.topics import Topics
from shared.repos import VectorRepo
from services.vector_index.handlers import (
    handle_embedding_created,
    create_embedding_created_handler,
)


@pytest.fixture
def vector_repo():
    """Create a fresh vector repository for each test."""
    return VectorRepo(dim=128)


@pytest.fixture
def valid_embedding_created_envelope():
    """Create a valid embedding.created event envelope."""
    return create_envelope(
        Topics.EMBEDDING_CREATED,
        {
            "image_id": "img_test123",
            "embeddings": [
                {"object_id": "obj_0", "vector_ref": "vec_store/img_test123_obj_0", "dim": 128},
                {"object_id": "obj_1", "vector_ref": "vec_store/img_test123_obj_1", "dim": 128},
            ],
        },
    )


class TestHandleEmbeddingCreated:
    """Tests for handle_embedding_created handler."""

    @pytest.mark.asyncio
    async def test_adds_vectors_to_index(self, vector_repo, valid_embedding_created_envelope):
        """Handler should add vectors to the index."""
        await handle_embedding_created(valid_embedding_created_envelope, vector_repo)

        assert vector_repo.ntotal == 2
        assert vector_repo.has("img_test123", "obj_0")
        assert vector_repo.has("img_test123", "obj_1")

    @pytest.mark.asyncio
    async def test_vectors_are_searchable(self, vector_repo, valid_embedding_created_envelope):
        """Added vectors should be searchable."""
        await handle_embedding_created(valid_embedding_created_envelope, vector_repo)

        # Get one vector and search for it
        vec = vector_repo.get("img_test123", "obj_0")
        results = vector_repo.search(vec, top_k=5)

        assert len(results) >= 1
        assert results[0].image_id == "img_test123"

    @pytest.mark.asyncio
    async def test_replaces_existing_vectors(self, vector_repo):
        """Re-indexing should replace existing vectors."""
        envelope1 = create_envelope(
            Topics.EMBEDDING_CREATED,
            {
                "image_id": "img_test",
                "embeddings": [{"object_id": "obj_0", "vector_ref": "ref", "dim": 128}],
            },
        )

        await handle_embedding_created(envelope1, vector_repo)
        assert vector_repo.ntotal == 1

        # Process again - should replace, not add
        await handle_embedding_created(envelope1, vector_repo)
        assert vector_repo.ntotal == 1

    @pytest.mark.asyncio
    async def test_deterministic_vectors(self, vector_repo):
        """Same inputs should produce same vectors."""
        envelope = create_envelope(
            Topics.EMBEDDING_CREATED,
            {
                "image_id": "img_determinism",
                "embeddings": [{"object_id": "obj_0", "vector_ref": "ref", "dim": 128}],
            },
        )

        await handle_embedding_created(envelope, vector_repo)
        vec1 = vector_repo.get("img_determinism", "obj_0")

        # Clear and re-add
        vector_repo.clear()
        await handle_embedding_created(envelope, vector_repo)
        vec2 = vector_repo.get("img_determinism", "obj_0")

        assert vec1 == vec2

    @pytest.mark.asyncio
    async def test_invalid_payload_does_not_crash(self, vector_repo):
        """Handler should gracefully handle invalid payloads."""
        invalid_envelope = create_envelope(
            Topics.EMBEDDING_CREATED,
            {"invalid_field": "value"},
        )

        # Should not raise
        await handle_embedding_created(invalid_envelope, vector_repo)

        assert vector_repo.ntotal == 0

    @pytest.mark.asyncio
    async def test_empty_embeddings(self, vector_repo):
        """Handler should handle empty embeddings list."""
        envelope = create_envelope(
            Topics.EMBEDDING_CREATED,
            {"image_id": "img_empty", "embeddings": []},
        )

        await handle_embedding_created(envelope, vector_repo)

        assert vector_repo.ntotal == 0


class TestCreateEmbeddingCreatedHandler:
    """Tests for the handler factory function."""

    @pytest.mark.asyncio
    async def test_returns_callable(self, vector_repo):
        """Factory should return a callable handler."""
        handler = create_embedding_created_handler(vector_repo)
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_processes_events(
        self, vector_repo, valid_embedding_created_envelope
    ):
        """Created handler should process events correctly."""
        handler = create_embedding_created_handler(vector_repo)
        await handler(valid_embedding_created_envelope)

        assert vector_repo.ntotal == 2
