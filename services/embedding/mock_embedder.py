"""Mock embedder with deterministic output for testing."""

import random
from dataclasses import dataclass
from typing import List


# Embedding dimension used throughout the system
EMBEDDING_DIM = 128


@dataclass
class EmbeddingResult:
    """Result of embedding an object."""

    object_id: str
    image_id: str
    vector: List[float]
    dim: int = EMBEDDING_DIM

    def to_dict(self) -> dict:
        """Convert to dictionary for event payload."""
        return {
            "object_id": self.object_id,
            "vector_ref": f"vec_store/{self.image_id}_{self.object_id}",
            "dim": self.dim,
        }


def mock_embed(image_id: str, object_id: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Generate a deterministic embedding vector.

    Uses hash of image_id:object_id as seed to produce reproducible
    128-dimensional vectors with Gaussian-distributed values.

    Args:
        image_id: Image identifier
        object_id: Object identifier within the image
        dim: Vector dimensionality (default 128)

    Returns:
        List of floats representing the embedding vector
    """
    # Create deterministic seed from image_id and object_id
    seed_str = f"{image_id}:{object_id}"
    seed = hash(seed_str) & 0xFFFFFFFF  # Ensure positive 32-bit int
    rng = random.Random(seed)

    # Generate Gaussian-distributed vector
    return [rng.gauss(0, 1) for _ in range(dim)]


def mock_embed_text(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Generate a deterministic embedding vector for text.

    Used for text queries. Creates reproducible vectors based on text content.

    Args:
        text: Text to embed
        dim: Vector dimensionality (default 128)

    Returns:
        List of floats representing the embedding vector
    """
    seed_str = f"text:{text}"
    seed = hash(seed_str) & 0xFFFFFFFF
    rng = random.Random(seed)

    return [rng.gauss(0, 1) for _ in range(dim)]


def mock_embed_image_query(image_path: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Generate a deterministic embedding vector for an image query.

    Used for image-based similarity queries. Creates reproducible vectors
    based on image path.

    Args:
        image_path: Path to the query image
        dim: Vector dimensionality (default 128)

    Returns:
        List of floats representing the embedding vector
    """
    seed_str = f"image:{image_path}"
    seed = hash(seed_str) & 0xFFFFFFFF
    rng = random.Random(seed)

    return [rng.gauss(0, 1) for _ in range(dim)]


def embed_objects(image_id: str, object_ids: List[str]) -> List[EmbeddingResult]:
    """
    Generate embeddings for multiple objects in an image.

    Args:
        image_id: Image identifier
        object_ids: List of object identifiers

    Returns:
        List of EmbeddingResult objects
    """
    results = []
    for object_id in object_ids:
        vector = mock_embed(image_id, object_id)
        results.append(
            EmbeddingResult(
                object_id=object_id,
                image_id=image_id,
                vector=vector,
            )
        )
    return results
