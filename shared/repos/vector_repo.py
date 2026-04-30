"""FAISS-based vector repository for similarity search."""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


@dataclass
class SearchResult:
    """Result from a similarity search."""

    image_id: str
    object_id: str
    distance: float
    score: float  # Normalized similarity score (higher is better)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "object_id": self.object_id,
            "distance": self.distance,
            "score": self.score,
        }


class VectorRepo:
    """
    FAISS-based vector repository for object embeddings.

    Features:
    - Flat L2 index for exact nearest neighbor search
    - ID mapping to track (image_id, object_id) pairs
    - Replace support for updated embeddings
    - Persistence to disk
    """

    def __init__(self, dim: int = 128, index_path: Optional[str] = None):
        """
        Initialize the vector repository.

        Args:
            dim: Dimensionality of vectors (default 128)
            index_path: Optional path for persistence
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required for VectorRepo")

        self._dim = dim
        self._index_path = index_path
        self._index = faiss.IndexFlatL2(dim)

        # ID mappings
        self._id_map: Dict[int, Tuple[str, str]] = {}  # faiss_id -> (image_id, object_id)
        self._reverse_map: Dict[Tuple[str, str], int] = {}  # (image_id, object_id) -> faiss_id

        # Track deleted IDs for reuse
        self._deleted_ids: List[int] = []

    @property
    def dim(self) -> int:
        """Return the vector dimensionality."""
        return self._dim

    @property
    def ntotal(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._id_map)

    def add(self, image_id: str, object_id: str, vector: List[float]) -> None:
        """
        Add a vector to the index.

        If a vector for this (image_id, object_id) already exists, it will be replaced.

        Args:
            image_id: Image identifier
            object_id: Object identifier
            vector: Embedding vector
        """
        key = (image_id, object_id)

        # If key exists, replace
        if key in self._reverse_map:
            self._remove(key)

        # Convert to numpy array
        vec_array = np.array([vector], dtype=np.float32)

        # Get the next FAISS ID
        faiss_id = self._index.ntotal

        # Add to index
        self._index.add(vec_array)

        # Update mappings
        self._id_map[faiss_id] = key
        self._reverse_map[key] = faiss_id

    def _remove(self, key: Tuple[str, str]) -> None:
        """
        Mark a vector as removed (soft delete).

        Note: FAISS IndexFlatL2 doesn't support true removal,
        so we just remove from our mappings.

        Args:
            key: (image_id, object_id) tuple
        """
        if key in self._reverse_map:
            faiss_id = self._reverse_map[key]
            del self._reverse_map[key]
            del self._id_map[faiss_id]
            self._deleted_ids.append(faiss_id)

    def remove(self, image_id: str, object_id: str) -> bool:
        """
        Remove a vector from the index.

        Args:
            image_id: Image identifier
            object_id: Object identifier

        Returns:
            True if vector was removed, False if not found
        """
        key = (image_id, object_id)
        if key in self._reverse_map:
            self._remove(key)
            return True
        return False

    def get(self, image_id: str, object_id: str) -> Optional[List[float]]:
        """
        Get a vector by image_id and object_id.

        Args:
            image_id: Image identifier
            object_id: Object identifier

        Returns:
            Vector if found, None otherwise
        """
        key = (image_id, object_id)
        if key not in self._reverse_map:
            return None

        faiss_id = self._reverse_map[key]
        vector = self._index.reconstruct(faiss_id)
        return vector.tolist()

    def search(self, vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """
        Search for nearest neighbors.

        Args:
            vector: Query vector
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by distance
        """
        if self.ntotal == 0:
            return []

        # Convert to numpy array
        vec_array = np.array([vector], dtype=np.float32)

        # Search - request more results in case some are deleted
        k = min(top_k + len(self._deleted_ids), self._index.ntotal)
        distances, indices = self._index.search(vec_array, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            if idx not in self._id_map:  # Skip deleted vectors
                continue

            image_id, object_id = self._id_map[idx]

            # Convert L2 distance to similarity score (1 / (1 + distance))
            score = 1.0 / (1.0 + float(dist))

            results.append(SearchResult(
                image_id=image_id,
                object_id=object_id,
                distance=float(dist),
                score=score,
            ))

            if len(results) >= top_k:
                break

        return results

    def search_by_ids(
        self, image_id: str, object_id: str, top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar vectors given an existing vector's IDs.

        Args:
            image_id: Image identifier of the query vector
            object_id: Object identifier of the query vector
            top_k: Number of results to return

        Returns:
            List of SearchResult objects (excluding the query vector)
        """
        vector = self.get(image_id, object_id)
        if vector is None:
            return []

        # Get one extra result since we'll exclude the query itself
        results = self.search(vector, top_k + 1)

        # Filter out the query vector
        return [r for r in results if not (r.image_id == image_id and r.object_id == object_id)][:top_k]

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the index and mappings to disk.

        Args:
            path: Optional path override (uses index_path if not provided)
        """
        save_path = path or self._index_path
        if save_path is None:
            raise ValueError("No path specified for saving")

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, f"{save_path}.faiss")

        # Save mappings
        mappings = {
            "id_map": self._id_map,
            "reverse_map": self._reverse_map,
            "deleted_ids": self._deleted_ids,
            "dim": self._dim,
        }
        with open(f"{save_path}.mappings", "wb") as f:
            pickle.dump(mappings, f)

    def load(self, path: Optional[str] = None) -> None:
        """
        Load the index and mappings from disk.

        Args:
            path: Optional path override (uses index_path if not provided)
        """
        load_path = path or self._index_path
        if load_path is None:
            raise ValueError("No path specified for loading")

        # Load FAISS index
        self._index = faiss.read_index(f"{load_path}.faiss")

        # Load mappings
        with open(f"{load_path}.mappings", "rb") as f:
            mappings = pickle.load(f)

        self._id_map = mappings["id_map"]
        self._reverse_map = mappings["reverse_map"]
        self._deleted_ids = mappings.get("deleted_ids", [])
        self._dim = mappings.get("dim", 128)

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._index = faiss.IndexFlatL2(self._dim)
        self._id_map.clear()
        self._reverse_map.clear()
        self._deleted_ids.clear()

    def has(self, image_id: str, object_id: str) -> bool:
        """
        Check if a vector exists in the index.

        Args:
            image_id: Image identifier
            object_id: Object identifier

        Returns:
            True if vector exists
        """
        return (image_id, object_id) in self._reverse_map

    def get_all_ids(self) -> List[Tuple[str, str]]:
        """
        Get all (image_id, object_id) pairs in the index.

        Returns:
            List of ID tuples
        """
        return list(self._reverse_map.keys())
