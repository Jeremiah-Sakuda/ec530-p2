"""TinyDB implementation of document repository for local development."""

import os
from typing import Any, Optional
from datetime import datetime, timezone

from tinydb import TinyDB, Query

from shared.repos.document_repo import DocumentRepo


class TinyDBRepo(DocumentRepo):
    """
    TinyDB-based document repository.

    File-based storage with zero configuration.
    Ideal for local development and testing.
    """

    def __init__(self, db_path: str = "annotations.json"):
        """
        Initialize TinyDB repository.

        Args:
            db_path: Path to the JSON database file
        """
        self._db_path = db_path
        self._db = TinyDB(db_path)
        self._table = self._db.table("annotations")

    async def upsert(self, image_id: str, data: dict[str, Any]) -> None:
        """Insert or update an annotation document."""
        doc = Query()

        # Ensure image_id is in the data
        data["image_id"] = image_id
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        existing = self._table.search(doc.image_id == image_id)
        if existing:
            # Preserve created_at from existing document
            if "created_at" not in data:
                data["created_at"] = existing[0].get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                )
            self._table.update(data, doc.image_id == image_id)
        else:
            if "created_at" not in data:
                data["created_at"] = datetime.now(timezone.utc).isoformat()
            self._table.insert(data)

    async def get(self, image_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an annotation document by image ID."""
        doc = Query()
        results = self._table.search(doc.image_id == image_id)
        return results[0] if results else None

    async def delete(self, image_id: str) -> bool:
        """Delete an annotation document."""
        doc = Query()
        removed = self._table.remove(doc.image_id == image_id)
        return len(removed) > 0

    async def query(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Query documents by filters.

        Supported filters:
        - label: Object label to match
        - min_conf: Minimum confidence threshold
        - status: Document status
        """
        all_docs = self._table.all()
        results = []

        for doc in all_docs:
            if self._matches_filters(doc, filters):
                results.append(doc)

        return results

    def _matches_filters(self, doc: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if a document matches the given filters."""
        # Status filter
        if "status" in filters:
            if doc.get("status") != filters["status"]:
                return False

        # Label filter - check if any object has the label
        if "label" in filters:
            objects = doc.get("objects", [])
            if not any(obj.get("label") == filters["label"] for obj in objects):
                return False

        # Min confidence filter - check if any object meets threshold
        if "min_conf" in filters:
            objects = doc.get("objects", [])
            if not any(obj.get("conf", 0) >= filters["min_conf"] for obj in objects):
                return False

        return True

    async def add_processed_event(self, image_id: str, event_id: str) -> None:
        """Record that an event has been processed."""
        doc = await self.get(image_id)
        if doc is None:
            # Create minimal document if it doesn't exist
            doc = {
                "image_id": image_id,
                "objects": [],
                "model_version": "unknown",
                "status": "pending",
                "processed_event_ids": [],
            }

        processed = doc.get("processed_event_ids", [])
        if event_id not in processed:
            processed.append(event_id)
            doc["processed_event_ids"] = processed
            await self.upsert(image_id, doc)

    async def has_processed_event(self, image_id: str, event_id: str) -> bool:
        """Check if an event has already been processed."""
        doc = await self.get(image_id)
        if doc is None:
            return False
        return event_id in doc.get("processed_event_ids", [])

    async def count(self) -> int:
        """Return total number of documents."""
        return len(self._table)

    async def clear(self) -> None:
        """Clear all documents."""
        self._table.truncate()

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def delete_db_file(self) -> None:
        """Delete the database file. Useful for test cleanup."""
        self.close()
        if os.path.exists(self._db_path):
            os.remove(self._db_path)


class InMemoryDocumentRepo(DocumentRepo):
    """
    In-memory document repository for unit tests.

    No persistence - data is lost when the object is destroyed.
    Faster than TinyDB for test scenarios.
    """

    def __init__(self):
        self._documents: dict[str, dict[str, Any]] = {}

    async def upsert(self, image_id: str, data: dict[str, Any]) -> None:
        """Insert or update an annotation document."""
        data["image_id"] = image_id
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        if image_id in self._documents:
            # Preserve created_at
            if "created_at" not in data:
                data["created_at"] = self._documents[image_id].get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                )
        else:
            if "created_at" not in data:
                data["created_at"] = datetime.now(timezone.utc).isoformat()

        self._documents[image_id] = data

    async def get(self, image_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an annotation document by image ID."""
        return self._documents.get(image_id)

    async def delete(self, image_id: str) -> bool:
        """Delete an annotation document."""
        if image_id in self._documents:
            del self._documents[image_id]
            return True
        return False

    async def query(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Query documents by filters."""
        results = []
        for doc in self._documents.values():
            if self._matches_filters(doc, filters):
                results.append(doc)
        return results

    def _matches_filters(self, doc: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if a document matches the given filters."""
        if "status" in filters:
            if doc.get("status") != filters["status"]:
                return False

        if "label" in filters:
            objects = doc.get("objects", [])
            if not any(obj.get("label") == filters["label"] for obj in objects):
                return False

        if "min_conf" in filters:
            objects = doc.get("objects", [])
            if not any(obj.get("conf", 0) >= filters["min_conf"] for obj in objects):
                return False

        return True

    async def add_processed_event(self, image_id: str, event_id: str) -> None:
        """Record that an event has been processed."""
        if image_id not in self._documents:
            self._documents[image_id] = {
                "image_id": image_id,
                "objects": [],
                "model_version": "unknown",
                "status": "pending",
                "processed_event_ids": [],
            }

        processed = self._documents[image_id].get("processed_event_ids", [])
        if event_id not in processed:
            processed.append(event_id)
            self._documents[image_id]["processed_event_ids"] = processed

    async def has_processed_event(self, image_id: str, event_id: str) -> bool:
        """Check if an event has already been processed."""
        doc = self._documents.get(image_id)
        if doc is None:
            return False
        return event_id in doc.get("processed_event_ids", [])

    async def count(self) -> int:
        """Return total number of documents."""
        return len(self._documents)

    async def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
