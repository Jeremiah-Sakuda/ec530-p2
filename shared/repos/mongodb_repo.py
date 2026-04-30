"""MongoDB implementation of document repository."""

from typing import Any, Optional
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from shared.repos.document_repo import DocumentRepo


class MongoDBRepo(DocumentRepo):
    """
    MongoDB-based document repository.

    Production-ready storage with:
    - Async operations via Motor
    - Efficient upsert with $addToSet for event tracking
    - Indexed queries for label and status filters
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "ec530_p2",
        collection_name: str = "annotations",
    ):
        """
        Initialize MongoDB repository.

        Args:
            uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name for annotations
        """
        self._uri = uri
        self._db_name = db_name
        self._collection_name = collection_name
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def connect(self) -> None:
        """
        Establish connection to MongoDB.

        Call this before using the repository.
        """
        self._client = AsyncIOMotorClient(self._uri)
        self._db = self._client[self._db_name]
        self._collection = self._db[self._collection_name]

        # Create indexes for efficient queries
        await self._collection.create_index("image_id", unique=True)
        await self._collection.create_index("status")
        await self._collection.create_index("objects.label")

    async def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if self._collection is None:
            raise RuntimeError("MongoDB repository not connected. Call connect() first.")

    async def upsert(self, image_id: str, data: dict[str, Any]) -> None:
        """Insert or update an annotation document."""
        self._ensure_connected()

        data["image_id"] = image_id
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Use upsert to insert or update
        await self._collection.update_one(
            {"image_id": image_id},
            {
                "$set": data,
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            },
            upsert=True,
        )

    async def get(self, image_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an annotation document by image ID."""
        self._ensure_connected()

        doc = await self._collection.find_one({"image_id": image_id})
        if doc:
            # Remove MongoDB's internal _id field
            doc.pop("_id", None)
        return doc

    async def delete(self, image_id: str) -> bool:
        """Delete an annotation document."""
        self._ensure_connected()

        result = await self._collection.delete_one({"image_id": image_id})
        return result.deleted_count > 0

    async def query(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Query documents by filters.

        Supported filters:
        - label: Object label to match
        - min_conf: Minimum confidence threshold
        - status: Document status
        """
        self._ensure_connected()

        mongo_filter = {}

        if "status" in filters:
            mongo_filter["status"] = filters["status"]

        if "label" in filters:
            mongo_filter["objects.label"] = filters["label"]

        if "min_conf" in filters:
            mongo_filter["objects.conf"] = {"$gte": filters["min_conf"]}

        cursor = self._collection.find(mongo_filter)
        results = []

        async for doc in cursor:
            doc.pop("_id", None)
            results.append(doc)

        return results

    async def add_processed_event(self, image_id: str, event_id: str) -> None:
        """
        Record that an event has been processed.

        Uses $addToSet to avoid duplicates efficiently.
        """
        self._ensure_connected()

        await self._collection.update_one(
            {"image_id": image_id},
            {
                "$addToSet": {"processed_event_ids": event_id},
                "$setOnInsert": {
                    "image_id": image_id,
                    "objects": [],
                    "model_version": "unknown",
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
            },
            upsert=True,
        )

    async def has_processed_event(self, image_id: str, event_id: str) -> bool:
        """Check if an event has already been processed."""
        self._ensure_connected()

        doc = await self._collection.find_one(
            {"image_id": image_id, "processed_event_ids": event_id}
        )
        return doc is not None

    async def count(self) -> int:
        """Return total number of documents."""
        self._ensure_connected()
        return await self._collection.count_documents({})

    async def clear(self) -> None:
        """Clear all documents."""
        self._ensure_connected()
        await self._collection.delete_many({})

    async def drop_collection(self) -> None:
        """Drop the entire collection. Use for testing cleanup."""
        self._ensure_connected()
        await self._collection.drop()
