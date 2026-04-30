"""Repository abstractions for data storage."""

from shared.repos.document_repo import DocumentRepo, AnnotationDocument
from shared.repos.tinydb_repo import TinyDBRepo, InMemoryDocumentRepo
from shared.repos.mongodb_repo import MongoDBRepo
from shared.repos.vector_repo import VectorRepo, SearchResult

__all__ = [
    "DocumentRepo",
    "AnnotationDocument",
    "TinyDBRepo",
    "InMemoryDocumentRepo",
    "MongoDBRepo",
    "VectorRepo",
    "SearchResult",
]
