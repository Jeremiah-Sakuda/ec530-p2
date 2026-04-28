"""Topic constants for pub-sub messaging."""


class Topics:
    """
    All topic names used in the event-driven pipeline.

    | Topic                | Publisher  | Subscribers      | Purpose                              |
    |----------------------|------------|------------------|--------------------------------------|
    | image.submitted      | Upload     | Inference        | New image enters the pipeline        |
    | inference.completed  | Inference  | Annotation       | Detector produced object list        |
    | annotation.stored    | Annotation | Embedding        | Document is persistent, ready to embed|
    | annotation.corrected | CLI        | Annotation, Embedding | Human correction triggers re-process |
    | embedding.created    | Embedding  | Vector Index     | New vectors ready to index           |
    | query.submitted      | CLI        | Query            | Async search request                 |
    | query.completed      | Query      | CLI              | Async search response                |
    """

    # Core pipeline topics
    IMAGE_SUBMITTED = "image.submitted"
    INFERENCE_COMPLETED = "inference.completed"
    ANNOTATION_STORED = "annotation.stored"
    ANNOTATION_CORRECTED = "annotation.corrected"
    EMBEDDING_CREATED = "embedding.created"

    # Query topics
    QUERY_SUBMITTED = "query.submitted"
    QUERY_COMPLETED = "query.completed"

    @classmethod
    def all_topics(cls) -> list[str]:
        """Return all topic names."""
        return [
            cls.IMAGE_SUBMITTED,
            cls.INFERENCE_COMPLETED,
            cls.ANNOTATION_STORED,
            cls.ANNOTATION_CORRECTED,
            cls.EMBEDDING_CREATED,
            cls.QUERY_SUBMITTED,
            cls.QUERY_COMPLETED,
        ]

    @classmethod
    def pipeline_topics(cls) -> list[str]:
        """Return core pipeline topics (excluding query)."""
        return [
            cls.IMAGE_SUBMITTED,
            cls.INFERENCE_COMPLETED,
            cls.ANNOTATION_STORED,
            cls.ANNOTATION_CORRECTED,
            cls.EMBEDDING_CREATED,
        ]
