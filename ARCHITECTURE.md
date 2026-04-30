# Architecture: Event-Driven Image Annotation and Retrieval System

## System Overview

This system implements an event-driven architecture for processing images through object detection, annotation storage, vector embedding, and semantic search. Services communicate asynchronously via a message broker, enabling loose coupling and horizontal scalability.

## Architecture Diagram

```
                                    +----------------+
                                    |     CLI        |
                                    |  (Click app)   |
                                    +-------+--------+
                                            |
                        REST API calls      |
         +----------------------------------+----------------------------------+
         |                                  |                                  |
         v                                  v                                  v
+----------------+               +------------------+               +------------------+
|  Upload API    |               |  Annotation API  |               |   Query API      |
|  POST /images  |               | GET/PATCH /ann   |               | POST /query      |
+-------+--------+               +--------+---------+               +--------+---------+
        |                                 |                                  |
        | publish                         | read/write                       | search
        v                                 v                                  v
+-------+--------+               +--------+---------+               +--------+---------+
| image.submitted|               |  Document Repo   |               |   Vector Repo    |
+-------+--------+               | (MongoDB/TinyDB) |               |    (FAISS)       |
        |                        +------------------+               +------------------+
        |                                 ^                                  ^
        |                                 |                                  |
        v                                 |                                  |
+-------+--------+                        |                                  |
|   Inference    |                        |                                  |
|   Service      |                        |                                  |
|  (mock_detect) |                        |                                  |
+-------+--------+                        |                                  |
        |                                 |                                  |
        | publish                         |                                  |
        v                                 |                                  |
+-------+------------+                    |                                  |
|inference.completed |                    |                                  |
+-------+------------+                    |                                  |
        |                                 |                                  |
        v                                 |                                  |
+-------+--------+                        |                                  |
|   Annotation   +------------------------+                                  |
|   Service      |  upsert                                                   |
+-------+--------+                                                           |
        |                                                                    |
        | publish                                                            |
        v                                                                    |
+-------+------------+                                                       |
|annotation.stored   |                                                       |
+-------+------------+                                                       |
        |                                                                    |
        v                                                                    |
+-------+--------+                                                           |
|   Embedding    |                                                           |
|   Service      |                                                           |
|  (mock_embed)  |                                                           |
+-------+--------+                                                           |
        |                                                                    |
        | publish                                                            |
        v                                                                    |
+-------+------------+                                                       |
|embedding.created   |                                                       |
+-------+------------+                                                       |
        |                                                                    |
        v                                                                    |
+-------+--------+                                                           |
| Vector Index   +-----------------------------------------------------------+
|   Service      |  add vectors
+----------------+
```

## Event Flow

### Primary Pipeline

1. **image.submitted** - Client uploads an image
2. **inference.completed** - Object detection results with bounding boxes and labels
3. **annotation.stored** - Annotation persisted to document store
4. **embedding.created** - Vector embeddings generated for each detected object
5. Vectors indexed in FAISS for similarity search

### Correction Flow

1. Client submits correction via API
2. **annotation.corrected** published
3. Annotation service applies patch, updates history
4. **annotation.stored** re-published to trigger re-embedding

### Query Flow

1. Client submits text or image query
2. Query vectorized using mock embedder
3. FAISS similarity search returns top-k matches
4. Results hydrated with annotation metadata

## Services

### Upload Service (`services/upload/`)

**Responsibility:** Accept image uploads, generate image IDs, publish submission events.

| Component | Description |
|-----------|-------------|
| `repo.py` | Image registry with content hash for idempotency |
| `handlers.py` | Upload handler with duplicate detection |
| `api.py` | FastAPI endpoints: `POST /images`, `GET /images/{id}` |

### Inference Service (`services/inference/`)

**Responsibility:** Consume image submissions, run object detection, publish results.

| Component | Description |
|-----------|-------------|
| `mock_detector.py` | Deterministic mock returning 1-5 objects per image |
| `handlers.py` | Event handler for `image.submitted` |

**Design Decision:** Uses seeded RNG based on `image_id` hash for deterministic, reproducible outputs during testing.

### Annotation Service (`services/annotation/`)

**Responsibility:** Store inference results, handle corrections, track history.

| Component | Description |
|-----------|-------------|
| `handlers.py` | Handlers for `inference.completed` and `annotation.corrected` |
| `api.py` | FastAPI endpoints: `GET/PATCH /annotations/{id}`, `GET /annotations` |

**Idempotency:** Tracks `processed_event_ids` per document to prevent duplicate processing.

### Embedding Service (`services/embedding/`)

**Responsibility:** Generate vector embeddings for detected objects.

| Component | Description |
|-----------|-------------|
| `mock_embedder.py` | Deterministic 128-dim Gaussian vectors |
| `handlers.py` | Event handler for `annotation.stored` |

**Design Decision:** Embedding dimension fixed at 128 for FAISS IndexFlatL2 compatibility.

### Vector Index Service (`services/vector_index/`)

**Responsibility:** Index embeddings in FAISS, provide search API.

| Component | Description |
|-----------|-------------|
| `handlers.py` | Event handler for `embedding.created` |
| `api.py` | FastAPI endpoint: `POST /search` |

### Query Service (`services/query/`)

**Responsibility:** Orchestrate end-to-end query flow.

| Component | Description |
|-----------|-------------|
| `handlers.py` | `handle_query_submitted` and `execute_query` |
| `api.py` | FastAPI endpoints: `POST /query/text`, `POST /query/image` |

### CLI Service (`services/cli/`)

**Responsibility:** Provide command-line interface for system interaction.

| Command | Description |
|---------|-------------|
| `upload` | Submit an image to the pipeline |
| `query` | Search for similar objects (text or image) |
| `get-annotation` | Retrieve annotation by image ID |
| `correct` | Apply a correction patch |
| `relabel` | Quick label update shortcut |
| `health` | Check service health status |

## Shared Components

### Event System (`shared/events/`)

| Module | Description |
|--------|-------------|
| `envelope.py` | `EventEnvelope` dataclass with `event_id`, `timestamp`, `payload` |
| `topics.py` | Topic constants: `IMAGE_SUBMITTED`, `INFERENCE_COMPLETED`, etc. |
| `schema.py` | Pydantic models for payload validation |

**Event Envelope Structure:**
```json
{
  "type": "publish",
  "topic": "image.submitted",
  "event_id": "evt_01HXZ...",
  "timestamp": "2024-04-07T14:33:00Z",
  "schema_version": 1,
  "payload": { ... }
}
```

### Broker Abstraction (`shared/broker/`)

| Implementation | Use Case |
|----------------|----------|
| `InMemoryBroker` | Unit tests, deterministic delivery |
| `RedisPubSubBroker` | Production, distributed systems |

**Interface:**
```python
class BaseBroker(ABC):
    async def publish(topic: str, envelope: EventEnvelope) -> None
    async def subscribe(topic: str, handler: Callable) -> None
    async def start() -> None
    async def stop() -> None
```

### Document Repository (`shared/repos/`)

| Implementation | Use Case |
|----------------|----------|
| `InMemoryDocumentRepo` | Unit tests |
| `TinyDBRepo` | Local development |
| `MongoDBRepo` | Production |

**Interface:**
```python
class DocumentRepo(ABC):
    async def upsert(image_id: str, data: dict) -> None
    async def get(image_id: str) -> Optional[dict]
    async def query(filters: dict) -> List[dict]
    async def add_processed_event(image_id: str, event_id: str) -> None
    async def has_processed_event(image_id: str, event_id: str) -> bool
```

### Vector Repository (`shared/repos/vector_repo.py`)

**Implementation:** FAISS IndexFlatL2 with custom ID mapping.

**Features:**
- 128-dimensional L2 distance search
- ID mapping: `(image_id, object_id)` <-> FAISS internal ID
- Vector replacement for re-embedding
- Persistence via `save()`/`load()`

## Design Decisions

### 1. Event-Driven Architecture

**Decision:** Services communicate exclusively via events, not direct calls.

**Rationale:**
- Loose coupling enables independent deployment
- Event replay supports debugging and recovery
- Natural fit for batch processing pipelines

### 2. Idempotent Event Processing

**Decision:** Each handler checks `processed_event_ids` before processing.

**Rationale:**
- At-least-once delivery semantics from Redis pub-sub
- Safe event replay for testing and recovery
- Prevents duplicate side effects (duplicate vectors, duplicate documents)

### 3. Deterministic Mocks

**Decision:** Mock detector and embedder use seeded RNG based on input hash.

**Rationale:**
- Reproducible test results
- Predictable outputs for integration testing
- Event replay produces identical state

### 4. Document Store for Annotations

**Decision:** Use MongoDB/TinyDB for annotation documents, not relational DB.

**Rationale:**
- Flexible schema for object arrays
- Natural fit for JSON event payloads
- History tracking via embedded arrays

### 5. FAISS for Vector Search

**Decision:** Use FAISS IndexFlatL2 for vector similarity search.

**Rationale:**
- In-process, no external dependencies
- Exact L2 search for small-scale use
- Can upgrade to IVF for larger datasets

## Data Models

### Annotation Document

```json
{
  "image_id": "img_abc123",
  "objects": [
    {
      "object_id": "obj_0",
      "label": "car",
      "bbox": [10, 20, 100, 150],
      "conf": 0.92
    }
  ],
  "model_version": "mock_v1",
  "status": "pending|corrected",
  "history": [
    {
      "timestamp": "2024-04-07T15:00:00Z",
      "event_id": "evt_xyz",
      "reviewer": "user@example.com",
      "patch": {"objects.0.label": "truck"}
    }
  ],
  "processed_event_ids": ["evt_001", "evt_002"]
}
```

### Vector Index Entry

```
Key: (image_id, object_id)
Value: 128-dimensional float32 vector
```

## Testing Strategy

### Unit Tests

Each service has isolated tests with in-memory dependencies:
- Mock broker captures published events
- In-memory document repo for annotation tests
- In-memory vector repo for index tests

### Integration Tests (`tests/integration/`)

End-to-end pipeline tests using in-memory implementations:
- Full flow from image submission to query results
- Correction flow with re-indexing
- Replay consistency verification

### Fault Injection Tests (`tests/fault_injection/`)

Resilience testing:
- Duplicate event idempotency
- Malformed payload handling
- Invalid envelope structures

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `DB_NAME` | `ec530_p2` | Database name |
| `USE_TINYDB` | `false` | Use TinyDB instead of MongoDB |

## Future Considerations

### Scalability

- Replace FAISS IndexFlatL2 with IndexIVF for larger datasets
- Add Redis Streams for persistent event log
- Horizontal scaling via consumer groups

### Production Readiness

- Add OpenTelemetry tracing
- Implement circuit breakers for external calls
- Add health check endpoints to all services
- Container orchestration with Kubernetes

### ML Integration

- Replace mock detector with real model (YOLO, Detectron2)
- Replace mock embedder with CLIP or similar
- Add model versioning and A/B testing
