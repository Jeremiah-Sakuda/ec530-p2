# Event-Driven Image Annotation and Retrieval System

An event-driven microservices system for processing images through object detection, annotation storage, vector embedding, and semantic search.

## Features

- **Event-Driven Architecture**: Services communicate via Redis pub-sub for loose coupling
- **Object Detection Pipeline**: Automatic detection with mock detector (extensible to real models)
- **Annotation Management**: Store, query, and correct object annotations
- **Vector Search**: FAISS-based similarity search for text and image queries
- **Idempotent Processing**: Safe event replay with deduplication
- **CLI Interface**: Command-line tool for all operations

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, in-memory broker available for testing)
- MongoDB (optional, TinyDB available for local development)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ec530-p2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional - defaults work for local dev)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov=shared --cov=tools

# Run specific test categories
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/fault_injection/   # Fault injection tests
```

## Project Structure

```
ec530-p2/
├── services/
│   ├── upload/          # Image upload service
│   ├── inference/       # Object detection service
│   ├── annotation/      # Annotation storage service
│   ├── embedding/       # Vector embedding service
│   ├── vector_index/    # FAISS indexing service
│   ├── query/           # Query orchestration service
│   └── cli/             # Command-line interface
├── shared/
│   ├── events/          # Event envelope, topics, schemas
│   ├── broker/          # Message broker abstraction
│   └── repos/           # Data repository abstractions
├── tools/
│   └── event_generator/ # Test event generation
└── tests/
    ├── unit/            # Unit tests
    ├── integration/     # End-to-end tests
    └── fault_injection/ # Resilience tests
```

## CLI Usage

The CLI provides commands for interacting with the system.

### Upload an Image

```bash
python -m services.cli.main upload /path/to/image.jpg --source camera_01
```

### Query for Objects

```bash
# Text-based search
python -m services.cli.main query --text "find cars"

# Image-based search
python -m services.cli.main query --image /path/to/query.jpg --top-k 10
```

### View Annotation

```bash
python -m services.cli.main get-annotation img_abc123
```

### Correct an Annotation

```bash
# Using a patch file
python -m services.cli.main correct img_abc123 patch.json --reviewer user@example.com

# Quick relabel
python -m services.cli.main relabel img_abc123 obj_0 truck --reviewer user@example.com
```

### Check Service Health

```bash
python -m services.cli.main health
```

## Event Topics

| Topic | Description |
|-------|-------------|
| `image.submitted` | New image uploaded |
| `inference.completed` | Object detection results |
| `annotation.stored` | Annotation persisted |
| `annotation.corrected` | Human correction applied |
| `embedding.created` | Vector embeddings generated |
| `query.submitted` | Search query initiated |
| `query.completed` | Search results returned |

## API Endpoints

### Upload Service

```
POST /images
  Request: {"path": "/images/photo.jpg", "source": "upload"}
  Response: {"image_id": "img_abc123", "status": "submitted"}

GET /images/{image_id}
  Response: {"image_id": "...", "status": "...", "submitted_at": "..."}
```

### Annotation Service

```
GET /annotations/{image_id}
  Response: {"image_id": "...", "objects": [...], "status": "..."}

PATCH /annotations/{image_id}
  Request: {"patch": {"objects.0.label": "truck"}, "reviewer": "user@example.com"}
  Response: {"image_id": "...", "status": "corrected", ...}

GET /annotations?label=car&min_conf=0.8
  Response: [{"image_id": "...", "objects": [...], ...}, ...]
```

### Query Service

```
POST /query/text
  Request: {"value": "find red cars", "top_k": 5}
  Response: {"results": [{"image_id": "...", "object_id": "...", "label": "...", "score": 0.95}]}

POST /query/image
  Request: {"path": "/query/sample.jpg", "top_k": 10}
  Response: {"results": [...]}
```

### Vector Index Service

```
POST /search
  Request: {"vector": [0.1, 0.2, ...], "top_k": 5}
  Response: {"results": [{"image_id": "...", "object_id": "...", "distance": 0.05}]}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `DB_NAME` | `ec530_p2` | Database name |
| `USE_TINYDB` | `false` | Use file-based TinyDB |
| `UPLOAD_SERVICE_URL` | `http://localhost:8001` | Upload service URL |
| `ANNOTATION_SERVICE_URL` | `http://localhost:8002` | Annotation service URL |
| `QUERY_SERVICE_URL` | `http://localhost:8003` | Query service URL |

### Local Development (No External Dependencies)

The system can run entirely in-memory for local development:

```python
from shared.broker import InMemoryBroker
from shared.repos import InMemoryDocumentRepo, VectorRepo

broker = InMemoryBroker()
doc_repo = InMemoryDocumentRepo()
vector_repo = VectorRepo(dim=128)
```

## Testing

### Test Categories

1. **Unit Tests**: Isolated component testing with mocks
2. **Integration Tests**: Full pipeline verification
3. **Fault Injection Tests**: Resilience under failure conditions

### Example Test Run

```bash
# Run all tests with verbose output
pytest -v

# Run only integration tests
pytest tests/integration/ -v

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Test Fixtures

The `tools/event_generator` provides deterministic test events:

```python
from tools.event_generator import EventGenerator

generator = EventGenerator(seed=42)
events = generator.generate_image_submitted(count=10)
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design documentation.

## Development

### Adding a New Service

1. Create directory under `services/`
2. Implement handlers for consumed events
3. Add FastAPI endpoints if needed
4. Add unit tests
5. Update integration tests

### Extending the Mock Detector

The mock detector in `services/inference/mock_detector.py` can be replaced with a real model:

```python
# Replace mock_detect with actual model inference
def detect(image_path: str) -> List[DetectedObject]:
    # Load and run YOLO, Detectron2, etc.
    ...
```

## License

MIT License
