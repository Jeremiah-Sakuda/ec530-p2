"""Mock object detector with deterministic output for testing."""

import random
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """A detected object from inference."""

    object_id: str
    label: str
    bbox: list[int]
    conf: float

    def to_dict(self) -> dict:
        """Convert to dictionary for event payload."""
        return {
            "object_id": self.object_id,
            "label": self.label,
            "bbox": self.bbox,
            "conf": self.conf,
        }


# Labels used by the mock detector
MOCK_LABELS = ["car", "person", "bicycle", "dog", "tree", "truck", "cat", "bus"]

# Model version for the mock detector
MOCK_MODEL_VERSION = "mock_v1"


def mock_detect(image_id: str) -> list[DetectedObject]:
    """
    Deterministic mock object detection.

    Uses hash of image_id as seed to produce reproducible results.
    Always returns 1-5 objects with consistent labels and bounding boxes
    for the same image_id.

    Args:
        image_id: Unique image identifier

    Returns:
        List of detected objects (1-5 items)
    """
    # Use hash of image_id for deterministic seeding
    seed = hash(image_id) & 0xFFFFFFFF  # Ensure positive 32-bit int
    rng = random.Random(seed)

    # Determine number of objects (1-5)
    count = rng.randint(1, 5)

    objects = []
    for i in range(count):
        # Generate deterministic bounding box (x1, y1, x2, y2)
        x1 = rng.randint(0, 400)
        y1 = rng.randint(0, 400)
        x2 = x1 + rng.randint(50, 200)
        y2 = y1 + rng.randint(50, 200)

        obj = DetectedObject(
            object_id=f"obj_{i}",
            label=rng.choice(MOCK_LABELS),
            bbox=[x1, y1, x2, y2],
            conf=round(rng.uniform(0.5, 0.99), 2),
        )
        objects.append(obj)

    return objects
