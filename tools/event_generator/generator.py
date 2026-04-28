"""Event generator for testing with deterministic seed mode."""

import random
import hashlib
from typing import Optional

from shared.events import (
    EventEnvelope,
    create_envelope,
    Topics,
    generate_event_id,
)


class EventGenerator:
    """
    Generates test events with optional deterministic seeding.

    In deterministic mode (with seed), produces identical event sequences
    for the same seed, enabling reproducible tests.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the event generator.

        Args:
            seed: Optional seed for deterministic generation.
                  If None, uses system randomness.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._event_counter = 0

    def _generate_deterministic_id(self, prefix: str) -> str:
        """Generate a deterministic ID based on counter and seed."""
        if self._seed is not None:
            # Deterministic ID based on seed and counter
            hash_input = f"{self._seed}:{self._event_counter}"
            hash_hex = hashlib.md5(hash_input.encode()).hexdigest()[:12]
            return f"{prefix}_{hash_hex}"
        else:
            # Random ID
            return f"{prefix}_{self._rng.randint(100000, 999999)}"

    def generate_image_submitted(self, count: int = 1) -> list[EventEnvelope]:
        """
        Generate image.submitted events.

        Args:
            count: Number of events to generate

        Returns:
            List of EventEnvelope objects
        """
        events = []
        sources = ["camera_A", "camera_B", "camera_C", "upload", "batch_import"]

        for _ in range(count):
            self._event_counter += 1

            image_id = self._generate_deterministic_id("img")
            event_id = self._generate_deterministic_id("evt")

            # Generate deterministic path and source
            path_num = self._rng.randint(1, 9999)
            source = self._rng.choice(sources)

            envelope = EventEnvelope(
                topic=Topics.IMAGE_SUBMITTED,
                event_id=event_id,
                payload={
                    "image_id": image_id,
                    "path": f"images/image_{path_num:04d}.jpg",
                    "source": source,
                },
            )
            events.append(envelope)

        return events

    def generate_annotation_corrected(
        self,
        image_id: str,
        object_index: int = 0,
        new_label: Optional[str] = None,
    ) -> EventEnvelope:
        """
        Generate an annotation.corrected event.

        Args:
            image_id: ID of the image to correct
            object_index: Index of the object to correct
            new_label: New label for the object. If None, picks randomly.

        Returns:
            EventEnvelope for the correction
        """
        self._event_counter += 1

        labels = ["car", "truck", "bus", "person", "bicycle", "motorcycle", "dog", "cat"]
        if new_label is None:
            new_label = self._rng.choice(labels)

        event_id = self._generate_deterministic_id("evt")

        return EventEnvelope(
            topic=Topics.ANNOTATION_CORRECTED,
            event_id=event_id,
            payload={
                "image_id": image_id,
                "patch": {f"objects.{object_index}.label": new_label},
                "reviewer": "cli_user",
            },
        )

    def generate_query_submitted(
        self,
        kind: str = "text",
        value: Optional[str] = None,
        top_k: int = 5,
    ) -> EventEnvelope:
        """
        Generate a query.submitted event.

        Args:
            kind: Query type ("text" or "image")
            value: Query value. If None, generates random text query.
            top_k: Number of results to request

        Returns:
            EventEnvelope for the query
        """
        self._event_counter += 1

        if value is None:
            if kind == "text":
                adjectives = ["red", "blue", "large", "small", "fast", "parked"]
                nouns = ["car", "truck", "person", "bicycle", "dog"]
                value = f"{self._rng.choice(adjectives)} {self._rng.choice(nouns)}"
            else:
                value = f"images/query_{self._rng.randint(1, 100):03d}.jpg"

        event_id = self._generate_deterministic_id("evt")
        query_id = self._generate_deterministic_id("q")

        return EventEnvelope(
            topic=Topics.QUERY_SUBMITTED,
            event_id=event_id,
            payload={
                "query_id": query_id,
                "kind": kind,
                "value": value,
                "top_k": top_k,
            },
        )

    def generate_mixed_sequence(
        self,
        num_images: int = 10,
        num_corrections: int = 2,
        num_queries: int = 3,
    ) -> list[EventEnvelope]:
        """
        Generate a mixed sequence of events for integration testing.

        Args:
            num_images: Number of image.submitted events
            num_corrections: Number of annotation.corrected events
            num_queries: Number of query.submitted events

        Returns:
            List of EventEnvelope objects in a realistic order
        """
        events = []

        # Generate image submissions
        image_events = self.generate_image_submitted(num_images)
        events.extend(image_events)

        # Generate corrections for random images
        image_ids = [e.payload["image_id"] for e in image_events]
        for _ in range(min(num_corrections, len(image_ids))):
            image_id = self._rng.choice(image_ids)
            correction = self.generate_annotation_corrected(image_id)
            events.append(correction)

        # Generate queries
        for _ in range(num_queries):
            query = self.generate_query_submitted()
            events.append(query)

        return events

    def reset(self) -> None:
        """Reset the generator to initial state with same seed."""
        self._rng = random.Random(self._seed)
        self._event_counter = 0

    @property
    def seed(self) -> Optional[int]:
        """Return the seed used for this generator."""
        return self._seed

    @property
    def event_count(self) -> int:
        """Return the number of events generated so far."""
        return self._event_counter
