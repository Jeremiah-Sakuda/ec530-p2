"""Tests for event generator determinism and functionality."""

import pytest
from tools.event_generator import EventGenerator
from shared.events import Topics


class TestEventGeneratorDeterminism:
    """Tests that seeded generator produces identical output."""

    def test_same_seed_produces_same_events(self):
        gen1 = EventGenerator(seed=42)
        gen2 = EventGenerator(seed=42)

        events1 = gen1.generate_image_submitted(5)
        events2 = gen2.generate_image_submitted(5)

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.event_id == e2.event_id
            assert e1.payload == e2.payload

    def test_different_seeds_produce_different_events(self):
        gen1 = EventGenerator(seed=42)
        gen2 = EventGenerator(seed=99)

        events1 = gen1.generate_image_submitted(5)
        events2 = gen2.generate_image_submitted(5)

        # At least one event should differ
        different = any(
            e1.event_id != e2.event_id or e1.payload != e2.payload
            for e1, e2 in zip(events1, events2)
        )
        assert different

    def test_reset_restores_initial_state(self):
        gen = EventGenerator(seed=42)

        events1 = gen.generate_image_submitted(3)
        gen.reset()
        events2 = gen.generate_image_submitted(3)

        for e1, e2 in zip(events1, events2):
            assert e1.event_id == e2.event_id
            assert e1.payload == e2.payload

    def test_no_seed_produces_random_events(self):
        gen1 = EventGenerator()  # No seed
        gen2 = EventGenerator()  # No seed

        events1 = gen1.generate_image_submitted(3)
        events2 = gen2.generate_image_submitted(3)

        # Events should be different (with very high probability)
        different = any(
            e1.event_id != e2.event_id
            for e1, e2 in zip(events1, events2)
        )
        assert different


class TestEventGeneratorImageSubmitted:
    """Tests for image.submitted event generation."""

    def test_generates_correct_topic(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_image_submitted(1)

        assert events[0].topic == Topics.IMAGE_SUBMITTED

    def test_generates_required_fields(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_image_submitted(1)

        payload = events[0].payload
        assert "image_id" in payload
        assert "path" in payload
        assert "source" in payload

    def test_generates_correct_count(self):
        gen = EventGenerator(seed=42)

        events = gen.generate_image_submitted(10)
        assert len(events) == 10

    def test_image_id_has_correct_prefix(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_image_submitted(1)

        assert events[0].payload["image_id"].startswith("img_")

    def test_event_id_has_correct_prefix(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_image_submitted(1)

        assert events[0].event_id.startswith("evt_")

    def test_path_is_jpg(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_image_submitted(1)

        assert events[0].payload["path"].endswith(".jpg")

    def test_source_is_valid(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_image_submitted(10)

        valid_sources = {"camera_A", "camera_B", "camera_C", "upload", "batch_import"}
        for event in events:
            assert event.payload["source"] in valid_sources


class TestEventGeneratorAnnotationCorrected:
    """Tests for annotation.corrected event generation."""

    def test_generates_correct_topic(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_annotation_corrected("img_001")

        assert event.topic == Topics.ANNOTATION_CORRECTED

    def test_includes_image_id(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_annotation_corrected("img_test")

        assert event.payload["image_id"] == "img_test"

    def test_includes_patch(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_annotation_corrected("img_001", object_index=2)

        assert "patch" in event.payload
        assert "objects.2.label" in event.payload["patch"]

    def test_uses_provided_label(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_annotation_corrected("img_001", new_label="airplane")

        assert event.payload["patch"]["objects.0.label"] == "airplane"

    def test_includes_reviewer(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_annotation_corrected("img_001")

        assert event.payload["reviewer"] == "cli_user"


class TestEventGeneratorQuerySubmitted:
    """Tests for query.submitted event generation."""

    def test_generates_correct_topic(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_query_submitted()

        assert event.topic == Topics.QUERY_SUBMITTED

    def test_text_query_has_correct_kind(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_query_submitted(kind="text")

        assert event.payload["kind"] == "text"

    def test_image_query_has_correct_kind(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_query_submitted(kind="image")

        assert event.payload["kind"] == "image"

    def test_uses_provided_value(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_query_submitted(kind="text", value="blue truck")

        assert event.payload["value"] == "blue truck"

    def test_uses_provided_top_k(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_query_submitted(top_k=20)

        assert event.payload["top_k"] == 20

    def test_has_query_id(self):
        gen = EventGenerator(seed=42)
        event = gen.generate_query_submitted()

        assert event.payload["query_id"].startswith("q_")


class TestEventGeneratorMixedSequence:
    """Tests for mixed event sequence generation."""

    def test_generates_all_event_types(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_mixed_sequence(
            num_images=3,
            num_corrections=2,
            num_queries=2,
        )

        topics = {e.topic for e in events}
        assert Topics.IMAGE_SUBMITTED in topics
        assert Topics.ANNOTATION_CORRECTED in topics
        assert Topics.QUERY_SUBMITTED in topics

    def test_generates_correct_counts(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_mixed_sequence(
            num_images=5,
            num_corrections=2,
            num_queries=3,
        )

        image_count = sum(1 for e in events if e.topic == Topics.IMAGE_SUBMITTED)
        correction_count = sum(1 for e in events if e.topic == Topics.ANNOTATION_CORRECTED)
        query_count = sum(1 for e in events if e.topic == Topics.QUERY_SUBMITTED)

        assert image_count == 5
        assert correction_count == 2
        assert query_count == 3

    def test_corrections_reference_generated_images(self):
        gen = EventGenerator(seed=42)
        events = gen.generate_mixed_sequence(
            num_images=3,
            num_corrections=2,
            num_queries=0,
        )

        image_ids = {
            e.payload["image_id"]
            for e in events
            if e.topic == Topics.IMAGE_SUBMITTED
        }
        correction_image_ids = {
            e.payload["image_id"]
            for e in events
            if e.topic == Topics.ANNOTATION_CORRECTED
        }

        # All corrected images should be from generated images
        assert correction_image_ids.issubset(image_ids)


class TestEventGeneratorProperties:
    """Tests for generator properties."""

    def test_seed_property(self):
        gen = EventGenerator(seed=42)
        assert gen.seed == 42

    def test_seed_none_when_not_provided(self):
        gen = EventGenerator()
        assert gen.seed is None

    def test_event_count_starts_at_zero(self):
        gen = EventGenerator(seed=42)
        assert gen.event_count == 0

    def test_event_count_increments(self):
        gen = EventGenerator(seed=42)
        gen.generate_image_submitted(3)
        assert gen.event_count == 3

        gen.generate_annotation_corrected("img_001")
        assert gen.event_count == 4

    def test_reset_clears_event_count(self):
        gen = EventGenerator(seed=42)
        gen.generate_image_submitted(5)
        gen.reset()
        assert gen.event_count == 0
