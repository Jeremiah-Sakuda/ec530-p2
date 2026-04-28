"""Tests for in-memory broker implementation."""

import pytest
from shared.broker import InMemoryBroker, BaseBroker
from shared.events import EventEnvelope, create_envelope, Topics


class TestInMemoryBrokerInterface:
    """Tests that InMemoryBroker implements BaseBroker correctly."""

    def test_implements_base_broker(self):
        broker = InMemoryBroker()
        assert isinstance(broker, BaseBroker)


class TestInMemoryBrokerLifecycle:
    """Tests for broker start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_starts_not_running(self):
        broker = InMemoryBroker()
        assert broker.is_running is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        broker = InMemoryBroker()
        await broker.start()
        assert broker.is_running is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        broker = InMemoryBroker()
        await broker.start()
        await broker.stop()
        assert broker.is_running is False


class TestInMemoryBrokerPublish:
    """Tests for publish functionality."""

    @pytest.mark.asyncio
    async def test_publishes_message(self):
        broker = InMemoryBroker()
        await broker.start()

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        assert len(broker.published_messages) == 1
        assert broker.published_messages[0][0] == Topics.IMAGE_SUBMITTED
        assert broker.published_messages[0][1].event_id == envelope.event_id

    @pytest.mark.asyncio
    async def test_get_published_for_topic(self):
        broker = InMemoryBroker()
        await broker.start()

        envelope1 = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        envelope2 = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {"image_id": "img_001", "model_version": "v1", "objects": []},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope1)
        await broker.publish(Topics.INFERENCE_COMPLETED, envelope2)

        image_messages = broker.get_published_for_topic(Topics.IMAGE_SUBMITTED)
        assert len(image_messages) == 1
        assert image_messages[0].event_id == envelope1.event_id

    @pytest.mark.asyncio
    async def test_clear_published(self):
        broker = InMemoryBroker()
        await broker.start()

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)
        broker.clear_published()

        assert len(broker.published_messages) == 0


class TestInMemoryBrokerSubscribe:
    """Tests for subscribe functionality."""

    @pytest.mark.asyncio
    async def test_handler_receives_message(self):
        broker = InMemoryBroker()
        await broker.start()

        received = []

        async def handler(envelope: EventEnvelope):
            received.append(envelope)

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        assert len(received) == 1
        assert received[0].event_id == envelope.event_id

    @pytest.mark.asyncio
    async def test_multiple_handlers_receive_message(self):
        broker = InMemoryBroker()
        await broker.start()

        received1 = []
        received2 = []

        async def handler1(envelope: EventEnvelope):
            received1.append(envelope)

        async def handler2(envelope: EventEnvelope):
            received2.append(envelope)

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler1)
        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler2)

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_handler_only_receives_subscribed_topic(self):
        broker = InMemoryBroker()
        await broker.start()

        received = []

        async def handler(envelope: EventEnvelope):
            received.append(envelope)

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)

        # Publish to different topic
        envelope = create_envelope(
            Topics.INFERENCE_COMPLETED,
            {"image_id": "img_001", "model_version": "v1", "objects": []},
        )
        await broker.publish(Topics.INFERENCE_COMPLETED, envelope)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_get_handler_count(self):
        broker = InMemoryBroker()

        async def handler(envelope: EventEnvelope):
            pass

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)
        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)

        assert broker.get_handler_count(Topics.IMAGE_SUBMITTED) == 2
        assert broker.get_handler_count(Topics.INFERENCE_COMPLETED) == 0


class TestInMemoryBrokerUnsubscribe:
    """Tests for unsubscribe functionality."""

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_handlers(self):
        broker = InMemoryBroker()
        await broker.start()

        received = []

        async def handler(envelope: EventEnvelope):
            received.append(envelope)

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)
        await broker.unsubscribe(Topics.IMAGE_SUBMITTED)

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_topic_no_error(self):
        broker = InMemoryBroker()
        # Should not raise
        await broker.unsubscribe(Topics.IMAGE_SUBMITTED)


class TestInMemoryBrokerErrorHandling:
    """Tests for error handling in handlers."""

    @pytest.mark.asyncio
    async def test_handler_error_does_not_stop_other_handlers(self):
        broker = InMemoryBroker()
        await broker.start()

        received = []

        async def bad_handler(envelope: EventEnvelope):
            raise ValueError("Handler error")

        async def good_handler(envelope: EventEnvelope):
            received.append(envelope)

        await broker.subscribe(Topics.IMAGE_SUBMITTED, bad_handler)
        await broker.subscribe(Topics.IMAGE_SUBMITTED, good_handler)

        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        # Good handler should still receive message
        assert len(received) == 1


class TestInMemoryBrokerReset:
    """Tests for reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_all_state(self):
        broker = InMemoryBroker()
        await broker.start()

        async def handler(envelope: EventEnvelope):
            pass

        await broker.subscribe(Topics.IMAGE_SUBMITTED, handler)
        envelope = create_envelope(
            Topics.IMAGE_SUBMITTED,
            {"image_id": "img_001", "path": "/test.jpg", "source": "cam"},
        )
        await broker.publish(Topics.IMAGE_SUBMITTED, envelope)

        broker.reset()

        assert broker.is_running is False
        assert len(broker.published_messages) == 0
        assert broker.get_handler_count(Topics.IMAGE_SUBMITTED) == 0
