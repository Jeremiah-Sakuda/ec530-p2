"""In-memory broker implementation for testing."""

import asyncio
import logging
from typing import Any
from collections import defaultdict

from shared.events import EventEnvelope
from .base import BaseBroker, MessageHandler

logger = logging.getLogger(__name__)


class InMemoryBroker(BaseBroker):
    """
    In-memory pub-sub broker for unit tests.

    Features:
    - Synchronous message delivery for deterministic tests
    - Records all published messages for assertions
    - No external dependencies
    """

    def __init__(self):
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._published: list[tuple[str, EventEnvelope]] = []
        self._running = False

    async def publish(self, topic: str, envelope: EventEnvelope) -> None:
        """
        Publish an event to a topic.

        Messages are delivered synchronously to all handlers for the topic.
        All published messages are recorded for test assertions.
        """
        if not self._running:
            logger.warning(f"Broker not running, but publishing to {topic}")

        # Record the published message
        self._published.append((topic, envelope))
        logger.debug(f"Published to {topic}: {envelope.event_id}")

        # Deliver to all handlers for this topic
        handlers = self._handlers.get(topic, [])
        for handler in handlers:
            try:
                await handler(envelope)
            except Exception as e:
                logger.error(f"Handler error for {topic}: {e}")

    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe a handler to a topic."""
        self._handlers[topic].append(handler)
        logger.debug(f"Subscribed to {topic}")

    async def unsubscribe(self, topic: str) -> None:
        """Remove all handlers for a topic."""
        if topic in self._handlers:
            del self._handlers[topic]
            logger.debug(f"Unsubscribed from {topic}")

    async def start(self) -> None:
        """Start the broker."""
        self._running = True
        logger.debug("In-memory broker started")

    async def stop(self) -> None:
        """Stop the broker."""
        self._running = False
        logger.debug("In-memory broker stopped")

    @property
    def is_running(self) -> bool:
        """Return whether the broker is running."""
        return self._running

    # ========================================================================
    # Test helpers
    # ========================================================================

    @property
    def published_messages(self) -> list[tuple[str, EventEnvelope]]:
        """Return all published messages as (topic, envelope) tuples."""
        return self._published.copy()

    def get_published_for_topic(self, topic: str) -> list[EventEnvelope]:
        """Return all envelopes published to a specific topic."""
        return [env for t, env in self._published if t == topic]

    def clear_published(self) -> None:
        """Clear the published messages list."""
        self._published.clear()

    def get_handler_count(self, topic: str) -> int:
        """Return number of handlers registered for a topic."""
        return len(self._handlers.get(topic, []))

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._handlers.clear()
        self._published.clear()
        self._running = False
