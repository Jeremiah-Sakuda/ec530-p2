"""Redis pub-sub broker implementation."""

import asyncio
import logging
from typing import Optional

import redis.asyncio as redis

from shared.events import EventEnvelope
from .base import BaseBroker, MessageHandler

logger = logging.getLogger(__name__)


class RedisBroker(BaseBroker):
    """
    Redis pub-sub broker implementation.

    Uses redis-py async client for pub-sub messaging.
    Handles connection pooling, reconnection, and message serialization.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize the Redis broker.

        Args:
            redis_url: Redis connection URL
        """
        self._redis_url = redis_url
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._running = False
        self._listener_task: Optional[asyncio.Task] = None

    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is established."""
        if self._client is None:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._pubsub = self._client.pubsub()
            logger.info(f"Connected to Redis at {self._redis_url}")

    async def publish(self, topic: str, envelope: EventEnvelope) -> None:
        """
        Publish an event to a Redis channel.

        Args:
            topic: Topic/channel name to publish to
            envelope: Event envelope to publish
        """
        await self._ensure_connected()

        message = envelope.to_json()
        await self._client.publish(topic, message)
        logger.debug(f"Published to {topic}: {envelope.event_id}")

    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Subscribe to a Redis channel with a handler.

        Args:
            topic: Topic/channel name to subscribe to
            handler: Async function to call when message is received
        """
        await self._ensure_connected()

        if topic not in self._handlers:
            self._handlers[topic] = []
            await self._pubsub.subscribe(topic)
            logger.debug(f"Subscribed to Redis channel: {topic}")

        self._handlers[topic].append(handler)

    async def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a Redis channel.

        Args:
            topic: Topic/channel name to unsubscribe from
        """
        if self._pubsub and topic in self._handlers:
            await self._pubsub.unsubscribe(topic)
            del self._handlers[topic]
            logger.debug(f"Unsubscribed from Redis channel: {topic}")

    async def start(self) -> None:
        """Start the broker and begin listening for messages."""
        await self._ensure_connected()
        self._running = True

        # Start the listener task
        self._listener_task = asyncio.create_task(self._listen())
        logger.info("Redis broker started")

    async def stop(self) -> None:
        """Stop the broker and cleanup resources."""
        self._running = False

        # Cancel listener task
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        # Close pubsub and client
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

        if self._client:
            await self._client.close()
            self._client = None

        logger.info("Redis broker stopped")

    @property
    def is_running(self) -> bool:
        """Return whether the broker is currently running."""
        return self._running

    async def _listen(self) -> None:
        """Listen for messages on subscribed channels."""
        logger.debug("Starting Redis message listener")

        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message is None:
                    continue

                if message["type"] != "message":
                    continue

                topic = message["channel"]
                data = message["data"]

                # Parse the envelope
                try:
                    envelope = EventEnvelope.from_json(data)
                except Exception as e:
                    logger.error(f"Failed to parse message on {topic}: {e}")
                    continue

                # Dispatch to handlers
                handlers = self._handlers.get(topic, [])
                for handler in handlers:
                    try:
                        await handler(envelope)
                    except Exception as e:
                        logger.error(f"Handler error for {topic}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Listener error: {e}")
                if self._running:
                    await asyncio.sleep(1)  # Brief pause before retry

        logger.debug("Redis message listener stopped")
