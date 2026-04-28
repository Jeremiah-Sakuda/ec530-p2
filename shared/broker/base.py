"""Base broker abstraction for pub-sub messaging."""

from abc import ABC, abstractmethod
from typing import Callable, Awaitable, Any

from shared.events import EventEnvelope


# Type alias for message handlers
MessageHandler = Callable[[EventEnvelope], Awaitable[None]]


class BaseBroker(ABC):
    """
    Abstract base class for pub-sub message brokers.

    Implementations must provide publish/subscribe functionality.
    Services use this abstraction to avoid coupling to a specific broker.
    """

    @abstractmethod
    async def publish(self, topic: str, envelope: EventEnvelope) -> None:
        """
        Publish an event to a topic.

        Args:
            topic: Topic name to publish to
            envelope: Event envelope to publish
        """
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Subscribe to a topic with a handler function.

        Args:
            topic: Topic name to subscribe to
            handler: Async function to call when message is received
        """
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a topic.

        Args:
            topic: Topic name to unsubscribe from
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the broker connection and begin processing messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the broker connection and cleanup resources."""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the broker is currently running."""
        pass
