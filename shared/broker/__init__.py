"""Broker abstraction for pub-sub messaging."""

from .base import BaseBroker, MessageHandler
from .in_memory import InMemoryBroker

# RedisBroker requires redis package - make import optional
try:
    from .redis_pubsub import RedisBroker
except ImportError:
    RedisBroker = None  # type: ignore

__all__ = [
    "BaseBroker",
    "MessageHandler",
    "InMemoryBroker",
    "RedisBroker",
]
