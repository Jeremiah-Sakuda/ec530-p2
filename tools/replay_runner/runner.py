"""Replay runner for replaying events from JSONL files."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from shared.events import EventEnvelope
from shared.broker import BaseBroker

logger = logging.getLogger(__name__)


class ReplayRunner:
    """
    Replays events from JSONL files through a broker.

    Used to reproduce bug states and drive load tests with
    recorded event sequences.
    """

    def __init__(self, broker: BaseBroker):
        """
        Initialize the replay runner.

        Args:
            broker: Broker to publish events through
        """
        self._broker = broker

    async def replay_from_file(
        self,
        path: str | Path,
        interval_ms: int = 100,
        skip_invalid: bool = True,
    ) -> int:
        """
        Replay events from a JSONL file.

        Args:
            path: Path to JSONL file containing events
            interval_ms: Milliseconds to wait between events
            skip_invalid: If True, skip invalid events; if False, raise on invalid

        Returns:
            Number of events successfully replayed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Event file not found: {path}")

        replayed = 0
        interval_sec = interval_ms / 1000.0

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    envelope = EventEnvelope.from_dict(data)
                except json.JSONDecodeError as e:
                    if skip_invalid:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue
                    raise ValueError(f"Line {line_num}: Invalid JSON - {e}")
                except (KeyError, TypeError) as e:
                    if skip_invalid:
                        logger.warning(f"Line {line_num}: Invalid envelope - {e}")
                        continue
                    raise ValueError(f"Line {line_num}: Invalid envelope - {e}")

                # Publish the event
                await self._broker.publish(envelope.topic, envelope)
                replayed += 1
                logger.debug(f"Replayed event {envelope.event_id} to {envelope.topic}")

                # Wait before next event
                if interval_ms > 0:
                    await asyncio.sleep(interval_sec)

        logger.info(f"Replayed {replayed} events from {path}")
        return replayed

    async def replay_events(
        self,
        events: list[EventEnvelope],
        interval_ms: int = 100,
    ) -> int:
        """
        Replay a list of events through the broker.

        Args:
            events: List of EventEnvelope objects to replay
            interval_ms: Milliseconds to wait between events

        Returns:
            Number of events replayed
        """
        interval_sec = interval_ms / 1000.0

        for i, envelope in enumerate(events):
            await self._broker.publish(envelope.topic, envelope)
            logger.debug(f"Replayed event {envelope.event_id} to {envelope.topic}")

            if interval_ms > 0 and i < len(events) - 1:
                await asyncio.sleep(interval_sec)

        logger.info(f"Replayed {len(events)} events")
        return len(events)

    @staticmethod
    def save_to_file(
        events: list[EventEnvelope],
        path: str | Path,
        append: bool = False,
    ) -> int:
        """
        Save events to a JSONL file.

        Args:
            events: List of EventEnvelope objects to save
            path: Path to output JSONL file
            append: If True, append to existing file; if False, overwrite

        Returns:
            Number of events saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for envelope in events:
                f.write(envelope.to_json() + "\n")

        logger.info(f"Saved {len(events)} events to {path}")
        return len(events)

    @staticmethod
    def load_from_file(path: str | Path) -> list[EventEnvelope]:
        """
        Load events from a JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of EventEnvelope objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file contains invalid data
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Event file not found: {path}")

        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    envelope = EventEnvelope.from_dict(data)
                    events.append(envelope)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    raise ValueError(f"Line {line_num}: Invalid event - {e}")

        return events
