"""Redis-based session memory store."""

from __future__ import annotations

import json
from typing import List, Tuple

import redis.asyncio as aioredis
from pydantic import AnyUrl

import structlog

logger = structlog.get_logger(__name__)


class MemoryStore:
    """Redis-backed session memory for conversation history."""

    def __init__(self, redis_url: AnyUrl, ttl_seconds: int = 3600):
        """
        Initialize memory store.

        Args:
            redis_url: Redis connection URL
            ttl_seconds: Time-to-live for session keys (default 1 hour)
        """
        self.redis_url = str(redis_url)
        self.ttl_seconds = ttl_seconds
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._client is None:
            self._client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("memory.connected", url=self.redis_url)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("memory.disconnected")

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"barista:session:{session_id}"

    async def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Retrieve conversation history for a session.

        Returns list of (human_message, ai_response) tuples.
        """
        if not self._client:
            await self.connect()

        key = self._session_key(session_id)
        data = await self._client.get(key)

        if data:
            try:
                history = json.loads(data)
                return [(h, a) for h, a in history]
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("memory.invalid_data", session_id=session_id)
                return []
        return []

    async def append_message(self, session_id: str, human_message: str, ai_response: str) -> None:
        """Append a message exchange to session history."""
        if not self._client:
            await self.connect()

        key = self._session_key(session_id)
        history = await self.get_history(session_id)
        history.append((human_message, ai_response))

        # Keep only last 20 exchanges
        history = history[-20:]

        await self._client.setex(
            key,
            self.ttl_seconds,
            json.dumps(history),
        )
        logger.debug("memory.appended", session_id=session_id, history_len=len(history))

    async def clear_session(self, session_id: str) -> None:
        """Clear session history."""
        if not self._client:
            await self.connect()

        key = self._session_key(session_id)
        await self._client.delete(key)
        logger.info("memory.cleared", session_id=session_id)

