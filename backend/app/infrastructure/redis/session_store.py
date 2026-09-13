"""Redis session store implementation.

Key schema:
`session:{user_id}:{session_id}` (or `session:anonymous:{session_id}` if user_id is None).

Design principles:
1. Fallback safety: Every Redis operation is wrapped in try/except. If Redis is down,
   unreachable, or raises an error, the store logs a warning and returns `None`/`False`.
   The application falls back to PostgreSQL seamlessly.
2. TTL handling: Every push or rebuild resets the key TTL to `SESSION_MEMORY_TTL_SECONDS`.
3. Multi-user isolation: The key includes `user_id` to enforce strict session boundaries.
"""

import json
import uuid
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisSessionStore:
    """Low-level Redis list store for user session messages."""

    def __init__(
        self,
        redis_client: Any | None,
        ttl_seconds: int | None = None,
        max_messages: int | None = None,
    ) -> None:
        settings = get_settings()
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds or settings.SESSION_MEMORY_TTL_SECONDS
        self.max_messages = max_messages or settings.SESSION_MEMORY_MAX_MESSAGES

    def build_key(self, user_id: str | None, session_id: str | uuid.UUID) -> str:
        """Construct the Redis key using the session:{user_id}:{session_id} schema."""
        user_part = user_id.strip() if user_id and user_id.strip() else "anonymous"
        return f"session:{user_part}:{session_id}"

    async def get_history(
        self, user_id: str | None, session_id: str | uuid.UUID, limit: int | None = None
    ) -> list[dict[str, str]] | None:
        """Retrieve recent message history from Redis.

        Returns:
            - `list[dict[str, str]]` if key exists and read succeeded.
            - `None` on cache miss (key does not exist) or if Redis is unreachable.
        """
        if self.redis is None:
            return None

        key = self.build_key(user_id, session_id)
        fetch_limit = limit or self.max_messages

        try:
            # -fetch_limit to -1 gets the last N elements from the list
            raw_items = await self.redis.lrange(key, -fetch_limit, -1)
            if not raw_items:
                return None

            messages: list[dict[str, str]] = []
            for item in raw_items:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                messages.append(json.loads(item))
            return messages
        except Exception:
            logger.exception("Redis get_history failed for key=%s", key)
            return None

    async def push_message_pair(
        self,
        user_id: str | None,
        session_id: str | uuid.UUID,
        user_content: str,
        assistant_content: str,
    ) -> bool:
        """Push a user message and assistant message pair to Redis in a single pipeline."""
        if self.redis is None:
            return False

        key = self.build_key(user_id, session_id)
        user_msg = json.dumps({"role": "user", "content": user_content})
        assistant_msg = json.dumps({"role": "assistant", "content": assistant_content})

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, user_msg, assistant_msg)
                # Cap the list size to max_messages
                pipe.ltrim(key, -self.max_messages, -1)
                pipe.expire(key, self.ttl_seconds)
                await pipe.execute()
            return True
        except Exception:
            logger.exception("Redis push_message_pair failed for key=%s", key)
            return False

    async def rebuild_from_messages(
        self,
        user_id: str | None,
        session_id: str | uuid.UUID,
        messages: list[dict[str, str]],
    ) -> bool:
        """Rebuild the Redis session list from PostgreSQL source of truth on cache miss."""
        if self.redis is None:
            return False

        key = self.build_key(user_id, session_id)

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                if messages:
                    encoded_msgs = [
                        json.dumps({"role": m["role"], "content": m["content"]}) for m in messages
                    ]
                    pipe.rpush(key, *encoded_msgs)
                    pipe.ltrim(key, -self.max_messages, -1)
                    pipe.expire(key, self.ttl_seconds)
                await pipe.execute()
            return True
        except Exception:
            logger.exception("Redis rebuild_from_messages failed for key=%s", key)
            return False

    async def delete_session(self, user_id: str | None, session_id: str | uuid.UUID) -> bool:
        """Explicitly remove a session from Redis."""
        if self.redis is None:
            return False

        key = self.build_key(user_id, session_id)
        try:
            await self.redis.delete(key)
            return True
        except Exception:
            logger.exception("Redis delete_session failed for key=%s", key)
            return False
