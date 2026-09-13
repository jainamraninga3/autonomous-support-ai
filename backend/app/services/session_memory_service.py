"""Session memory service orchestrating Redis cache and PostgreSQL sync."""

import uuid
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.infrastructure.redis.session_store import RedisSessionStore
from app.repositories.chat_repository import ChatRepository
from app.services.session_context_builder import SessionContextBuilder

logger = get_logger(__name__)


class SessionMemoryService:
    """Manages hot session memory in Redis and falls back to PostgreSQL on cache miss or error."""

    def __init__(
        self,
        chat_repository: ChatRepository,
        redis_client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.chat_repository = chat_repository
        self.enabled = settings.SESSION_MEMORY_ENABLED
        self.store = RedisSessionStore(redis_client) if self.enabled else None

    async def get_recent_history(
        self,
        session_id: uuid.UUID,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        """Retrieve recent conversation history for a session.

        1. If Redis is enabled, attempts to fetch history from Redis.
        2. On cache HIT: returns formatted tuples.
        3. On cache MISS or Redis error: fetches history from PostgreSQL,
           rebuilds the Redis cache for subsequent turns, and returns formatted tuples.
        """
        fetch_limit = limit or self.store.max_messages if self.store else 6

        if self.enabled and self.store is not None:
            cached_messages = await self.store.get_history(user_id, session_id, limit=fetch_limit)
            if cached_messages is not None:
                logger.debug(
                    "Session memory HIT for user_id=%s session_id=%s (messages=%d)",
                    user_id,
                    session_id,
                    len(cached_messages),
                )
                return SessionContextBuilder.build_history_tuples(cached_messages)
            logger.debug(
                "Session memory MISS for user_id=%s session_id=%s — querying PostgreSQL",
                user_id,
                session_id,
            )

        # PostgreSQL fallback (source of truth)
        pg_messages = await self.chat_repository.get_recent_messages(session_id, limit=fetch_limit)
        history_tuples = SessionContextBuilder.build_history_tuples(pg_messages)

        # Rebuild Redis cache asynchronously on cache miss
        if self.enabled and self.store is not None and pg_messages:
            formatted_dicts = [{"role": msg.role, "content": msg.content} for msg in pg_messages]
            await self.store.rebuild_from_messages(user_id, session_id, formatted_dicts)

        return history_tuples

    async def append_message_pair(
        self,
        session_id: uuid.UUID,
        user_id: str | None,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Push a user & assistant message pair into Redis hot memory."""
        if not self.enabled or self.store is None:
            return

        await self.store.push_message_pair(user_id, session_id, user_content, assistant_content)
