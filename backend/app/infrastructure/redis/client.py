"""Redis async client — opened once in app lifespan, shared across requests.

`create_redis_client()` is called during FastAPI startup and the result
stored in `app.state.redis_client`. If Redis is unreachable at startup the
function returns `None` and the rest of the app degrades gracefully: every
call that would use Redis falls back to PostgreSQL instead.

`ping_redis()` is the health-check primitive — it never raises.
"""

from typing import Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def create_redis_client() -> Optional[aioredis.Redis]:
    """Open an async Redis connection pool.

    Returns the client on success, or ``None`` if Redis is unreachable or
    session memory is disabled via ``SESSION_MEMORY_ENABLED=false``.
    The caller must never treat a ``None`` return as an error.
    """
    settings = get_settings()

    if not settings.SESSION_MEMORY_ENABLED:
        logger.info("Session memory disabled (SESSION_MEMORY_ENABLED=false) — Redis client not created")
        return None

    try:
        client: aioredis.Redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=1.0,
            # Connection pool limits: one pool shared across all requests.
            max_connections=20,
        )
        # Verify the connection is actually usable before reporting success.
        await client.ping()
        logger.info("Redis connected: %s", settings.REDIS_URL)
        return client
    except Exception:
        logger.warning(
            "Redis unavailable at startup — session memory will fall back to PostgreSQL for every request",
            exc_info=True,
        )
        return None


async def close_redis_client(client: Optional[aioredis.Redis]) -> None:
    """Gracefully close the connection pool. Safe to call with ``None``."""
    if client is not None:
        try:
            await client.aclose()
            logger.info("Redis connection closed")
        except Exception:
            logger.warning("Error closing Redis connection", exc_info=True)


async def ping_redis(client: Optional[aioredis.Redis]) -> bool:
    """Return True if Redis is reachable, False otherwise. Never raises.

    Used by the health-check endpoint and lifespan startup probe.
    """
    if client is None:
        return False
    try:
        return bool(await client.ping())
    except Exception:
        return False
