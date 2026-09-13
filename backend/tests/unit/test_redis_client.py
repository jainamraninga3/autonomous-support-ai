"""Unit tests for low-level Redis client functions."""

import pytest
from app.infrastructure.redis.client import close_redis_client, ping_redis


@pytest.mark.asyncio
async def test_ping_redis_with_none() -> None:
    assert await ping_redis(None) is False


@pytest.mark.asyncio
async def test_close_redis_client_with_none() -> None:
    # Should not raise any exception
    await close_redis_client(None)
