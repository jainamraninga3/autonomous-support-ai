"""Unit tests for SessionMemoryService."""

import uuid
from unittest.mock import AsyncMock

import pytest
import fakeredis.aioredis

from app.models.chat import Message
from app.services.session_memory_service import SessionMemoryService


@pytest.mark.asyncio
async def test_session_memory_service_cache_miss_queries_pg_and_rebuilds() -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()
    mock_repo = AsyncMock()

    session_id = uuid.uuid4()
    pg_msg1 = Message(session_id=session_id, role="user", content="Question")
    pg_msg2 = Message(session_id=session_id, role="assistant", content="Answer")
    mock_repo.get_recent_messages.return_value = [pg_msg1, pg_msg2]

    service = SessionMemoryService(chat_repository=mock_repo, redis_client=fake_redis)

    # First call: Redis empty, hits PG
    history = await service.get_recent_history(session_id, user_id="user-1")
    assert history == [("user", "Question"), ("assistant", "Answer")]
    mock_repo.get_recent_messages.assert_called_once()

    # Second call: Redis now populated by rebuild, returns from cache without calling PG again
    mock_repo.get_recent_messages.reset_mock()
    history_cached = await service.get_recent_history(session_id, user_id="user-1")
    assert history_cached == [("user", "Question"), ("assistant", "Answer")]
    mock_repo.get_recent_messages.assert_not_called()

    await fake_redis.aclose()


@pytest.mark.asyncio
async def test_session_memory_service_redis_down_graceful_pg_fallback() -> None:
    mock_repo = AsyncMock()
    session_id = uuid.uuid4()
    pg_msg = Message(session_id=session_id, role="user", content="Hi")
    mock_repo.get_recent_messages.return_value = [pg_msg]

    # Redis is None (down)
    service = SessionMemoryService(chat_repository=mock_repo, redis_client=None)

    history = await service.get_recent_history(session_id, user_id="user-1")
    assert history == [("user", "Hi")]
    mock_repo.get_recent_messages.assert_called_once()
