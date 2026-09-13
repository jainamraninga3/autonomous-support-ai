"""Unit tests for RedisSessionStore."""

import uuid
import pytest
import fakeredis.aioredis

from app.infrastructure.redis.session_store import RedisSessionStore


@pytest.mark.asyncio
async def test_redis_session_store_key_schema() -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()
    store = RedisSessionStore(fake_redis)

    session_id = uuid.uuid4()
    key_user = store.build_key("user-123", session_id)
    assert key_user == f"session:user-123:{session_id}"

    key_anon = store.build_key(None, session_id)
    assert key_anon == f"session:anonymous:{session_id}"

    await fake_redis.aclose()


@pytest.mark.asyncio
async def test_redis_session_store_push_and_get() -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60, max_messages=10)

    session_id = uuid.uuid4()
    user_id = "user-abc"

    # Initially cache miss
    history = await store.get_history(user_id, session_id)
    assert history is None

    # Push pair
    success = await store.push_message_pair(user_id, session_id, "Hello", "Hi there!")
    assert success is True

    # Get history
    history = await store.get_history(user_id, session_id)
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    await fake_redis.aclose()


@pytest.mark.asyncio
async def test_redis_session_store_failsafe_when_redis_none() -> None:
    # Pass None as redis client (e.g. Redis connection down)
    store = RedisSessionStore(None)
    session_id = uuid.uuid4()

    assert await store.get_history("user-1", session_id) is None
    assert await store.push_message_pair("user-1", session_id, "q", "a") is False
    assert await store.rebuild_from_messages("user-1", session_id, []) is False
    assert await store.delete_session("user-1", session_id) is False


@pytest.mark.asyncio
async def test_redis_session_store_rebuild() -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()
    store = RedisSessionStore(fake_redis)

    session_id = uuid.uuid4()
    messages = [
        {"role": "user", "content": "Msg 1"},
        {"role": "assistant", "content": "Msg 2"},
    ]

    ok = await store.rebuild_from_messages("user-x", session_id, messages)
    assert ok is True

    history = await store.get_history("user-x", session_id)
    assert history == messages

    await fake_redis.aclose()
