"""Unit tests for ChatRepository's session ownership rules.

These assert on the SQL that is actually sent, not on a Python-side
check after the fetch. That distinction is the whole point: the bug
these cover was a `SELECT ... WHERE id = :id` with the ownership test
missing entirely, so anyone holding another person's `conversation_id`
got that person's session — and their history — back.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.chat import ChatSession
from app.repositories.chat_repository import ChatRepository


def _repo_capturing_sql(found: ChatSession | None):
    """A ChatRepository whose session records every statement it runs."""
    statements: list[str] = []
    session = AsyncMock()

    async def execute(stmt, *args, **kwargs):
        statements.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        result = MagicMock()
        result.scalar_one_or_none.return_value = found
        return result

    async def scalar(stmt, *args, **kwargs):
        statements.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        return None

    session.execute = execute
    session.scalar = scalar
    session.add = MagicMock()
    return ChatRepository(session), statements


@pytest.mark.asyncio
async def test_lookup_filters_by_user_id_in_sql() -> None:
    session_id = uuid.uuid4()
    repo, statements = _repo_capturing_sql(found=None)

    await repo.get_or_create_session(session_id, user_id="user-a")

    lookup = statements[0]
    assert "user_id" in lookup, "ownership must be in the WHERE clause, not checked in Python"
    assert "user-a" in lookup


@pytest.mark.asyncio
async def test_anonymous_caller_only_matches_unowned_sessions() -> None:
    session_id = uuid.uuid4()
    repo, statements = _repo_capturing_sql(found=None)

    await repo.get_or_create_session(session_id, user_id=None)

    assert "user_id IS NULL" in statements[0]


@pytest.mark.asyncio
async def test_another_users_session_id_yields_a_different_session() -> None:
    """The id is already taken, so the caller gets a fresh one rather than
    the other user's conversation — and rather than a primary key error."""
    session_id = uuid.uuid4()
    session = AsyncMock()

    async def execute(stmt, *args, **kwargs):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None  # not owned by this caller
        return result

    async def scalar(stmt, *args, **kwargs):
        return session_id  # but the row does exist

    session.execute = execute
    session.scalar = scalar
    session.add = MagicMock()

    created = await ChatRepository(session).get_or_create_session(session_id, user_id="user-b")

    assert created.id != session_id
    assert created.user_id == "user-b"


@pytest.mark.asyncio
async def test_unowned_legacy_session_is_adopted() -> None:
    """Rows predating the user_id column have NULL and belong to whoever
    claims them first — otherwise every existing conversation breaks."""
    session_id = uuid.uuid4()
    legacy = ChatSession(id=session_id, user_id=None)
    repo, _ = _repo_capturing_sql(found=legacy)

    returned = await repo.get_or_create_session(session_id, user_id="user-a")

    assert returned is legacy
    assert returned.user_id == "user-a"


@pytest.mark.asyncio
async def test_history_breaks_timestamp_ties_on_role_not_random_id() -> None:
    """The user message and the assistant reply of one turn are inserted
    in the same transaction, so PostgreSQL's `now()` gives them an
    identical `created_at`. Tiebreaking on `id` (a random uuid4) made the
    pair's order a coin flip — measured at 64 of 121 turns inverted on
    the live database, which handed the model the answer before the
    question. Redis was unaffected, so this only bit on a cache miss.
    """
    session = AsyncMock()
    captured: list[str] = []

    async def execute(stmt, *args, **kwargs):
        captured.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        return result

    session.execute = execute

    await ChatRepository(session).get_recent_messages(uuid.uuid4(), limit=6)

    sql = captured[0]
    order_by = sql[sql.index("ORDER BY"):]
    assert "CASE" in order_by, "role must break the created_at tie"
    # The role rank has to come BEFORE the random id, or the id decides.
    assert order_by.index("CASE") < order_by.index("messages.id"), order_by
