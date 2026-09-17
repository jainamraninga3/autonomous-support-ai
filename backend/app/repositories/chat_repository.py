"""Repository for chat sessions and messages."""

import uuid

from sqlalchemy import case, or_, select

from app.models.chat import ChatSession, Message
from app.repositories.base import BaseRepository


# Within one turn the user message and the assistant reply are INSERTed
# in the same transaction, and `created_at` defaults to PostgreSQL's
# `now()`, which returns the TRANSACTION start time — so both rows get a
# byte-identical timestamp. The tiebreaker used to be `id`, a random
# uuid4, which made the order of the pair a coin flip: measured on the
# live database, 64 of 121 turns came back with the assistant's answer
# BEFORE the question it answered. Redis was unaffected (RPUSH keeps
# insertion order), so the cache was right and the source of truth was
# wrong — and the scrambling only showed on a cache miss, a Redis
# outage, or after the TTL.
#
# A tie can only be a single turn's pair, since one request is one
# transaction, and within a pair the user always spoke first. Ranking on
# role is therefore exact and needs no migration. The alternative — a
# `clock_timestamp()` server default, or a monotonic sequence column —
# is a schema change, and this is not worth one.
_ROLE_ORDER = case((Message.role == "user", 0), else_=1)


class ChatRepository(BaseRepository[ChatSession]):
    """Persists chat sessions and messages."""

    async def get_or_create_session(
        self, session_id: uuid.UUID | None, user_id: str | None = None
    ) -> ChatSession:
        """Return the caller's session for `session_id`, or create one.

        OWNERSHIP IS ENFORCED IN SQL, in the same WHERE clause that looks
        the session up — never by fetching the row and comparing
        `user_id` in Python. This used to select on `id` alone, which
        meant anyone who knew (or guessed) another person's
        `conversation_id` got that person's `ChatSession` back, and
        `get_recent_messages` then fed their conversation into the
        prompt. Redis hid it — its key is namespaced by user — so the
        leak only showed on a cache miss, a Redis outage, or after the
        TTL expired.

        Sessions with `user_id IS NULL` predate the column and are
        adopted by the first caller who claims them, which is what the
        `or_` covers.

        A `session_id` that exists but belongs to someone else is NOT an
        error: the caller is quietly given a brand-new session under a
        new id (the id they sent is already taken, so reusing it would
        collide on the primary key). The new id comes back in the
        response as `conversation_id`.
        """
        if session_id is not None:
            owned = or_(ChatSession.user_id == user_id, ChatSession.user_id.is_(None)) if user_id else ChatSession.user_id.is_(None)
            result = await self.session.execute(
                select(ChatSession).where(ChatSession.id == session_id, owned)
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                if user_id and not existing.user_id:
                    existing.user_id = user_id
                    await self.session.flush()
                return existing

            # Not ours (or not there). Only reuse the requested id if no
            # row holds it — otherwise INSERT would raise on the PK.
            taken = await self.session.scalar(
                select(ChatSession.id).where(ChatSession.id == session_id)
            )
            new_id = uuid.uuid4() if taken else session_id
        else:
            new_id = uuid.uuid4()

        chat_session = ChatSession(id=new_id, user_id=user_id)
        self.session.add(chat_session)
        await self.session.flush()
        return chat_session

    async def get_recent_messages(self, session_id: uuid.UUID, limit: int = 6) -> list[Message]:
        """Return the last `limit` messages for a session, oldest first.

        Used to resolve follow-up questions. Without this the graph saw
        only the current message, so "where is it located?" had no
        antecedent and was classified as unrelated to the company.

        Deliberately a small window: history is used to interpret the
        question, not as context to answer from. Feeding in long history
        would also start polluting retrieval with terms from earlier
        turns.
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc(), _ROLE_ORDER.desc(), Message.id.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def add_message(self, session_id: uuid.UUID, role: str, content: str) -> Message:
        message = Message(session_id=session_id, role=role, content=content)
        self.session.add(message)
        await self.session.flush()
        return message
