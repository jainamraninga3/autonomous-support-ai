"""Chat session and message models.

**Reconstructed 2026-09-07** — see `base.py` for why these files were
missing from git.

Single-user by design: there is no owner column on either table. Chat
sessions and messages belong to whoever is using the app. Matches
migration 0001 exactly, which is the schema the database actually has.

Note `messages.session_id` has NO index. That is deliberate and matches
0001: the table is only ever read by `session_id` through a small
`LIMIT 6` query for conversation history, and PostgreSQL will use the
foreign key's implicit constraint check plus a sequential scan happily
at this size. Adding `index=True` here without a matching migration
would make `alembic check` report drift.
"""

import uuid

from sqlalchemy import ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    """One conversation with the assistant."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base, TimestampMixin):
    """A single message within a chat session.

    `ondelete="CASCADE"` puts the deletion in the database, so removing a
    session removes its messages in one statement rather than an
    application sweep that could fail halfway and leave orphans.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
