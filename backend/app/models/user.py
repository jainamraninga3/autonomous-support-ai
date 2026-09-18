"""Login accounts and their active sessions.

`User.password` is stored hashed (bcrypt, via `app/services/auth_service.py`)
— never plaintext, regardless of what called this table.

`UserSession` is the source of truth for "who is logged in right now"; it is
a real table, not a Redis key, precisely because Redis has already been
observed to come up disconnected on a container-startup race in this
project (see docs/PROJECT_LOG.md) and a login system cannot silently forget
every logged-in user when that happens.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """A login account. `role` is `"admin"` or `"user"` — checked in code,
    not a database enum, so adding a role never needs a migration."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    sessions: Mapped[list["UserSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserSession(Base):
    """One logged-in session. `token` is the opaque bearer value the client
    holds; looking it up here is the entire authentication check.

    No `TimestampMixin` — `created_at` here means "session started", and an
    `updated_at` with no other columns that ever change would be dead
    weight. `expires_at` is advanced directly by `AuthService` on each
    authenticated request (sliding expiry).
    """

    __tablename__ = "user_sessions"

    token: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    user: Mapped["User"] = relationship(back_populates="sessions")
