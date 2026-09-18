"""Data access for `User` and `UserSession`."""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, UserSession
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Session is committed by the caller (`AuthService`), matching the
    pattern `ChatRepository` already uses — a route/service decides
    transaction boundaries, not the repository."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_username(self, username: str) -> User | None:
        result = await self.session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create_user(
        self, username: str, hashed_password: str, full_name: str, role: str
    ) -> User:
        user = User(username=username, password=hashed_password, full_name=full_name, role=role)
        self.session.add(user)
        await self.session.flush()
        return user

    async def create_session(self, user_id: uuid.UUID, token: str, ttl_seconds: int) -> UserSession:
        now = datetime.now(timezone.utc)
        # Opportunistic cleanup of this user's own expired sessions on every
        # new login — cheap, and avoids ever needing a cron job just to keep
        # `user_sessions` from growing forever.
        await self.session.execute(
            delete(UserSession).where(UserSession.user_id == user_id, UserSession.expires_at < now)
        )
        session_row = UserSession(
            token=token,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self.session.add(session_row)
        await self.session.flush()
        return session_row

    async def get_session_with_user(self, token: str) -> tuple[UserSession, User] | None:
        result = await self.session.execute(
            select(UserSession, User).join(User, UserSession.user_id == User.id).where(
                UserSession.token == token
            )
        )
        row = result.first()
        return (row[0], row[1]) if row else None

    async def touch_session(self, token: str, ttl_seconds: int) -> None:
        """Slide the session's expiry forward — an active user stays logged in."""
        session_row = await self.session.get(UserSession, token)
        if session_row is not None:
            session_row.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            await self.session.flush()

    async def delete_session(self, token: str) -> None:
        await self.session.execute(delete(UserSession).where(UserSession.token == token))
