"""Repository for chat sessions and messages."""

import uuid

from sqlalchemy import select

from app.models.chat import ChatSession, Message
from app.repositories.base import BaseRepository


class ChatRepository(BaseRepository[ChatSession]):
    """Persists chat sessions and messages."""

    async def get_or_create_session(self, session_id: uuid.UUID | None) -> ChatSession:
        if session_id is not None:
            result = await self.session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                return existing

        chat_session = ChatSession(id=session_id or uuid.uuid4())
        self.session.add(chat_session)
        await self.session.flush()
        return chat_session

    async def add_message(self, session_id: uuid.UUID, role: str, content: str) -> Message:
        message = Message(session_id=session_id, role=role, content=content)
        self.session.add(message)
        await self.session.flush()
        return message
