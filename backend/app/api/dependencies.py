"""Shared FastAPI dependency providers."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db_session
from app.llm.base import LLMClient, StubLLMClient
from app.services.chat_service import ChatService


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for the duration of a request."""
    async for session in get_db_session():
        yield session


def get_llm_client() -> LLMClient:
    """Provide the configured LLM client implementation.

    Returns a stub implementation until a real provider is wired in.
    """
    return StubLLMClient()


def get_chat_service() -> ChatService:
    """Provide a ChatService instance with its dependencies injected."""
    return ChatService(llm_client=get_llm_client())
