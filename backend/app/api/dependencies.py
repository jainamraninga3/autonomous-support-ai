"""Shared FastAPI dependency providers."""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.database.connection import AsyncSessionLocal, get_db_session
from app.graph.workflow import build_graph
from app.llm.base import GroqLLMClient, LLMClient, StubLLMClient
from app.repositories.chat_repository import ChatRepository
from app.services.admin_service import AdminService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService

logger = get_logger(__name__)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for the duration of a request."""
    async for session in get_db_session():
        yield session


@lru_cache
def get_llm_client() -> LLMClient:
    """Provide the configured LLM client implementation.

    Uses Groq when at least one of `GROQ_API_KEY`/`GROQ_API_1`/`_2`/`_3`
    is set (falling through to the next configured key on a rate limit
    or a rejected key — see `GroqLLMClient`); falls back to the echo stub
    otherwise (e.g. local dev without a key, or tests) so the app still
    boots and the chat endpoint stays usable without external calls.

    **`lru_cache` is what makes `GroqLLMClient`'s key stickiness real.**
    Without it FastAPI built a fresh client per request, so
    `_current_index` reset to 0 every time and a revoked primary key cost
    a wasted round trip on EVERY chat rather than one per process.
    Measured 2026-09-08: a dead `GROQ_API_KEY` in front of three working
    keys produced one 401 per request, indefinitely.

    It also shares the underlying httpx connection pool across requests,
    which is what you want from an HTTP client anyway.

    The cost is that a key change in `.env` needs a process restart to
    take effect. That is already true of `get_settings()`, which is
    `lru_cache`d for the same reason, so this adds no new surprise.
    """
    settings = get_settings()
    keys = settings.groq_api_keys
    if keys:
        return GroqLLMClient(api_keys=keys, model=settings.GROQ_MODEL)
    logger.warning("No Groq API key set — falling back to StubLLMClient (echo)")
    return StubLLMClient()


from app.services.session_memory_service import SessionMemoryService


def get_redis_client(request: Request) -> Any | None:
    """Provide the application-wide async Redis client stored on app.state."""
    return getattr(request.app.state, "redis_client", None)


def get_chat_repository(session: AsyncSession = Depends(get_session)) -> ChatRepository:
    """Provide a ChatRepository bound to the request-scoped session."""
    return ChatRepository(session=session)


def get_session_memory_service(
    request: Request,
    chat_repository: ChatRepository = Depends(get_chat_repository),
) -> SessionMemoryService:
    """Provide a SessionMemoryService using the app's Redis client and request-scoped ChatRepository."""
    redis_client = get_redis_client(request)
    return SessionMemoryService(chat_repository=chat_repository, redis_client=redis_client)


def get_chat_service(
    request: Request,
    chat_repository: ChatRepository = Depends(get_chat_repository),
    session_memory_service: SessionMemoryService = Depends(get_session_memory_service),
    llm_client: LLMClient = Depends(get_llm_client),
) -> ChatService:
    """Provide a ChatService instance with its dependencies injected."""
    graph = build_graph(
        llm_client=llm_client,
        weaviate_client=getattr(request.app.state, "weaviate_client", None),
    )
    return ChatService(
        chat_repository=chat_repository,
        session_memory_service=session_memory_service,
        graph=graph,
    )


def get_document_service(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> DocumentService:
    """Provide a DocumentService bound to the request-scoped session and
    the app-wide Weaviate client. `getattr` rather than direct attribute
    access because `app.state.weaviate_client` is set in the lifespan
    handler, which some ASGI test harnesses skip."""
    return DocumentService(
        session=session,
        weaviate_client=getattr(request.app.state, "weaviate_client", None),
        session_factory=AsyncSessionLocal,
    )


def get_admin_service(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> AdminService:
    """Provide an AdminService bound to the request-scoped session and
    the app-wide Weaviate client."""
    return AdminService(
        session=session,
        weaviate_client=getattr(request.app.state, "weaviate_client", None),
    )
