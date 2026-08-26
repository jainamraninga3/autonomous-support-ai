"""Shared FastAPI dependency providers."""

from collections.abc import AsyncGenerator

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
from app.services.rag_service import RagQueryService

logger = get_logger(__name__)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for the duration of a request."""
    async for session in get_db_session():
        yield session


def get_llm_client() -> LLMClient:
    """Provide the configured LLM client implementation.

    Uses Groq when `GROQ_API_KEY` is set; falls back to the echo stub
    otherwise (e.g. local dev without a key, or tests) so the app still
    boots and the chat endpoint stays usable without external calls.
    """
    settings = get_settings()
    if settings.GROQ_API_KEY:
        return GroqLLMClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
    logger.warning("GROQ_API_KEY not set — falling back to StubLLMClient (echo)")
    return StubLLMClient()


def get_chat_repository(session: AsyncSession = Depends(get_session)) -> ChatRepository:
    """Provide a ChatRepository bound to the request-scoped session."""
    return ChatRepository(session=session)


def get_chat_service(
    request: Request,
    chat_repository: ChatRepository = Depends(get_chat_repository),
    llm_client: LLMClient = Depends(get_llm_client),
) -> ChatService:
    """Provide a ChatService instance with its dependencies injected.

    Builds the LangGraph workflow fresh per request (cheap — it's just
    wiring, not model loading) around this request's LLM client and the
    app-wide Weaviate client. See `get_rag_query_service` for the same
    `getattr` guard reasoning on `weaviate_client`.
    """
    graph = build_graph(
        llm_client=llm_client,
        weaviate_client=getattr(request.app.state, "weaviate_client", None),
    )
    return ChatService(chat_repository=chat_repository, graph=graph)


def get_rag_query_service(
    request: Request,
    llm_client: LLMClient = Depends(get_llm_client),
) -> RagQueryService:
    """Provide a RagQueryService bound to the app-wide Weaviate client
    opened once at startup (see `app.main`'s lifespan) — not reconnected
    per request. `getattr` guards against the lifespan not having run
    (e.g. certain ASGI test harnesses) — `RagQueryService` already
    handles `weaviate_client=None` by raising a clean 503."""
    return RagQueryService(
        weaviate_client=getattr(request.app.state, "weaviate_client", None),
        llm_client=llm_client,
    )


def get_document_service(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> DocumentService:
    """Provide a DocumentService bound to the request-scoped session and
    the app-wide Weaviate client. Same `getattr` guard reasoning as
    `get_rag_query_service` above."""
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
