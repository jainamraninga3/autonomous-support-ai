"""Business logic for chat handling.

Routes call into this service; the service runs the LangGraph workflow
(classify -> general LLM reply, or the full RAG pipeline with rewriting
and verification) and persists the conversation. No LLM/RAG logic lives
in the route layer.
"""

import uuid

from langgraph.graph.state import CompiledStateGraph

from app.core.logging import get_logger
from app.graph.state import initial_state
from app.rag.generation.context_builder import Citation
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.rag import CitationResponse

logger = get_logger(__name__)


def _to_citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        label=citation.label,
        document_name=citation.document_name,
        start_page=citation.start_page,
        end_page=citation.end_page,
        chunk_id=citation.chunk_id,
    )


def _parse_session_id(value: str | None) -> uuid.UUID | None:
    """Parse a client-supplied conversation id, ignoring invalid values.

    An invalid or unrecognized id starts a new session rather than
    raising, since strict validation of this field isn't part of this
    phase yet.
    """
    if value is None:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


class ChatService:
    """Coordinates chat requests between the API layer, the LangGraph
    workflow, and persistence."""

    def __init__(self, chat_repository: ChatRepository, graph: CompiledStateGraph) -> None:
        self.chat_repository = chat_repository
        self.graph = graph

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        session_id = _parse_session_id(request.conversation_id)
        chat_session = await self.chat_repository.get_or_create_session(session_id)

        logger.info("Handling chat message for conversation_id=%s", chat_session.id)

        await self.chat_repository.add_message(chat_session.id, role="user", content=request.message)

        result = await self.graph.ainvoke(initial_state(request.message))
        reply = result["response"] or ""
        citations = [_to_citation_response(c) for c in result.get("citations") or []]

        await self.chat_repository.add_message(chat_session.id, role="assistant", content=reply)
        await self.chat_repository.session.commit()

        return ChatResponse(reply=reply, conversation_id=str(chat_session.id), citations=citations)
