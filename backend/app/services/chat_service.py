"""Business logic for chat handling.

Routes call into this service; the service in turn talks to the LLM
abstraction. No LLM provider or business logic lives in the route layer.
"""

from app.core.logging import get_logger
from app.llm.base import LLMClient
from app.schemas.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)


class ChatService:
    """Coordinates chat requests between the API layer and the LLM layer."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        logger.info("Handling chat message for conversation_id=%s", request.conversation_id)
        reply = await self.llm_client.generate_reply(request.message)
        return ChatResponse(reply=reply, conversation_id=request.conversation_id)
