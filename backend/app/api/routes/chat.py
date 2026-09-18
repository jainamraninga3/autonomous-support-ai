"""Chat route.

Contains no business logic — requests are delegated to the ChatService.

Requires a logged-in caller. The 401 this returns when logged out is the
REAL boundary; the frontend's own "please log in" message is only there so
the user sees something friendlier than a failed request.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service, get_current_user
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Send a message to the chat pipeline and receive a reply.

    THE CLIENT'S OWN `user_id` IS OVERWRITTEN with the authenticated user's
    id, never trusted. `ChatRepository.get_or_create_session` enforces
    session ownership in SQL against that value, so honouring a
    client-supplied one would let anyone read anyone else's conversation by
    claiming their id — a real hole that only closed once accounts existed.
    The field stays on the schema (ignored) rather than being removed, so an
    older client sending it gets the safe behaviour instead of a 422.
    """
    request = request.model_copy(update={"user_id": str(current_user.id)})
    return await chat_service.handle_message(request)
