"""Pydantic schemas for the chat endpoint."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    message: str = Field(..., min_length=1, description="User's chat message.")
    conversation_id: str | None = Field(
        default=None, description="Optional identifier to group messages into a conversation."
    )


class ChatResponse(BaseModel):
    """Chat response payload."""

    reply: str
    conversation_id: str | None = None
