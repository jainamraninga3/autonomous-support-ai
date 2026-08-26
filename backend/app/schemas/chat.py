"""Pydantic schemas for the chat endpoint."""

from pydantic import BaseModel, Field

from app.schemas.rag import CitationResponse


class ChatRequest(BaseModel):
    """Incoming chat request payload.

    No document id, ever: retrieval always searches every embedded chunk
    in Weaviate — there's nothing to scope a chat request to a specific
    document. `conversation_id` is unrelated to documents; it only
    threads messages into the same conversation across calls.
    """

    message: str = Field(..., min_length=1, description="User's chat message.")
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Optional session UUID to continue an existing conversation. "
            "Omit this (or send null) for a new conversation — do NOT "
            "send the literal example text 'string'; that would try to "
            "look up a session with that exact id and fail to match, "
            "silently starting a new one anyway."
        ),
    )

    model_config = {"json_schema_extra": {"example": {"message": "What is the leave policy?", "conversation_id": None}}}


class ChatResponse(BaseModel):
    """Chat response payload.

    `citations` is empty for a general (non-RAG) reply — it's only
    populated when the query was classified as RAG-required and chunks
    were actually retrieved.
    """

    reply: str
    conversation_id: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
