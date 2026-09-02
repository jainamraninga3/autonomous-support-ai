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
    rewritten_query: str | None = Field(
        default=None,
        description=(
            "The retrieval-optimized rewrite of `message` (app/rag/query_rewriting.py). "
            "Only present on the RAG_REQUIRED branch — null for a GENERAL-classified "
            "refusal, since no rewriting happens there. Retrieval searches with this "
            "rewrite; the final answer is still generated from the ORIGINAL `message`."
        ),
    )
    english_query: str | None = Field(
        default=None,
        description=(
            "An English translation of `message`, used for a SECOND retrieval pass whose hits "
            "are merged with the original-language pass (app/rag/query_translation.py). Null "
            "for an already-English message, a refusal, or a greeting. The reply itself is "
            "never translated — it is generated once, directly in your language."
        ),
    )
    retrieved_chunk_count: int = Field(
        default=0,
        description=(
            "How many chunks were retrieved from Weaviate for this query, before "
            "generation. Always 0 for a GENERAL-classified refusal (retrieval never "
            "ran). Can still be > 0 even when answer_source is 'general_fallback' — "
            "retrieval may find chunks that exist in the documents but aren't "
            "actually relevant to the question; it's generation, not retrieval, "
            "that decides whether they answer it (see the NOT_FOUND_IN_CONTEXT "
            "sentinel in app/rag/generation/answer_generator.py)."
        ),
    )
    verified: bool | None = Field(
        default=None,
        description=(
            "Whether the verification step (app/rag/generation/verification.py) judged "
            "the generated answer to be supported by its own retrieved context. Only set "
            "on the 'rag' path — null for a refusal or the general-knowledge fallback, "
            "where there is nothing to verify. When this is false, `reply` is NOT the "
            "generated answer: it's a fixed 'could not verify' message, and the answer "
            "that was produced has been discarded."
        ),
    )
    verification_reason: str | None = Field(
        default=None,
        description=(
            "The verifier's one-sentence justification for `verified`. Exposed mainly so "
            "an over-eager verifier is diagnosable — if `verified` is false, this is the "
            "only place that says why the answer was thrown away."
        ),
    )
    answer_source: str | None = Field(
        default=None,
        description=(
            "Which path in the graph produced `reply`: "
            "'off_topic_refusal' (classify said GENERAL, fixed refusal, no LLM call), "
            "'small_talk' (a bare greeting or pleasantry — a short conversational reply, no "
            "retrieval, citations empty), "
            "'rag' (a real, document-grounded answer, verified against its own citations), or "
            "'general_fallback' (classify said RAG_REQUIRED but nothing relevant was found, "
            "so the LLM answered from general knowledge — reply discloses this, citations "
            "stay empty). Lets a caller tell these apart programmatically instead of "
            "string-matching the reply text."
        ),
    )
