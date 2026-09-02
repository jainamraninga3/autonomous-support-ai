"""Pydantic schemas for the RAG query endpoint."""

from pydantic import BaseModel, Field


class RagAskRequest(BaseModel):
    """A question to answer using the retrieval -> grounded-answer pipeline.

    Mirrors `scripts.ask`'s CLI flags — this endpoint is an HTTP entry
    point onto the same pipeline, not a separate implementation.
    """

    question: str = Field(..., min_length=1)
    limit: int = Field(default=30, ge=1, le=100, description="Stage 1 Top-N candidates.")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0=BM25 only, 1.0=vector only.")
    rerank: bool | None = Field(
        default=None,
        description=(
            "Apply Stage 2 (BGE-Reranker-v2-M3) reranking before answering. "
            "Omit or send null to follow the app-wide RERANK_ENABLED setting "
            "(default: disabled) — this used to default to true, which meant "
            "a single Swagger call silently downloaded the ~2.3GB reranker "
            "model and added tens of seconds per query even when reranking "
            "was switched off everywhere else. Set true/false to override "
            "for one request, which is the point of this endpoint."
        ),
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Max chunks used as context.")


class CitationResponse(BaseModel):
    """One source citation for the generated answer."""

    label: str
    document_name: str
    start_page: int
    end_page: int
    chunk_id: str


class RagAskResponse(BaseModel):
    """The generated answer plus the citations it's grounded in."""

    answer: str
    citations: list[CitationResponse]
    was_answerable: bool
    rewritten_query: str = Field(
        description=(
            "The retrieval-optimized rewrite of `question` (app/rag/query_rewriting.py). "
            "Retrieval searches with this rewrite; the final answer is still generated "
            "from the ORIGINAL `question`. Echoes `question` unchanged when "
            "`classification` short-circuited before retrieval — there is nothing to "
            "rewrite a greeting into."
        )
    )
    english_query: str | None = Field(
        default=None,
        description=(
            "An English translation of `question`, used for a SECOND retrieval pass whose "
            "hits are merged with the original-language pass (app/rag/query_translation.py). "
            "Null when the question was already English, or when translation was skipped — "
            "in both cases retrieval ran once, in the question's own language. The ANSWER is "
            "never translated: it is generated once, directly in the question's language."
        ),
    )
    classification: str = Field(
        description=(
            "How the front gate classified `question` (app/rag/classification.py): "
            "'RAG_REQUIRED' (a company/workplace question — the full retrieval pipeline "
            "ran), 'SMALL_TALK' (a bare greeting or pleasantry — answered conversationally "
            "by the LLM, no retrieval), or 'GENERAL' (off-topic — fixed refusal, no "
            "retrieval and no LLM call on the message itself). Retrieval is skipped for "
            "the latter two, so `citations` is empty and `was_answerable` is false for "
            "both — neither is a failure to find an answer."
        )
    )
