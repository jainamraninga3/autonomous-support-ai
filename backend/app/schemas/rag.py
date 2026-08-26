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
    rerank: bool = Field(default=True, description="Apply Stage 2 reranking before answering.")
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
