"""Shared state definitions for the LangGraph workflow."""

from typing import Any, TypedDict


class GraphState(TypedDict):
    """State passed between LangGraph nodes.

    `chunks`/`citations` hold `RetrievedChunk`/`Citation` instances
    (`app.rag.retrieval.hybrid_search`/`app.rag.generation.context_builder`)
    as plain Python objects — nothing here is persisted or serialized,
    so there's no need to flatten them into primitives.
    """

    original_query: str
    # Recent (role, content) turns, oldest first. Used ONLY to work out
    # what a follow-up refers to ("where is it located?"), never as
    # context to answer from.
    history: list[tuple[str, str]]
    # Per-request retrieval overrides. None means "use the configured
    # default" rather than "off", so a caller that sends nothing behaves
    # exactly as before these existed.
    limit: int | None
    alpha: float | None
    rerank: bool | None
    top_k: int | None
    classification: str | None
    rewritten_query: str | None
    english_query: str | None
    chunks: list[Any]
    citations: list[Any]
    answer: str | None
    was_answerable: bool | None
    verified: bool | None
    verification_reason: str | None
    # How many times `generate` has run for this question. Starts at 0
    # and is what BOUNDS the verify -> generate cycle: LangGraph will
    # loop forever otherwise, and a verifier that rejects every attempt
    # would spend Groq quota until the recursion limit killed the
    # request.
    generation_attempts: int
    response: str | None
    answer_source: str | None


def initial_state(
    query: str,
    *,
    history: list[tuple[str, str]] | None = None,
    limit: int | None = None,
    alpha: float | None = None,
    rerank: bool | None = None,
    top_k: int | None = None,
) -> GraphState:
    """Build a fully-populated starting state for one graph run.

    Every key is set up front (rather than relying on each node to fill
    in the ones it doesn't touch) so no downstream node — or caller
    reading the final result — ever hits a missing key on a path that
    skips some nodes (e.g. the GENERAL branch never runs `retrieve`).
    """
    return GraphState(
        original_query=query,
        history=history or [],
        limit=limit,
        alpha=alpha,
        rerank=rerank,
        top_k=top_k,
        classification=None,
        rewritten_query=None,
        english_query=None,
        chunks=[],
        citations=[],
        answer=None,
        was_answerable=None,
        verified=None,
        verification_reason=None,
        generation_attempts=0,
        response=None,
        answer_source=None,
    )
