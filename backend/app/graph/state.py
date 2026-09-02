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
    classification: str | None
    rewritten_query: str | None
    english_query: str | None
    chunks: list[Any]
    citations: list[Any]
    answer: str | None
    was_answerable: bool | None
    verified: bool | None
    verification_reason: str | None
    response: str | None
    answer_source: str | None


def initial_state(query: str) -> GraphState:
    """Build a fully-populated starting state for one graph run.

    Every key is set up front (rather than relying on each node to fill
    in the ones it doesn't touch) so no downstream node — or caller
    reading the final result — ever hits a missing key on a path that
    skips some nodes (e.g. the GENERAL branch never runs `retrieve`).
    """
    return GraphState(
        original_query=query,
        classification=None,
        rewritten_query=None,
        english_query=None,
        chunks=[],
        citations=[],
        answer=None,
        was_answerable=None,
        verified=None,
        verification_reason=None,
        response=None,
        answer_source=None,
    )
