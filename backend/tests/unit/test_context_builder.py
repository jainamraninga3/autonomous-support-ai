"""Unit tests for context/citation construction from retrieved chunks."""

from app.rag.generation.context_builder import build_citations, build_context
from app.rag.retrieval.hybrid_search import RetrievedChunk


def _chunk(chunk_id: str, doc: str, start: int, end: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name=doc,
        version=1,
        chunk_index=0,
        start_page=start,
        end_page=end,
        content=content,
        score=0.9,
    )


def test_build_context_labels_and_includes_content() -> None:
    chunks = [
        _chunk("c1", "policy.pdf", 1, 1, "Leave is 20 days."),
        _chunk("c2", "policy.pdf", 2, 3, "Sick leave needs a certificate."),
    ]

    context = build_context(chunks)

    assert "[Source 1] policy.pdf (page 1)" in context
    assert "Leave is 20 days." in context
    assert "[Source 2] policy.pdf (pages 2-3)" in context
    assert "Sick leave needs a certificate." in context


def test_build_context_dedups_by_chunk_id() -> None:
    chunks = [
        _chunk("c1", "policy.pdf", 1, 1, "Leave is 20 days."),
        _chunk("c1", "policy.pdf", 1, 1, "Leave is 20 days."),  # duplicate
    ]

    context = build_context(chunks)

    assert context.count("[Source") == 1


def test_build_context_empty_input_returns_empty_string() -> None:
    assert build_context([]) == ""


def test_build_citations_matches_labels_and_metadata() -> None:
    chunks = [
        _chunk("c1", "policy.pdf", 1, 1, "..."),
        _chunk("c2", "kitchen.pdf", 5, 5, "..."),
    ]

    citations = build_citations(chunks)

    assert [c.label for c in citations] == ["Source 1", "Source 2"]
    assert citations[0].document_name == "policy.pdf"
    assert citations[1].document_name == "kitchen.pdf"
    assert citations[1].chunk_id == "c2"
