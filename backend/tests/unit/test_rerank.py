"""Unit tests for `rerank_chunks`, with a fake score function — no real
reranker model needed.
"""

import pytest

from app.rag.retrieval.hybrid_search import RetrievedChunk
from app.rag.retrieval.rerank import rerank_chunks


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        document_id="doc-1",
        document_name="policy.pdf",
        version=1,
        chunk_index=index,
        start_page=1,
        end_page=1,
        content=f"content {index}",
        score=0.5,
    )


def test_rerank_sorts_by_score_descending() -> None:
    candidates = [_chunk(0), _chunk(1), _chunk(2)]

    def fake_score(query: str, texts: list[str]) -> list[float]:
        return [0.1, 0.9, 0.5]

    result = rerank_chunks("q", candidates, fake_score, top_k=10)

    assert [r.chunk.chunk_index for r in result] == [1, 2, 0]
    assert result[0].rerank_score == 0.9


def test_rerank_caps_at_top_k() -> None:
    candidates = [_chunk(i) for i in range(5)]

    def fake_score(query: str, texts: list[str]) -> list[float]:
        return [float(i) for i in range(5)]

    result = rerank_chunks("q", candidates, fake_score, top_k=2)

    assert len(result) == 2
    assert [r.chunk.chunk_index for r in result] == [4, 3]


def test_rerank_applies_score_threshold_without_forcing_top_k_count() -> None:
    candidates = [_chunk(0), _chunk(1), _chunk(2)]

    def fake_score(query: str, texts: list[str]) -> list[float]:
        return [0.9, 0.2, 0.1]

    result = rerank_chunks("q", candidates, fake_score, top_k=10, score_threshold=0.5)

    assert len(result) == 1  # not forced to top_k=10, only 1 clears the threshold
    assert result[0].chunk.chunk_index == 0


def test_rerank_empty_candidates_short_circuits() -> None:
    def fake_score(query: str, texts: list[str]) -> list[float]:
        raise AssertionError("should not be called for empty candidates")

    assert rerank_chunks("q", [], fake_score) == []


def test_rerank_raises_on_score_count_mismatch() -> None:
    candidates = [_chunk(0), _chunk(1)]

    def fake_score(query: str, texts: list[str]) -> list[float]:
        return [0.5]  # wrong count on purpose

    with pytest.raises(ValueError):
        rerank_chunks("q", candidates, fake_score)
