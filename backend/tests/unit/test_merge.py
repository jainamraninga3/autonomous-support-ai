"""Unit tests for `merge_by_rank` — the dual-language retrieval merge."""

from app.rag.retrieval.hybrid_search import RetrievedChunk
from app.rag.retrieval.merge import merge_by_rank


def _chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="Leave_Policy.pdf",
        version=1,
        chunk_index=0,
        start_page=1,
        end_page=1,
        content=f"content of {chunk_id}",
        score=score,
    )


def test_interleaves_primary_first_at_each_rank() -> None:
    primary = [_chunk("a"), _chunk("b")]
    secondary = [_chunk("x"), _chunk("y")]

    result = merge_by_rank(primary, secondary, limit=10)

    assert [c.chunk_id for c in result] == ["a", "x", "b", "y"]


def test_drops_chunks_already_seen_from_the_other_list() -> None:
    """The whole point is recovering chunks the original-language query
    missed — a chunk both queries found must not take two slots."""
    primary = [_chunk("a"), _chunk("shared")]
    secondary = [_chunk("shared"), _chunk("x")]

    result = merge_by_rank(primary, secondary, limit=10)

    assert [c.chunk_id for c in result] == ["a", "shared", "x"]


def test_empty_secondary_returns_primary_truncated() -> None:
    """No translation available (an English question) must behave exactly
    like the single-pass retrieval it replaces."""
    primary = [_chunk("a"), _chunk("b"), _chunk("c")]

    assert [c.chunk_id for c in merge_by_rank(primary, [], limit=10)] == ["a", "b", "c"]
    assert [c.chunk_id for c in merge_by_rank(primary, [], limit=2)] == ["a", "b"]


def test_appends_the_tail_of_whichever_list_is_longer() -> None:
    primary = [_chunk("a")]
    secondary = [_chunk("x"), _chunk("y"), _chunk("z")]

    result = merge_by_rank(primary, secondary, limit=10)

    assert [c.chunk_id for c in result] == ["a", "x", "y", "z"]


def test_respects_the_limit() -> None:
    primary = [_chunk(f"p{i}") for i in range(10)]
    secondary = [_chunk(f"s{i}") for i in range(10)]

    result = merge_by_rank(primary, secondary, limit=4)

    # Alternating, primary first — so an equal-length pair splits the
    # slots evenly rather than the English pass displacing the original.
    assert [c.chunk_id for c in result] == ["p0", "s0", "p1", "s1"]
