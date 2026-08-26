"""Stage 2 of retrieval: rerank Stage 1's Top-30 down to a relevance-based
Top-10-max (plan.md sections 18-19).

`score_fn` is injected (not constructed here) for the same testability
reason as `hybrid_search.py`: no real reranker model needed for tests.
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.rag.retrieval.hybrid_search import RetrievedChunk

ScoreFn = Callable[[str, list[str]], list[float]]


@dataclass(frozen=True)
class RerankedChunk:
    """A Stage-1 chunk plus its Stage-2 reranker score."""

    chunk: RetrievedChunk
    rerank_score: float


def rerank_chunks(
    query: str,
    candidates: list[RetrievedChunk],
    score_fn: ScoreFn,
    top_k: int = 10,
    score_threshold: float | None = None,
) -> list[RerankedChunk]:
    """Rerank Stage 1 candidates and return up to `top_k`, highest score first.

    `score_threshold`, if set, drops anything scoring below it even if
    that leaves fewer than `top_k` results — plan.md section 18 is
    explicit that the final count should NOT be forced to exactly 10.
    """
    if not candidates:
        return []

    scores = score_fn(query, [c.content for c in candidates])
    if len(scores) != len(candidates):
        raise ValueError(f"Expected {len(candidates)} scores, got {len(scores)}")

    reranked = sorted(
        (RerankedChunk(chunk=c, rerank_score=s) for c, s in zip(candidates, scores, strict=True)),
        key=lambda r: r.rerank_score,
        reverse=True,
    )

    if score_threshold is not None:
        reranked = [r for r in reranked if r.rerank_score >= score_threshold]

    return reranked[:top_k]
