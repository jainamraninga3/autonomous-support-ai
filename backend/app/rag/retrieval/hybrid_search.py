"""Hybrid (dense + BM25) search over the `DocumentChunk` collection.

Plan.md section 12: combine dense vector search with BM25 keyword
search rather than relying on vector similarity alone (dense misses
exact names/IDs/acronyms; BM25 misses paraphrases). Weaviate's own
`hybrid()` query does the fusion — we just have to supply both the
query text (for its BM25 side) and the query's dense vector (for its
vector side, since this collection's `vectorizer` is `none`).

`collection` and `embed_query_fn` are injected rather than constructed
here, same reasoning as `vector_writer.py`: testable without a real
Weaviate connection or a real embedding model.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from weaviate.classes.query import Filter, MetadataQuery

EmbedQueryFn = Callable[[str], list[float]]


@dataclass(frozen=True)
class RetrievedChunk:
    """One hybrid-search hit, with plan.md section 20's citation metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    version: int
    chunk_index: int
    start_page: int
    end_page: int
    content: str
    score: float


def hybrid_search(
    collection: Any,
    query: str,
    embed_query_fn: EmbedQueryFn,
    limit: int = 30,
    alpha: float = 0.5,
    filters: Filter | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid search and return up to `limit` candidates, highest score first.

    `alpha` balances BM25 (0.0) against dense vector search (1.0);
    0.5 is a neutral starting point — plan.md section 39 lists
    dense-only/BM25-only/hybrid as something to A/B once an evaluation
    set exists, so this is deliberately not tuned yet.

    This is Stage 1 (high recall, Top 30) only — no reranking here.
    """
    if not query.strip():
        return []

    query_vector = embed_query_fn(query)

    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(score=True),
    )

    return [
        RetrievedChunk(
            chunk_id=obj.properties["chunk_id"],
            document_id=obj.properties["document_id"],
            document_name=obj.properties["document_name"],
            version=obj.properties["version"],
            chunk_index=obj.properties["chunk_index"],
            start_page=obj.properties["start_page"],
            end_page=obj.properties["end_page"],
            content=obj.properties["content"],
            score=obj.metadata.score or 0.0,
        )
        for obj in response.objects
    ]
