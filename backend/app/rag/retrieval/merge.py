"""Merge two ranked retrieval result lists into one.

Used for the dual-language retrieval pass (see
`app/rag/query_translation.py`): the same question is searched once in
the user's language and once translated to English, and the two ranked
lists have to become one Top-K.

Scores are NOT comparable across the two searches — each Weaviate hybrid
query normalizes within its own result set, so a 0.9 from one says
nothing about a 0.9 from the other. Sorting the union by score would
therefore be meaningless. Interleaving by RANK is comparable: "the best
hit for the Hindi query" and "the best hit for the English query" are
both rank 1, whatever their raw numbers.
"""

from app.rag.retrieval.hybrid_search import RetrievedChunk


def merge_by_rank(
    primary: list[RetrievedChunk],
    secondary: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    """Interleave two ranked lists, dropping duplicate chunks, up to `limit`.

    `primary` goes first at every rank, so when both lists are equally
    long the original-language query keeps the majority of the slots and
    the English pass fills the gaps — this is meant to recover chunks the
    original query missed, not to override its ranking.

    An empty `secondary` returns `primary` truncated, so the caller
    doesn't need to special-case "no translation available".
    """
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()

    for pair in zip(primary, secondary, strict=False):
        for chunk in pair:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)

    # Whichever list was longer still has a tail; zip() stopped at the
    # shorter one.
    for chunk in [*primary, *secondary]:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    return merged[:limit]
