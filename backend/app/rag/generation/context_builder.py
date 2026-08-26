"""Builds the LLM context block and citation list from retrieved chunks
(plan.md section 20).

Takes plain `RetrievedChunk`s (the shape both `hybrid_search` and
`rerank_chunks` ultimately produce — a reranked list is just
`[r.chunk for r in reranked]`), so this doesn't care whether reranking
happened upstream.

No parent/neighbor expansion (plan.md section 10, parent-child
retrieval — not implemented yet) and no dedup beyond `chunk_id`
(chunks should already be unique post-retrieval; this is a defensive
backstop, not a real dedup pass).
"""

from dataclasses import dataclass

from app.rag.retrieval.hybrid_search import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    """Source metadata for one chunk used in an answer (plan.md section 25).

    Built directly from retrieved metadata, never from anything the LLM
    generates — plan.md section 25 is explicit: "Do NOT hallucinate
    page numbers."
    """

    label: str  # e.g. "Source 1" — matches the marker used in the context block
    document_name: str
    start_page: int
    end_page: int
    chunk_id: str


def _dedup_by_chunk_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    deduped = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append(chunk)
    return deduped


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Return one citation per unique chunk, in the given (already-ranked) order."""
    return [
        Citation(
            label=f"Source {i}",
            document_name=chunk.document_name,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            chunk_id=chunk.chunk_id,
        )
        for i, chunk in enumerate(_dedup_by_chunk_id(chunks), start=1)
    ]


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Build the structured context block to insert into the LLM prompt.

    Each chunk is labeled with the same "Source N" marker used in its
    matching `Citation`, so the model can (optionally) refer to sources
    by label without needing to restate page numbers itself.
    """
    deduped = _dedup_by_chunk_id(chunks)
    if not deduped:
        return ""

    blocks = []
    for i, chunk in enumerate(deduped, start=1):
        page_range = (
            f"page {chunk.start_page}" if chunk.start_page == chunk.end_page else f"pages {chunk.start_page}-{chunk.end_page}"
        )
        blocks.append(f"[Source {i}] {chunk.document_name} ({page_range})\n{chunk.content}")

    return "\n\n".join(blocks)
