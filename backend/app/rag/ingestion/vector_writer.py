"""Writes embedded chunks into the Weaviate `DocumentChunk` collection.

Deliberately separate from `pipeline.py` (extraction/cleaning/chunking,
which already persists to PostgreSQL): embedding needs BGE-M3
(`sentence-transformers`/`torch`, not installed by default — see
`app/rag/embeddings.py`) and a running Weaviate. Keeping this as its own
step means the already-verified extract/clean/chunk pipeline doesn't
depend on either.

`collection` and `embed_texts_fn` are injected rather than constructed
here so this is testable without a real Weaviate connection or a real
embedding model.
"""

from collections.abc import Callable
from typing import Any
from uuid import UUID

from weaviate.classes.data import DataObject
from weaviate.util import generate_uuid5

from app.core.logging import get_logger

logger = get_logger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]


def _chunk_uuid(document_id: UUID, version: int, chunk_index: int) -> str:
    """Deterministic UUID so re-writing the same chunk upserts, not duplicates."""
    return generate_uuid5(f"{document_id}:{version}:{chunk_index}")


def write_chunks_to_weaviate(
    collection: Any,
    document_id: UUID,
    document_name: str,
    version: int,
    chunks: list[dict],
    embed_texts_fn: EmbedFn,
) -> int:
    """Embed and upsert a document version's chunks into Weaviate.

    `chunks` are plain dicts with the same shape the ingestion pipeline
    writes to `data/processed/.../v<version>.json`
    (`index`, `text`, `token_count`, `start_page`, `end_page`).

    Returns the number of chunks written.
    """
    if not chunks:
        return 0

    vectors = embed_texts_fn([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError(f"Expected {len(chunks)} vectors, got {len(vectors)}")

    objects = [
        DataObject(
            uuid=_chunk_uuid(document_id, version, chunk["index"]),
            vector=vector,
            properties={
                "chunk_id": _chunk_uuid(document_id, version, chunk["index"]),
                "document_id": str(document_id),
                "document_name": document_name,
                "version": version,
                "chunk_index": chunk["index"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "content": chunk["text"],
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    result = collection.data.insert_many(objects)
    if result.has_errors:
        logger.error("Errors writing chunks to Weaviate for document_id=%s: %s", document_id, result.errors)
        raise RuntimeError(f"Failed to write {len(result.errors)} of {len(objects)} chunks to Weaviate")

    logger.info("Wrote %d chunks to Weaviate for document_id=%s version=%s", len(objects), document_id, version)
    return len(objects)
