"""Embeds an already-ingested document's chunks into Weaviate.

Separate from `pipeline.py` (extraction/cleaning/chunking, which only
needs PostgreSQL) since embedding needs BGE-M3 and a reachable Weaviate
— see `app/rag/embeddings.py`. Used by the `/api/v1/documents/{id}/ingest`
endpoint and by `scripts.embed_document`.
"""

import json
from pathlib import Path
from uuid import UUID

from weaviate.classes.query import Filter

from app.core.logging import get_logger
from app.database.vector_store import CHUNK_COLLECTION_NAME, ensure_chunk_collection
from app.rag.embeddings import get_embedder
from app.rag.ingestion.vector_writer import write_chunks_to_weaviate
from app.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)


async def read_chunks(session_factory, document_id: UUID, version: int) -> list[dict]:
    async with session_factory() as session:
        repository = DocumentRepository(session=session)
        doc_version = await repository.get_version(document_id, version)
    if doc_version is None:
        raise FileNotFoundError(f"No version {version} found for document_id={document_id}")
    payload = json.loads(Path(doc_version.storage_path).read_text(encoding="utf-8"))
    return payload["chunks"]


def is_already_embedded(weaviate_client, document_id: UUID, version: int) -> bool:
    """Check whether Weaviate already holds chunks for this document/version
    — the actual source of truth, rather than a separately-tracked flag
    that could drift from what's really been written."""
    if weaviate_client is None or not weaviate_client.collections.exists(CHUNK_COLLECTION_NAME):
        return False
    collection = weaviate_client.collections.get(CHUNK_COLLECTION_NAME)
    response = collection.query.fetch_objects(
        filters=(
            Filter.by_property("document_id").equal(str(document_id))
            & Filter.by_property("version").equal(version)
        ),
        limit=1,
    )
    return len(response.objects) > 0


async def embed_document(session_factory, document_id: UUID, version: int, weaviate_client) -> int:
    """Embed one document version's chunks into Weaviate. Idempotent —
    `write_chunks_to_weaviate` upserts by a deterministic UUID, so
    re-embedding an already-embedded document overwrites in place rather
    than duplicating. Returns the number of chunks written."""
    async with session_factory() as session:
        repository = DocumentRepository(session=session)
        document = await repository.get_by_id(document_id)
        if document is None:
            raise FileNotFoundError(f"No document found with id={document_id}")

    chunks = await read_chunks(session_factory, document_id, version)

    ensure_chunk_collection(weaviate_client)
    collection = weaviate_client.collections.get(CHUNK_COLLECTION_NAME)
    written = write_chunks_to_weaviate(
        collection=collection,
        document_id=document_id,
        document_name=document.name,
        version=version,
        chunks=chunks,
        embed_texts_fn=get_embedder().embed_texts,
    )
    logger.info("Embedded document_id=%s version=%s (%d chunks)", document_id, version, written)
    return written
