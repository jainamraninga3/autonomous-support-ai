"""Orchestrates PDF ingestion: validate -> extract -> clean -> chunk -> persist.

Nothing touches the filesystem: the uploaded PDF's bytes and the chunks
derived from them are both written to PostgreSQL. Chunk text used to live
only in a JSON file on disk with the database holding just its path,
which meant deleting a directory silently destroyed the text and left the
row pointing at nothing.

Does not touch Weaviate or generate embeddings — that's `embedder.py`,
triggered separately by `/api/v1/documents/{id}/ingest`.
"""

import uuid
from dataclasses import asdict, dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.ingestion.chunker import chunk_pages
from app.rag.ingestion.hashing import hash_bytes
from app.rag.ingestion.pdf_extractor import PdfExtractionError, extract_pages
from app.rag.ingestion.text_cleaner import clean_text
from app.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)


class IngestionError(Exception):
    """Raised when a document fails validation or processing."""


@dataclass(frozen=True)
class IngestionResult:
    document_id: uuid.UUID
    version: int
    is_duplicate: bool
    page_count: int
    chunk_count: int


def _validate(file_bytes: bytes, filename: str) -> None:
    if not filename.lower().endswith(".pdf"):
        raise IngestionError(f"Only PDF files are supported, got: {filename}")
    if not file_bytes:
        raise IngestionError(f"File is empty: {filename}")


async def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    session: AsyncSession,
    force_reprocess: bool = False,
) -> IngestionResult:
    """Run the ingestion pipeline for one PDF's bytes.

    If this content has already been ingested, this is a no-op (dedup,
    plan.md section 27) unless `force_reprocess=True`, in which case a
    new version is created for the existing document — that's how a
    document gets re-chunked after `CHUNK_SIZE_TOKENS` changes, reading
    the stored bytes back rather than needing the original file.
    """
    _validate(file_bytes, filename)

    content_hash = hash_bytes(file_bytes)
    repository = DocumentRepository(session=session)

    existing = await repository.get_by_hash(content_hash)
    if existing is not None and not force_reprocess:
        logger.info("Document '%s' already ingested (document_id=%s) — skipping", filename, existing.id)
        latest_version = await repository.next_version_number(existing.id) - 1
        return IngestionResult(
            document_id=existing.id,
            version=max(latest_version, 0),
            is_duplicate=True,
            page_count=0,
            chunk_count=0,
        )

    try:
        pages = extract_pages(file_bytes, name=filename)
    except PdfExtractionError as exc:
        logger.exception("PDF extraction failed for '%s'", filename)
        raise IngestionError(str(exc)) from exc

    cleaned_pages = [
        page.__class__(page_number=page.page_number, text=clean_text(page.text)) for page in pages
    ]

    settings = get_settings()
    chunks = chunk_pages(
        cleaned_pages,
        chunk_size_tokens=settings.CHUNK_SIZE_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
    )

    document = existing if existing is not None else await repository.create_document(
        name=filename, content_hash=content_hash
    )
    version = await repository.next_version_number(document.id)

    doc_version = await repository.create_version(
        document.id, version=version, pdf_bytes=file_bytes, page_count=len(pages)
    )
    await repository.create_chunks(doc_version.id, [asdict(chunk) for chunk in chunks])
    await session.commit()

    logger.info(
        "Ingested '%s' -> document_id=%s version=%s pages=%s chunks=%s",
        filename,
        document.id,
        version,
        len(pages),
        len(chunks),
    )

    return IngestionResult(
        document_id=document.id,
        version=version,
        is_duplicate=False,
        page_count=len(pages),
        chunk_count=len(chunks),
    )
