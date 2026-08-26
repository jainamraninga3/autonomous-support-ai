"""Orchestrates PDF ingestion: validate -> extract -> clean -> chunk -> persist.

Does not touch Weaviate or generate embeddings — chunks are written to a
JSON file under `PROCESSED_DATA_DIR` for the next phase to pick up.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.ingestion.chunker import chunk_pages
from app.rag.ingestion.hashing import hash_file
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
    output_path: str


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise IngestionError(f"Only PDF files are supported, got: {path.suffix}")
    if path.stat().st_size == 0:
        raise IngestionError(f"File is empty: {path}")


async def ingest_pdf(path: Path, session: AsyncSession, force_reprocess: bool = False) -> IngestionResult:
    """Run the ingestion pipeline for a single PDF file.

    If the file's content has already been ingested, this is a no-op
    (dedup, plan.md section 27) unless `force_reprocess=True`, in which
    case a new version is created for the existing document (e.g. to
    re-chunk with different settings).
    """
    _validate_file(path)

    content_hash = hash_file(path)
    repository = DocumentRepository(session=session)

    existing = await repository.get_by_hash(content_hash)
    if existing is not None and not force_reprocess:
        logger.info("Document '%s' already ingested (document_id=%s) — skipping", path.name, existing.id)
        latest_version = await repository.next_version_number(existing.id) - 1
        return IngestionResult(
            document_id=existing.id,
            version=max(latest_version, 0),
            is_duplicate=True,
            page_count=0,
            chunk_count=0,
            output_path="",
        )

    try:
        pages = extract_pages(path)
    except PdfExtractionError as exc:
        logger.exception("PDF extraction failed for '%s'", path)
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
        name=path.name, content_hash=content_hash
    )
    version = await repository.next_version_number(document.id)

    output_dir = Path(settings.PROCESSED_DATA_DIR) / str(document.id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"v{version}.json"
    output_path.write_text(
        json.dumps(
            {
                "document_id": str(document.id),
                "document_name": path.name,
                "version": version,
                "content_hash": content_hash,
                "page_count": len(pages),
                "chunks": [asdict(chunk) for chunk in chunks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    await repository.create_version(document.id, version=version, storage_path=str(output_path))
    await session.commit()

    logger.info(
        "Ingested '%s' -> document_id=%s version=%s pages=%s chunks=%s",
        path.name,
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
        output_path=str(output_path),
    )
