"""Document upload / ingest / list / get / delete routes.

Upload only extracts + chunks + writes to Postgres (no embedding) and
returns a `document_id`; that id is then passed to `ingest` to embed the
document's chunks into Weaviate. Kept as two calls rather than one so a
caller can upload many documents cheaply and choose when to pay for the
(slower) embedding step.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_document_service
from app.rag.ingestion.embedder import is_already_embedded
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentListItem,
    DocumentUploadResponse,
    DocumentVersionInfo,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Upload a PDF: extracts, cleans, and chunks it, and writes it to
    PostgreSQL. Does NOT embed it — call `POST /{document_id}/ingest`
    with the returned `document_id` next."""
    file_bytes = await file.read()
    result = await service.upload(file_bytes, file.filename or "upload.pdf")
    return DocumentUploadResponse(
        document_id=result.document_id,
        version=result.version,
        filename=file.filename or "upload.pdf",
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        is_duplicate=result.is_duplicate,
    )


@router.post("/{document_id}/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    document_id: UUID,
    version: int | None = None,
    force: bool = False,
    service: DocumentService = Depends(get_document_service),
) -> DocumentIngestResponse:
    """Embed a document version's chunks into Weaviate. Checks whether
    this version was already embedded first, and skips re-embedding
    unless `force=true` (embedding is idempotent, but re-running it is
    still wasted work when nothing changed)."""
    resolved_version, embedded_count, already_embedded = await service.ingest(document_id, version, force)
    message = (
        "Already embedded — pass ?force=true to re-embed anyway."
        if already_embedded
        else f"Embedded {embedded_count} chunks."
    )
    return DocumentIngestResponse(
        document_id=document_id,
        version=resolved_version,
        embedded_chunk_count=embedded_count,
        already_embedded=already_embedded,
        message=message,
    )


@router.get("", response_model=list[DocumentListItem])
async def list_documents(service: DocumentService = Depends(get_document_service)) -> list[DocumentListItem]:
    documents = await service.list_documents()
    return [
        DocumentListItem(
            document_id=document.id,
            name=document.name,
            content_hash=document.content_hash,
            versions=[
                DocumentVersionInfo(
                    version=v.version,
                    is_active=v.is_active,
                    storage_path=v.storage_path,
                    is_embedded=is_already_embedded(service.weaviate_client, document.id, v.version),
                )
                for v in document.versions
            ],
        )
        for document in documents
    ]


@router.get("/{document_id}", response_model=DocumentListItem)
async def get_document(
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> DocumentListItem:
    document = await service.get_document(document_id)
    return DocumentListItem(
        document_id=document.id,
        name=document.name,
        content_hash=document.content_hash,
        versions=[
            DocumentVersionInfo(
                version=v.version,
                is_active=v.is_active,
                storage_path=v.storage_path,
                is_embedded=is_already_embedded(service.weaviate_client, document.id, v.version),
            )
            for v in document.versions
        ],
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    """Deletes the document's Postgres rows, its stored chunk JSON files,
    and any chunks it has in Weaviate."""
    await service.delete_document(document_id)
    return DocumentDeleteResponse(document_id=document_id, message="Document deleted.")
