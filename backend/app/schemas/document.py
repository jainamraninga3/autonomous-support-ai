"""Pydantic schemas for the document upload/ingest/list/delete endpoints."""

from uuid import UUID

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """Result of uploading a PDF: extraction + chunking + Postgres write,
    but no embedding yet — call `POST /documents/{document_id}/ingest`
    with this `document_id` next."""

    document_id: UUID
    version: int
    filename: str
    page_count: int
    chunk_count: int
    is_duplicate: bool


class DocumentIngestResponse(BaseModel):
    """Result of embedding a document version's chunks into Weaviate."""

    document_id: UUID
    version: int
    embedded_chunk_count: int
    already_embedded: bool
    message: str


class DocumentVersionInfo(BaseModel):
    version: int
    is_active: bool
    page_count: int
    is_embedded: bool


class DocumentListItem(BaseModel):
    document_id: UUID
    name: str
    content_hash: str
    versions: list[DocumentVersionInfo]


class DocumentDeleteResponse(BaseModel):
    document_id: UUID
    message: str
