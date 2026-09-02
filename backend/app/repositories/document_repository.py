"""Repository for documents and document versions.

Interpretation of the existing schema (see `app/models/document.py`):
`Document.content_hash` is unique, so a `Document` row represents one
exact byte-for-byte file content — re-ingesting identical bytes dedups
to the same `Document` (plan.md section 27). A `DocumentVersion` is one
ingestion/processing run over that content (e.g. re-chunked with
different settings); `is_active` marks the run currently used downstream
(plan.md section 28). A change to the source file's actual content
produces a new `content_hash` and therefore a new `Document`, not a new
version of the old one.
"""

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.document import Document, DocumentChunk, DocumentVersion
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Persists documents and their ingestion versions."""

    async def list_all(self) -> list[Document]:
        result = await self.session.execute(select(Document).options(selectinload(Document.versions)))
        return list(result.scalars().all())

    async def get_by_id_with_versions(self, document_id) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.id == document_id).options(selectinload(Document.versions))
        )
        return result.scalar_one_or_none()

    async def delete_document(self, document: Document) -> None:
        """Deletes the document row; `document_versions` cascades via the
        FK's `ondelete="CASCADE"`."""
        await self.session.delete(document)
        await self.session.commit()

    async def get_by_hash(self, content_hash: str) -> Document | None:
        result = await self.session.execute(select(Document).where(Document.content_hash == content_hash))
        return result.scalar_one_or_none()

    async def get_by_id(self, document_id) -> Document | None:
        result = await self.session.execute(select(Document).where(Document.id == document_id))
        return result.scalar_one_or_none()

    async def get_version(self, document_id, version: int) -> DocumentVersion | None:
        result = await self.session.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id, DocumentVersion.version == version
            )
        )
        return result.scalar_one_or_none()

    async def get_active_version(self, document_id) -> DocumentVersion | None:
        result = await self.session.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id, DocumentVersion.is_active.is_(True)
            )
        )
        return result.scalar_one_or_none()

    async def create_document(self, name: str, content_hash: str) -> Document:
        document = Document(name=name, content_hash=content_hash)
        self.session.add(document)
        await self.session.flush()
        return document

    async def next_version_number(self, document_id) -> int:
        result = await self.session.execute(
            select(DocumentVersion.version)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version.desc())
            .limit(1)
        )
        latest = result.scalar_one_or_none()
        return (latest or 0) + 1

    async def create_chunks(self, document_version_id, chunks: list[dict]) -> int:
        """Persist a version's chunks. `chunks` are dicts as produced by
        `dataclasses.asdict(Chunk)` — the pipeline's own shape, so the
        caller doesn't have to translate."""
        self.session.add_all(
            [
                DocumentChunk(
                    document_version_id=document_version_id,
                    chunk_index=chunk["index"],
                    text=chunk["text"],
                    token_count=chunk["token_count"],
                    start_page=chunk["start_page"],
                    end_page=chunk["end_page"],
                )
                for chunk in chunks
            ]
        )
        await self.session.flush()
        return len(chunks)

    async def get_chunks(self, document_version_id) -> list[DocumentChunk]:
        """A version's chunks in document order."""
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_version_id == document_version_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def create_version(
        self, document_id, version: int, pdf_bytes: bytes, page_count: int
    ) -> DocumentVersion:
        """Create a new version and deactivate any previously active one."""
        await self.session.execute(
            DocumentVersion.__table__.update()
            .where(DocumentVersion.document_id == document_id)
            .values(is_active=False)
        )
        doc_version = DocumentVersion(
            document_id=document_id,
            version=version,
            is_active=True,
            pdf_bytes=pdf_bytes,
            page_count=page_count,
        )
        self.session.add(doc_version)
        await self.session.flush()
        return doc_version
