"""Document lifecycle: upload -> ingest (embed) -> list/get -> delete.

Upload and ingestion are deliberately separate HTTP calls (mirroring the
existing `scripts.ingest_document` / `scripts.embed_document` split):
upload only needs PostgreSQL (extract/clean/chunk), ingestion additionally
needs BGE-M3 + a reachable Weaviate. A caller uploads a PDF, gets back its
`document_id`, then explicitly triggers embedding with that id whenever
it's ready to.
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from weaviate.classes.query import Filter

from app.core.config import get_settings
from app.core.exceptions import BadRequestError, NotFoundError
from app.core.logging import get_logger
from app.database.vector_store import CHUNK_COLLECTION_NAME
from app.rag.embeddings import EmbeddingModelUnavailableError
from app.rag.ingestion.embedder import embed_document, is_already_embedded
from app.rag.ingestion.pipeline import IngestionError, IngestionResult, ingest_pdf
from app.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, session: AsyncSession, weaviate_client, session_factory) -> None:
        self.session = session
        self.weaviate_client = weaviate_client
        # `embed_document`/`read_chunks` open their own short-lived
        # sessions (they may run after this request's session context
        # closes, e.g. if this grows a background-task path later) —
        # same pattern the old auto-ingest code used.
        self.session_factory = session_factory
        self.repository = DocumentRepository(session=session)

    async def upload(self, file_bytes: bytes, filename: str) -> IngestionResult:
        """Extract, chunk, and persist a PDF — entirely in PostgreSQL.

        The uploaded bytes are never written to disk: they go into the
        document version row alongside the chunks derived from them, so
        there is no filesystem state to lose or to get out of sync with
        the database.
        """
        try:
            return await ingest_pdf(file_bytes, filename, session=self.session)
        except IngestionError as exc:
            raise BadRequestError(str(exc)) from exc

    async def ingest(self, document_id: UUID, version: int | None, force: bool) -> tuple[int, int, bool]:
        """Embed a document version's chunks into Weaviate. Returns
        (resolved_version, embedded_chunk_count, already_embedded). If
        already embedded and `force` is False, skips the (expensive)
        embed call entirely."""
        document = await self.repository.get_by_id(document_id)
        if document is None:
            raise NotFoundError(f"No document found with id={document_id}")

        doc_version = (
            await self.repository.get_version(document_id, version)
            if version is not None
            else await self.repository.get_active_version(document_id)
        )
        if doc_version is None:
            raise NotFoundError(f"No matching version found for document_id={document_id}")

        if not force and is_already_embedded(self.weaviate_client, document_id, doc_version.version):
            return doc_version.version, 0, True

        try:
            written = await embed_document(self.session_factory, document_id, doc_version.version, self.weaviate_client)
        except FileNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        return doc_version.version, written, False

    async def list_documents(self) -> list:
        return await self.repository.list_all()

    async def get_document(self, document_id: UUID):
        document = await self.repository.get_by_id_with_versions(document_id)
        if document is None:
            raise NotFoundError(f"No document found with id={document_id}")
        return document

    async def delete_document(self, document_id: UUID) -> None:
        document = await self.repository.get_by_id_with_versions(document_id)
        if document is None:
            raise NotFoundError(f"No document found with id={document_id}")

        if self.weaviate_client is not None and self.weaviate_client.collections.exists(CHUNK_COLLECTION_NAME):
            collection = self.weaviate_client.collections.get(CHUNK_COLLECTION_NAME)
            collection.data.delete_many(where=Filter.by_property("document_id").equal(str(document_id)))

        await self.repository.delete_document(document)
