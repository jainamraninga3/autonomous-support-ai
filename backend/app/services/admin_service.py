"""Destructive data-reset operations for local development/testing —
wipe PostgreSQL's document/chat data, or empty the Weaviate vector
store, independently or together.
"""

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger
from app.database.vector_store import CHUNK_COLLECTION_NAME, ensure_chunk_collection
from app.models.chat import ChatSession
from app.models.document import Document

logger = get_logger(__name__)


class AdminService:
    def __init__(self, session: AsyncSession, weaviate_client) -> None:
        self.session = session
        self.weaviate_client = weaviate_client

    async def reset_postgres(self) -> list[str]:
        """Deletes all `documents` and `chat_sessions` rows.
        `document_versions` and `messages` cascade via their FKs'
        `ondelete="CASCADE"`. Does not touch files on disk (the chunk
        JSON under `PROCESSED_DATA_DIR`, or uploaded PDFs)."""
        doc_result = await self.session.execute(delete(Document))
        chat_result = await self.session.execute(delete(ChatSession))
        await self.session.commit()
        details = [
            f"Deleted {doc_result.rowcount} document(s) (document_versions cascaded).",
            f"Deleted {chat_result.rowcount} chat session(s) (messages cascaded).",
        ]
        logger.warning("Postgres reset: %s", " ".join(details))
        return details

    def reset_vector_store(self) -> list[str]:
        """Deletes and recreates the `DocumentChunk` Weaviate collection,
        discarding every embedded chunk."""
        if self.weaviate_client is None:
            raise ServiceUnavailableError("Weaviate is not available.")

        existed = self.weaviate_client.collections.exists(CHUNK_COLLECTION_NAME)
        if existed:
            self.weaviate_client.collections.delete(CHUNK_COLLECTION_NAME)
        ensure_chunk_collection(self.weaviate_client)
        detail = (
            f"Deleted and recreated the '{CHUNK_COLLECTION_NAME}' collection."
            if existed
            else f"'{CHUNK_COLLECTION_NAME}' didn't exist — created it fresh."
        )
        logger.warning("Vector store reset: %s", detail)
        return [detail]
