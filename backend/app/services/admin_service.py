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
        `document_versions`, `document_chunks`, and `messages` cascade via
        their FKs' `ondelete="CASCADE"` — which now includes the stored
        PDF bytes and chunk text, since nothing lives on disk. Does not
        touch Weaviate; use `reset_vector_store` for that."""
        doc_result = await self.session.execute(delete(Document))
        chat_result = await self.session.execute(delete(ChatSession))
        await self.session.commit()
        details = [
            f"Deleted {doc_result.rowcount} document(s) (document_versions cascaded).",
            f"Deleted {chat_result.rowcount} chat session(s) (messages cascaded).",
        ]
        logger.warning("Postgres reset: %s", " ".join(details))
        return details

    def request_restart(self) -> tuple[bool, list[str]]:
        """Ask the process to exit so a supervisor restarts it.

        Returns (will_actually_restart, details).

        A process cannot restart itself — something outside it has to
        start it again. Under `docker compose` the backend service is
        declared `restart: unless-stopped`, so exiting IS a restart. Run
        directly with `uvicorn --reload` there is no supervisor, so the
        same call just stops the server; that is why this reports whether
        a restart is actually expected rather than claiming success
        either way.

        The exit is scheduled on the event loop rather than done inline so
        this HTTP response is delivered first — otherwise the caller sees
        a connection reset and can't tell "restarting" from "crashed".
        """
        import asyncio
        import os
        import signal

        supervised = os.environ.get("RUNNING_IN_DOCKER") == "1"

        async def _shutdown() -> None:
            await asyncio.sleep(0.5)
            logger.warning("Restart requested via /admin/restart — sending SIGTERM to self")
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.get_running_loop().create_task(_shutdown())

        if supervised:
            details = [
                "SIGTERM scheduled. Docker will restart the container "
                "(restart: unless-stopped), so the API should be back within "
                "a few seconds — the embedding model stays cached in the "
                "model_cache volume.",
            ]
        else:
            details = [
                "SIGTERM scheduled, but this process does not look "
                "supervised (RUNNING_IN_DOCKER is not set), so nothing will "
                "start it again — the server will simply stop. Start it "
                "yourself, or run the stack with docker compose.",
            ]
        return supervised, details

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
