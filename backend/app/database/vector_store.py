"""Weaviate connection management and the document-chunk collection schema.

The retrieval pipeline itself (hybrid search, reranking, etc.) is still a
later phase — this only defines where ingested chunks + their BGE-M3
vectors are stored, per plan.md section 4.1.
"""

import weaviate
from weaviate.classes.config import Configure, DataType, Property

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CHUNK_COLLECTION_NAME = "DocumentChunk"


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Return a Weaviate client connected to the configured instance."""
    settings = get_settings()
    return weaviate.connect_to_custom(
        http_host=settings.WEAVIATE_HOST,
        http_port=settings.WEAVIATE_PORT,
        http_secure=False,
        grpc_host=settings.WEAVIATE_HOST,
        grpc_port=settings.WEAVIATE_GRPC_PORT,
        grpc_secure=False,
    )


def ensure_chunk_collection(client: weaviate.WeaviateClient) -> None:
    """Create the `DocumentChunk` collection if it doesn't already exist.

    Vectorizer is `none` — we supply BGE-M3 vectors ourselves at insert
    time rather than have Weaviate call out to a vectorizer module.
    Properties mirror plan.md section 4.1's example. `tenant_id`,
    `document_type`, `access_level`, `section`, and `parent_id` exist in
    the schema for forward compatibility (metadata filtering, ACLs,
    parent-child retrieval — all later phases) but are NOT populated by
    the ingestion pipeline yet; only the fields it actually has are set.
    """
    if client.collections.exists(CHUNK_COLLECTION_NAME):
        return

    client.collections.create(
        name=CHUNK_COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="document_name", data_type=DataType.TEXT),
            Property(name="version", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="start_page", data_type=DataType.INT),
            Property(name="end_page", data_type=DataType.INT),
            Property(name="content", data_type=DataType.TEXT),
            # Forward-compatible, unpopulated for now (see docstring):
            Property(name="section", data_type=DataType.TEXT),
            Property(name="parent_id", data_type=DataType.TEXT),
            Property(name="tenant_id", data_type=DataType.TEXT),
            Property(name="document_type", data_type=DataType.TEXT),
            Property(name="access_level", data_type=DataType.TEXT),
        ],
    )
    logger.info("Created Weaviate collection '%s'", CHUNK_COLLECTION_NAME)


def check_weaviate_connection() -> bool:
    """Verify that Weaviate is reachable. Used by the health endpoint."""
    try:
        client = get_weaviate_client()
        try:
            return client.is_ready()
        finally:
            client.close()
    except Exception:
        logger.exception("Weaviate connectivity check failed")
        return False
