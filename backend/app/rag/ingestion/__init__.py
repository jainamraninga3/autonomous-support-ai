"""Document ingestion pipeline (extraction -> cleaning -> chunking).

Standalone for now: this module reads a PDF and produces cleaned,
fixed-size chunks with metadata, and records the document/version in
PostgreSQL. It does NOT talk to Weaviate or generate embeddings yet —
that's the next phase, per `../../plan.md` V1 ordering.
"""

from app.rag.ingestion.pipeline import IngestionError, IngestionResult, ingest_pdf

__all__ = ["IngestionError", "IngestionResult", "ingest_pdf"]
