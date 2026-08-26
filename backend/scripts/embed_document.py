"""CLI entry point for embedding an already-ingested document into Weaviate.

Separate step from `scripts.ingest_document` on purpose: this one needs
BGE-M3 (`sentence-transformers` + `torch`, NOT installed by default —
see `app/rag/embeddings.py`) and a reachable Weaviate. Run
`scripts.ingest_document` first; this reads its JSON chunk output.

Usage (from backend/, with the venv active):
    python -m scripts.embed_document <document_id> [--version N]

Without --version, embeds the document's currently active version.
"""

import argparse
import asyncio
import sys
import uuid

from scripts._console import fix_windows_console_encoding

fix_windows_console_encoding()

from app.database.connection import AsyncSessionLocal
from app.database.vector_store import get_weaviate_client
from app.rag.embeddings import EmbeddingModelUnavailableError
from app.rag.ingestion.embedder import embed_document
from app.repositories.document_repository import DocumentRepository


async def _resolve_version(document_id: uuid.UUID, version: int | None) -> int | None:
    async with AsyncSessionLocal() as session:
        repository = DocumentRepository(session=session)
        if version is not None:
            return version
        doc_version = await repository.get_active_version(document_id)
        return doc_version.version if doc_version else None


def main(document_id: uuid.UUID, version: int | None) -> int:
    resolved_version = asyncio.run(_resolve_version(document_id, version))
    if resolved_version is None:
        print(f"No document/version found for document_id={document_id}", file=sys.stderr)
        return 1

    client = get_weaviate_client()
    try:
        written = asyncio.run(embed_document(AsyncSessionLocal, document_id, resolved_version, client))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except EmbeddingModelUnavailableError as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(f"Wrote {written} chunks to Weaviate for document_id={document_id} version={resolved_version}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed an ingested document's chunks into Weaviate.")
    parser.add_argument("document_id", type=uuid.UUID)
    parser.add_argument("--version", type=int, default=None, help="Defaults to the active version.")
    args = parser.parse_args()
    sys.exit(main(args.document_id, args.version))
