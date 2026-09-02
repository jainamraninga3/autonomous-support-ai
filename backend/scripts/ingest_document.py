"""CLI entry point for the document ingestion pipeline.

Usage (from backend/, with the venv active):
    python -m scripts.ingest_document path/to/file.pdf [--force]

Interface is terminal/CLI for this phase, per plan.md ("The initial
interface will be TERMINAL/CLI only"). An upload API endpoint can be
added later without changing `app/rag/ingestion/pipeline.py` itself.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from scripts._console import fix_windows_console_encoding

fix_windows_console_encoding()

from app.database.connection import AsyncSessionLocal
from app.rag.ingestion.pipeline import IngestionError, ingest_pdf


async def main(path: Path, force: bool) -> int:
    if not path.exists():
        print(f"Ingestion failed: File not found: {path}", file=sys.stderr)
        return 1

    # The pipeline stores the PDF's bytes in PostgreSQL and never reads
    # from disk itself, so the CLI is the one that loads the file.
    file_bytes = path.read_bytes()

    async with AsyncSessionLocal() as session:
        try:
            result = await ingest_pdf(file_bytes, path.name, session=session, force_reprocess=force)
        except IngestionError as exc:
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            return 1

    if result.is_duplicate:
        print(f"Already ingested — document_id={result.document_id} (latest version={result.version})")
    else:
        print(
            f"Ingested document_id={result.document_id} version={result.version} "
            f"pages={result.page_count} chunks={result.chunk_count}\n"
            f"Output: {result.output_path}"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into the pipeline (extract, clean, chunk).")
    parser.add_argument("path", type=Path, help="Path to the PDF file to ingest.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if this exact file content was already ingested.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.path, args.force)))
