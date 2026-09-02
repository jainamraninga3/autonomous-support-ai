"""PDF text extraction.

Plain per-page text extraction only, via `pypdf`. Table extraction and
OCR are explicitly out of scope for this pass (plan.md sections 6-7 list
table extraction as a V1 feature, but not required for this initial
extraction/cleaning/chunking slice — OCR stays deferred entirely).
"""

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class ExtractedPage:
    """Raw text extracted from a single PDF page."""

    page_number: int  # 1-indexed, matches how humans refer to PDF pages
    text: str


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be read or parsed."""


def extract_pages(data: bytes, name: str = "<uploaded pdf>") -> list[ExtractedPage]:
    """Extract raw text from every page of a PDF, in order.

    Takes the PDF's bytes rather than a path: uploads are never written
    to disk, and re-chunking reads the bytes back out of Postgres. `name`
    is only used in error messages.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, OSError) as exc:
        raise PdfExtractionError(f"Could not read PDF '{name}': {exc}") from exc

    if reader.is_encrypted:
        raise PdfExtractionError(f"PDF '{name}' is encrypted — encrypted PDFs are not supported yet.")

    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf can raise a variety of parser errors per page
            raise PdfExtractionError(f"Failed to extract text from page {index} of '{name}': {exc}") from exc
        pages.append(ExtractedPage(page_number=index, text=text))

    return pages
