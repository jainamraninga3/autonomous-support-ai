"""PDF text extraction.

Plain per-page text extraction only, via `pypdf`. Table extraction and
OCR are explicitly out of scope for this pass (plan.md sections 6-7 list
table extraction as a V1 feature, but not required for this initial
extraction/cleaning/chunking slice — OCR stays deferred entirely).
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class ExtractedPage:
    """Raw text extracted from a single PDF page."""

    page_number: int  # 1-indexed, matches how humans refer to PDF pages
    text: str


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be read or parsed."""


def extract_pages(path: Path) -> list[ExtractedPage]:
    """Extract raw text from every page of a PDF, in order."""
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        raise PdfExtractionError(f"Could not read PDF '{path}': {exc}") from exc

    if reader.is_encrypted:
        raise PdfExtractionError(f"PDF '{path}' is encrypted — encrypted PDFs are not supported yet.")

    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf can raise a variety of parser errors per page
            raise PdfExtractionError(f"Failed to extract text from page {index} of '{path}': {exc}") from exc
        pages.append(ExtractedPage(page_number=index, text=text))

    return pages
