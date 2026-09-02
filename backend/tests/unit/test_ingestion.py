"""Unit tests for the ingestion pipeline's pure functions.

No PDF file or database is needed for these — they exercise cleaning
and chunking directly.
"""

import io

import pytest

from app.rag.ingestion.chunker import chunk_pages
from app.rag.ingestion.pdf_extractor import ExtractedPage
from app.rag.ingestion.text_cleaner import clean_text


def test_clean_text_collapses_whitespace() -> None:
    raw = "Line one.   \n\n\n\nLine   two.  \n"
    cleaned = clean_text(raw)
    assert cleaned == "Line one.\n\nLine two."


def test_chunk_pages_respects_size_and_overlap() -> None:
    # ~2000 words, comfortably more than one 600-token chunk.
    pages = [ExtractedPage(page_number=1, text=("word " * 2000).strip())]

    chunks = chunk_pages(pages, chunk_size_tokens=600, overlap_tokens=100)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.token_count == 600
    assert chunks[-1].token_count <= 600
    # Chunks are indexed contiguously from 0.
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_tracks_page_span_across_boundary() -> None:
    pages = [
        ExtractedPage(page_number=1, text="word " * 500),
        ExtractedPage(page_number=2, text="word " * 500),
    ]

    chunks = chunk_pages(pages, chunk_size_tokens=600, overlap_tokens=100)

    # The first chunk should span from page 1 into page 2.
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 2


def test_chunk_pages_empty_input_returns_no_chunks() -> None:
    assert chunk_pages([], chunk_size_tokens=600, overlap_tokens=100) == []


def test_chunk_pages_rejects_overlap_not_smaller_than_size() -> None:
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([ExtractedPage(page_number=1, text="word")], chunk_size_tokens=100, overlap_tokens=100)


def test_extract_pages_reads_from_bytes() -> None:
    """Uploads are never written to disk — extraction takes the PDF's
    bytes, and re-chunking reads them back out of PostgreSQL."""
    from pypdf import PdfWriter

    from app.rag.ingestion.pdf_extractor import extract_pages

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    pages = extract_pages(buffer.getvalue(), name="test.pdf")

    assert [p.page_number for p in pages] == [1, 2]


def test_extract_pages_rejects_bytes_that_are_not_a_pdf() -> None:
    from app.rag.ingestion.pdf_extractor import PdfExtractionError, extract_pages

    with pytest.raises(PdfExtractionError):
        extract_pages(b"this is not a pdf", name="bogus.pdf")


def test_chunk_settings_are_internally_consistent() -> None:
    """`chunk_pages` requires overlap < size, and a zero/negative stride
    would loop forever — so this guards the one thing that must hold
    whatever the tuning.

    The values themselves are a deliberate choice, not asserted here.
    History worth knowing if answers start claiming the policy omits
    something it states: at 100 tokens this corpus splits mid-sentence,
    and the smaller the overlap the more likely a sentence stating two
    rules survives whole in neither neighbour. See CHUNK_SIZE_TOKENS in
    app/core/config.py.
    """
    from app.core.config import get_settings

    settings = get_settings()

    assert settings.CHUNK_OVERLAP_TOKENS < settings.CHUNK_SIZE_TOKENS
    assert settings.CHUNK_OVERLAP_TOKENS >= 0
