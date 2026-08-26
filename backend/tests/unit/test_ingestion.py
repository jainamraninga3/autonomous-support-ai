"""Unit tests for the ingestion pipeline's pure functions.

No PDF file or database is needed for these — they exercise cleaning
and chunking directly.
"""

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
