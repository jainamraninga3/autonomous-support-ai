"""Fixed-size token chunking with overlap (plan.md sections 6 & 8).

Baseline strategy only: no contextual chunking (section 9) and no
parent-child structure (section 10) yet — those are explicitly later
steps in the plan, layered on top of this once a baseline exists to
compare against.

Token counting uses `tiktoken`'s `cl100k_base` encoding as a
model-agnostic approximation. It won't match BGE-M3's own tokenizer
exactly, but is a reasonable stand-in until embeddings are wired in.
"""

from dataclasses import dataclass

import tiktoken

from app.rag.ingestion.pdf_extractor import ExtractedPage

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    """A fixed-size chunk of document text, with its source page span."""

    index: int  # 0-indexed position within the document
    text: str
    token_count: int
    start_page: int
    end_page: int


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size_tokens: int = 600,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Chunk a document's pages into fixed-size, overlapping token windows.

    Pages are concatenated in order; each token is tagged with the page
    it came from so a chunk spanning a page boundary can report an
    accurate `[start_page, end_page]` range.
    """
    if chunk_size_tokens <= overlap_tokens:
        raise ValueError("chunk_size_tokens must be greater than overlap_tokens")

    tokens: list[int] = []
    token_pages: list[int] = []
    for page in pages:
        if not page.text:
            continue
        page_tokens = _ENCODING.encode(page.text)
        tokens.extend(page_tokens)
        token_pages.extend([page.page_number] * len(page_tokens))

    if not tokens:
        return []

    stride = chunk_size_tokens - overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        window = tokens[start:end]
        chunks.append(
            Chunk(
                index=index,
                text=_ENCODING.decode(window),
                token_count=len(window),
                start_page=token_pages[start],
                end_page=token_pages[end - 1],
            )
        )
        index += 1
        if end == len(tokens):
            break
        start += stride

    return chunks
