"""Citation schema, shared by every response that returns sources.

Named `rag.py` for historical reasons: it used to hold the request and
response models for a separate `POST /api/v1/rag/ask` endpoint, which was
merged into `POST /api/v1/chat` — one endpoint rather than two subtly
different RAG paths.
"""

from pydantic import BaseModel

class CitationResponse(BaseModel):
    """One source citation for the generated answer."""

    label: str
    document_name: str
    start_page: int
    end_page: int
    chunk_id: str
