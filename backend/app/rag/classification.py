"""Query classification (plan.md section 14): decide whether a query
needs retrieval at all, or can go straight to the LLM.

LLM-based rather than a hardcoded keyword list — the distinction
("what is 2+2?" vs "what does the leave policy say?") is semantic, and
the project's `LLMClient` abstraction is already available for exactly
this kind of small classification call.
"""

from enum import Enum

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

_CLASSIFICATION_PROMPT = """Decide whether answering the question below requires looking up information from a private knowledge base (documents, policies, manuals), or whether it's a general question you could answer directly without any lookup.

Respond with EXACTLY one word: GENERAL or RAG_REQUIRED. No explanation.

Question: {query}"""


class QueryClassification(str, Enum):
    GENERAL = "GENERAL"
    RAG_REQUIRED = "RAG_REQUIRED"


async def classify_query(query: str, llm_client: LLMClient) -> QueryClassification:
    """Classify `query` as GENERAL or RAG_REQUIRED.

    Defaults to RAG_REQUIRED on any ambiguous/unexpected LLM response —
    retrieving unnecessarily is a much cheaper mistake than skipping
    retrieval when it was actually needed.
    """
    raw = await llm_client.generate_reply(_CLASSIFICATION_PROMPT.format(query=query))
    normalized = raw.strip().upper()

    if "GENERAL" in normalized and "RAG_REQUIRED" not in normalized:
        return QueryClassification.GENERAL
    return QueryClassification.RAG_REQUIRED
