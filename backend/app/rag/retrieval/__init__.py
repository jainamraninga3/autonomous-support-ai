"""Retrieval over the `DocumentChunk` Weaviate collection (plan.md section 18).

Stage 1 (`hybrid_search.py`): hybrid (dense + BM25) search for a
high-recall Top-30 candidate set.
Stage 2 (`rerank.py`): BGE-Reranker-v2-M3 reranks those candidates down
to a relevance-based Top-10-max.

Not implemented yet: query classification/rewriting upstream of this,
or context construction/citations/answer generation downstream of it.
"""

from app.rag.retrieval.hybrid_search import RetrievedChunk, hybrid_search
from app.rag.retrieval.rerank import RerankedChunk, rerank_chunks

__all__ = ["RetrievedChunk", "hybrid_search", "RerankedChunk", "rerank_chunks"]
