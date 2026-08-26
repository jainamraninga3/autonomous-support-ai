"""RAG query orchestration: hybrid search -> optional rerank -> grounded
answer generation.

This is the HTTP-reachable counterpart to `scripts.ask` — same pipeline,
same scope. No query classification (plan.md section 14): every call is
treated as a RAG query. No query rewriting (section 15) or answer
verification (section 26) either.
"""

from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.core.logging import get_logger
from app.database.vector_store import CHUNK_COLLECTION_NAME
from app.llm.base import LLMClient
from app.rag.embeddings import EmbeddingModelUnavailableError, get_embedder
from app.rag.generation.answer_generator import RAGAnswer, generate_answer
from app.rag.reranking import get_reranker
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.rerank import rerank_chunks

logger = get_logger(__name__)


class RagQueryService:
    """Runs the retrieval -> grounded-answer pipeline against a shared
    Weaviate client (opened once at app startup, not per request)."""

    def __init__(self, weaviate_client, llm_client: LLMClient) -> None:
        self.weaviate_client = weaviate_client
        self.llm_client = llm_client

    async def ask(
        self,
        question: str,
        limit: int = 30,
        alpha: float = 0.5,
        do_rerank: bool = True,
        top_k: int = 10,
    ) -> RAGAnswer:
        if self.weaviate_client is None:
            raise ServiceUnavailableError("Weaviate is not available.")

        if not self.weaviate_client.collections.exists(CHUNK_COLLECTION_NAME):
            raise NotFoundError(
                "No documents have been ingested yet "
                f"(collection '{CHUNK_COLLECTION_NAME}' doesn't exist)."
            )
        collection = self.weaviate_client.collections.get(CHUNK_COLLECTION_NAME)

        embed_query_fn = get_embedder().embed_query
        try:
            candidates = hybrid_search(collection, question, embed_query_fn, limit=limit, alpha=alpha)
            if do_rerank:
                reranked = rerank_chunks(question, candidates, get_reranker().score, top_k=top_k)
                chunks = [r.chunk for r in reranked]
            else:
                chunks = candidates[:top_k]
        except EmbeddingModelUnavailableError as exc:
            raise ServiceUnavailableError(str(exc)) from exc

        return await generate_answer(question, chunks, self.llm_client)
