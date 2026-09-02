"""RAG query orchestration: query rewrite -> hybrid search -> optional
rerank -> grounded answer generation.

This is the HTTP-reachable counterpart to `scripts.ask`. Classification
(plan.md section 14) runs first, so a bare greeting or an off-topic
question is answered without paying for retrieval — embedding + reranking
a message like "Hi" costs tens of seconds to find nothing. Everything
past that gate is the same rewrite -> retrieve -> generate pipeline.
No answer verification (section 26) — that stays specific to the
/api/v1/chat graph.
"""

import asyncio
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.core.logging import get_logger
from app.database.vector_store import CHUNK_COLLECTION_NAME
from app.llm.base import LLMClient
from app.rag.classification import QueryClassification, classify_query
from app.rag.embeddings import EmbeddingModelUnavailableError, get_embedder
from app.rag.generation.answer_generator import Citation, generate_answer
from app.rag.generation.small_talk import generate_small_talk_reply
from app.rag.query_rewriting import rewrite_query
from app.rag.query_translation import translate_query_to_english
from app.rag.reranking import get_reranker
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.merge import merge_by_rank
from app.rag.retrieval.rerank import rerank_chunks

logger = get_logger(__name__)

# Kept identical in wording to the graph's refusal (app/graph/workflow.py)
# so both endpoints present the same front gate to a user.
_OUT_OF_SCOPE_ANSWER = (
    "I'm a support assistant for our company's policies and documents — I can only help "
    "with questions related to that. I can't help with unrelated topics like general "
    "knowledge, math, or coding questions."
)


@dataclass(frozen=True)
class RagAskResult:
    """A generated answer plus the citations it's grounded in, the
    retrieval-optimized rewrite of the original question, and how the
    question was classified.

    `rewritten_query` echoes the original question unchanged when
    classification short-circuited before the rewrite step ran — there
    is nothing to rewrite a greeting into."""

    answer: str
    citations: list[Citation]
    was_answerable: bool
    rewritten_query: str
    english_query: str | None
    classification: str


class RagQueryService:
    """Runs the rewrite -> retrieval -> grounded-answer pipeline against a
    shared Weaviate client (opened once at app startup, not per request)."""

    def __init__(self, weaviate_client, llm_client: LLMClient) -> None:
        self.weaviate_client = weaviate_client
        self.llm_client = llm_client

    async def ask(
        self,
        question: str,
        limit: int = 30,
        alpha: float = 0.5,
        do_rerank: bool | None = None,
        top_k: int = 10,
    ) -> RagAskResult:
        # None means "use the app-wide default" rather than "off", so this
        # endpoint and the /api/v1/chat graph can't disagree about whether
        # reranking is on. Loading the reranker is a ~2.3GB download on
        # first use, so an accidental default here is expensive.
        if do_rerank is None:
            do_rerank = get_settings().RERANK_ENABLED

        # Front gate first: a greeting or an off-topic question never
        # needs Weaviate, an embedding pass, or the reranker. Doing this
        # before the availability checks below also means small talk keeps
        # working when no documents have been ingested yet.
        classification = await classify_query(question, self.llm_client)
        logger.info("Query classified as %s: %r", classification.value, question)

        if classification is QueryClassification.SMALL_TALK:
            reply = await generate_small_talk_reply(question, self.llm_client)
            return RagAskResult(
                answer=reply,
                citations=[],
                was_answerable=False,
                rewritten_query=question,
                english_query=None,
                classification=classification.value,
            )

        if classification is QueryClassification.GENERAL:
            return RagAskResult(
                answer=_OUT_OF_SCOPE_ANSWER,
                citations=[],
                was_answerable=False,
                rewritten_query=question,
                english_query=None,
                classification=classification.value,
            )

        if self.weaviate_client is None:
            raise ServiceUnavailableError("Weaviate is not available.")

        if not self.weaviate_client.collections.exists(CHUNK_COLLECTION_NAME):
            raise NotFoundError(
                "No documents have been ingested yet "
                f"(collection '{CHUNK_COLLECTION_NAME}' doesn't exist)."
            )
        collection = self.weaviate_client.collections.get(CHUNK_COLLECTION_NAME)

        # Rewrite is used ONLY for retrieval — generation still answers the
        # user's ORIGINAL question, same rule as the /api/v1/chat graph
        # (app/graph/workflow.py's rewrite_node / app/rag/query_rewriting.py).
        # Concurrent: both only read `question`, so the English
        # translation adds no wall-clock over the rewrite alone.
        rewritten_query, english_query = await asyncio.gather(
            rewrite_query(question, self.llm_client),
            translate_query_to_english(question, self.llm_client),
        )
        logger.info("Query rewrite: %r -> %r", question, rewritten_query)
        if english_query:
            logger.info("English retrieval query: %r", english_query)

        embed_query_fn = get_embedder().embed_query
        try:
            candidates = hybrid_search(collection, rewritten_query, embed_query_fn, limit=limit, alpha=alpha)

            # Second retrieval pass in English, merged by rank — see
            # app/rag/query_translation.py. No-op for English questions.
            if english_query:
                english_candidates = hybrid_search(
                    collection, english_query, embed_query_fn, limit=limit, alpha=alpha
                )
                candidates = merge_by_rank(candidates, english_candidates, limit=limit)

            if do_rerank:
                reranked = rerank_chunks(rewritten_query, candidates, get_reranker().score, top_k=top_k)
                chunks = [r.chunk for r in reranked]
            else:
                chunks = candidates[:top_k]
        except EmbeddingModelUnavailableError as exc:
            raise ServiceUnavailableError(str(exc)) from exc

        result = await generate_answer(question, chunks, self.llm_client)
        return RagAskResult(
            answer=result.answer,
            citations=result.citations,
            was_answerable=result.was_answerable,
            rewritten_query=rewritten_query,
            english_query=english_query,
            classification=classification.value,
        )
