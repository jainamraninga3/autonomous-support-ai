"""LangGraph RAG workflow — plan.md's "Final V1 Pipeline":

START -> classify -> GENERAL -> llm -> END
                   -> RAG_REQUIRED -> rewrite -> retrieve -> generate -> verify -> END

This is the first real graph for this project — earlier versions of this
module were a single placeholder echo node, kept only to prove LangGraph
compiled and ran. Dependencies (LLM client, Weaviate client) are captured
via closures built inside `build_graph()` rather than threaded through
`GraphState` — state should hold data, not live service handles.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.logging import get_logger
from app.database.vector_store import CHUNK_COLLECTION_NAME
from app.graph.state import GraphState
from app.llm.base import LLMClient
from app.rag.classification import QueryClassification, classify_query
from app.rag.embeddings import EmbeddingModelUnavailableError, get_embedder
from app.rag.generation.answer_generator import generate_answer
from app.rag.generation.context_builder import build_context
from app.rag.generation.verification import verify_answer
from app.rag.query_rewriting import rewrite_query
from app.rag.reranking import get_reranker
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.rerank import rerank_chunks

logger = get_logger(__name__)

_UNSUPPORTED_ANSWER = (
    "I found some information but could not fully verify that this answer is "
    "supported by it, so I'd rather not risk giving you an unverified answer."
)


def build_graph(llm_client: LLMClient, weaviate_client) -> CompiledStateGraph:
    """Compile the full classify -> (general | RAG) -> verify workflow.

    `weaviate_client` may be `None` (e.g. Weaviate was unreachable at
    app startup) — the RAG branch degrades to "no context found" rather
    than raising, consistent with how the rest of the app treats a
    missing Weaviate connection.
    """

    async def classify_node(state: GraphState) -> dict:
        classification = await classify_query(state["original_query"], llm_client)
        return {"classification": classification.value}

    async def general_node(state: GraphState) -> dict:
        response = await llm_client.generate_reply(state["original_query"])
        return {"response": response, "answer": response, "citations": [], "verified": None}

    async def rewrite_node(state: GraphState) -> dict:
        rewritten = await rewrite_query(state["original_query"], llm_client)
        return {"rewritten_query": rewritten}

    async def retrieve_node(state: GraphState) -> dict:
        if weaviate_client is None or not weaviate_client.collections.exists(CHUNK_COLLECTION_NAME):
            return {"chunks": []}

        collection = weaviate_client.collections.get(CHUNK_COLLECTION_NAME)
        query = state["rewritten_query"] or state["original_query"]
        try:
            embed_query_fn = get_embedder().embed_query
            candidates = hybrid_search(collection, query, embed_query_fn)
            reranked = rerank_chunks(query, candidates, get_reranker().score)
            chunks = [r.chunk for r in reranked]
        except EmbeddingModelUnavailableError:
            logger.exception("Embedding/reranking unavailable during graph retrieval")
            chunks = []
        return {"chunks": chunks}

    async def generate_node(state: GraphState) -> dict:
        result = await generate_answer(state["original_query"], state["chunks"], llm_client)
        return {"answer": result.answer, "citations": result.citations}

    async def verify_node(state: GraphState) -> dict:
        if not state["chunks"]:
            # Nothing was retrieved at all — generate_answer already returned
            # its fixed "not found" answer; there's nothing to verify against.
            return {"verified": False, "response": state["answer"]}

        context = build_context(state["chunks"])
        verification = await verify_answer(state["answer"], context, llm_client)
        response = state["answer"] if verification.supported else _UNSUPPORTED_ANSWER
        return {"verified": verification.supported, "response": response}

    def _route_after_classify(state: GraphState) -> str:
        return "general" if state["classification"] == QueryClassification.GENERAL.value else "rewrite"

    graph = StateGraph(GraphState)
    graph.add_node("classify", classify_node)
    graph.add_node("general", general_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _route_after_classify, {"general": "general", "rewrite": "rewrite"})
    graph.add_edge("general", END)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
