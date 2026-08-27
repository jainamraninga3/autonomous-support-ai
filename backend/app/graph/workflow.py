"""LangGraph RAG workflow — plan.md's "Final V1 Pipeline", extended with
a disclosed general-knowledge fallback, and scoped to refuse anything
that isn't company-related:

START -> classify -> GENERAL -> refuse (out of scope) -> END
                   -> RAG_REQUIRED -> rewrite -> retrieve -> generate -> [was_answerable?]
                                                                 -> yes -> verify -> END
                                                                 -> no  -> general_fallback -> END

Two DIFFERENT things both involve "general knowledge," on purpose — do
not conflate them:
- `general` (classify said GENERAL — e.g. "what is 2+2", "write me a
  Python function", any topic unrelated to the company) — this bot is
  scoped to company policies/documents ONLY, so these are refused
  outright with a fixed message. No LLM call happens here at all: not
  just "the answer is off-topic" but a deliberate security boundary —
  the raw user message is never handed to an LLM asking it to freely
  answer anything, which is exactly the kind of open surface a user
  could otherwise abuse (jailbreak attempts, "ignore previous
  instructions", unrelated homework help, etc.).
- `general_fallback` (classify said RAG_REQUIRED — i.e. plausibly a
  company question — but retrieval/generation then found nothing
  relevant in our documents) — THIS is where a general-knowledge LLM
  call still happens, but only for a question already judged
  company-relevant, and the reply always discloses it isn't sourced
  from our documents. This is not a backdoor around the refusal above:
  `classify` already sent unrelated questions to `general` before
  retrieval was ever attempted.

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

_GENERAL_FALLBACK_PREFIX = (
    "This question isn't covered by our available documents, so here is a "
    "general-knowledge answer instead (not verified against our internal policies):\n\n"
)

_OUT_OF_SCOPE_ANSWER = (
    "I'm a support assistant for our company's policies and documents — I can only help "
    "with questions related to that. I can't help with unrelated topics like general "
    "knowledge, math, or coding questions."
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
        # Deliberately does NOT call the LLM with the raw user message —
        # see the module docstring. A classified-GENERAL query is, by
        # definition, unrelated to the company's documents; answering it
        # anyway would make this bot an open general-purpose assistant.
        return {
            "response": _OUT_OF_SCOPE_ANSWER,
            "answer": _OUT_OF_SCOPE_ANSWER,
            "citations": [],
            "verified": None,
        }

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
        return {"answer": result.answer, "citations": result.citations, "was_answerable": result.was_answerable}

    async def general_fallback_node(state: GraphState) -> dict:
        # Retrieval/generation found nothing relevant for a question
        # `classify` already judged RAG_REQUIRED — answer from the LLM's
        # own general knowledge rather than just refusing, but disclose
        # that plainly so the user knows it isn't grounded in our
        # documents. Citations stay empty: nothing here is sourced.
        general_answer = await llm_client.generate_reply(state["original_query"])
        response = _GENERAL_FALLBACK_PREFIX + general_answer
        return {"response": response, "answer": response, "citations": [], "verified": None}

    async def verify_node(state: GraphState) -> dict:
        context = build_context(state["chunks"])
        verification = await verify_answer(state["answer"], context, llm_client)
        response = state["answer"] if verification.supported else _UNSUPPORTED_ANSWER
        return {"verified": verification.supported, "response": response}

    def _route_after_classify(state: GraphState) -> str:
        return "general" if state["classification"] == QueryClassification.GENERAL.value else "rewrite"

    def _route_after_generate(state: GraphState) -> str:
        return "verify" if state["was_answerable"] else "general_fallback"

    graph = StateGraph(GraphState)
    graph.add_node("classify", classify_node)
    graph.add_node("general", general_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("general_fallback", general_fallback_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _route_after_classify, {"general": "general", "rewrite": "rewrite"})
    graph.add_edge("general", END)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges(
        "generate", _route_after_generate, {"verify": "verify", "general_fallback": "general_fallback"}
    )
    graph.add_edge("general_fallback", END)
    graph.add_edge("verify", END)

    return graph.compile()
