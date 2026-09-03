"""LangGraph RAG workflow — plan.md's "Final V1 Pipeline", extended so
the user always gets a real answer:

START -> classify -> SMALL_TALK    -> conversational reply         -> END
                  -> GENERAL       -> general-knowledge answer      -> END
                  -> RAG_REQUIRED  -> rewrite -> retrieve -> generate
                                       -> answerable? -> verify
                                                          -> supported   -> END
                                                          -> unsupported -> general answer -> END
                                       -> not answerable ------------------> general answer -> END

**The GENERAL branch declines without calling the LLM, on purpose.**
Maths, coding, homework, trivia — none of it reaches the model. Two
reasons, and they are both about it being a COMPANY tool: staff would
otherwise run personal work through the company's API budget, and every
message would become a prompt-injection surface. A prompt telling the
model to behave can be argued with by the message; not making the call
cannot. This boundary was briefly removed during development and
restored deliberately.

A question about the ASSISTANT ("who are you?") is not off-topic — that
is SMALL_TALK, and it is answered.

Two routes still end in a general-knowledge answer. Both are for
questions already judged company-related, so they are not a way around
the boundary above:
- `general_fallback` — a company question the documents didn't cover.
- `unverified_fallback` — the documents produced an answer, but
  verification couldn't confirm it was supported, so it was discarded
  and answered generally instead. Previously this path returned only "I
  could not verify this", which left the user with nothing.

Both use the deliberately restrained prompt in
`app/rag/generation/general_answer.py`: a long generic essay about "what
companies typically do" is indistinguishable from real policy to the
reader, and was measured as a high hallucination risk.

This is the first real graph for this project — earlier versions of this
module were a single placeholder echo node, kept only to prove LangGraph
compiled and ran. Dependencies (LLM client, Weaviate client) are captured
via closures built inside `build_graph()` rather than threaded through
`GraphState` — state should hold data, not live service handles.
"""

import asyncio

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import get_settings
from app.core.logging import get_logger
from app.database.vector_store import CHUNK_COLLECTION_NAME
from app.graph.state import GraphState
from app.llm.base import LLMClient
from app.rag.classification import QueryClassification, classify_query
from app.rag.embeddings import EmbeddingModelUnavailableError, get_embedder
from app.rag.generation.answer_generator import generate_answer
from app.rag.generation.context_builder import build_context
from app.rag.generation.general_answer import generate_general_answer
from app.rag.generation.small_talk import generate_small_talk_reply
from app.rag.generation.verification import verify_answer
from app.rag.query_rewriting import rewrite_query
from app.rag.query_translation import translate_query_to_english
from app.rag.retrieval.merge import merge_by_rank
from app.rag.reranking import get_reranker
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.rerank import rerank_chunks

logger = get_logger(__name__)

# Shown above an answer that did NOT come from the company's documents,
# so a reader can never mistake general knowledge for policy. Kept short
# and plain — the earlier wording read like an apology and buried the
# answer under it.
_NOT_IN_DOCUMENTS_NOTE = (
    "_Not found in our documents — answering from general knowledge, "
    "so please confirm anything policy-specific with HR._\n\n"
)

_UNVERIFIED_NOTE = (
    "_Our documents mention this, but I couldn't confirm the details well enough to "
    "quote them as policy — here's what I can tell you generally instead._\n\n"
)

# Sent for anything classified GENERAL. Deliberately fixed text: no LLM
# call happens on this path at all, so there is nothing for a crafted
# message to influence and no API cost to a misuse attempt. Worded to
# explain the scope and redirect rather than just say no — the earlier
# version ("I can't help with unrelated topics like general knowledge,
# math, or coding questions") read as a rebuke.
_OUT_OF_SCOPE_ANSWER = (
    "I'm the assistant for our company's policies and documents — things like leave, travel, "
    "reimbursements, IT rules, conduct, and joining or exit processes. I'm not set up to work "
    "through maths, coding, or general questions, so I'd only guess at those.\n\n"
    "Ask me anything about how things work here and I'll find it in the documents for you."
)

def build_graph(llm_client: LLMClient, weaviate_client) -> CompiledStateGraph:
    """Compile the full classify -> (general | RAG) -> verify workflow.

    `weaviate_client` may be `None` (e.g. Weaviate was unreachable at
    app startup) — the RAG branch degrades to "no context found" rather
    than raising, consistent with how the rest of the app treats a
    missing Weaviate connection.
    """

    async def classify_node(state: GraphState) -> dict:
        classification = await classify_query(
            state["original_query"], llm_client, history=state["history"]
        )
        logger.info(
            "Classified as %s: %r%s",
            classification.value,
            state["original_query"],
            f" (with {len(state['history'])} turns of history)" if state["history"] else "",
        )
        return {"classification": classification.value}

    async def general_node(state: GraphState) -> dict:
        # NO LLM CALL. See the module docstring: this is the misuse and
        # injection boundary, and it only holds because the message never
        # reaches the model.
        #
        # Logged at INFO so out-of-scope traffic is visible — if staff are
        # routinely asking this thing to do their coding, that is worth
        # knowing rather than silently declining forever.
        logger.info("Declined out-of-scope question: %r", state["original_query"])
        return {
            "response": _OUT_OF_SCOPE_ANSWER,
            "answer": _OUT_OF_SCOPE_ANSWER,
            "citations": [],
            "verified": None,
            "answer_source": "off_topic",
        }

    async def small_talk_node(state: GraphState) -> dict:
        # A pure greeting is neither a document question nor something to
        # refuse — see app/rag/generation/small_talk.py for why this one
        # LLM call is safe where `general_node`'s deliberately isn't.
        reply = await generate_small_talk_reply(state["original_query"], llm_client)
        return {
            "response": reply,
            "answer": reply,
            "citations": [],
            "verified": None,
            "answer_source": "small_talk",
        }

    async def rewrite_node(state: GraphState) -> dict:
        # Both calls read only `original_query`, so they run concurrently
        # — the English translation costs no extra wall-clock, it just
        # rides along beside the rewrite it would otherwise wait for.
        rewritten, english = await asyncio.gather(
            rewrite_query(state["original_query"], llm_client, history=state["history"]),
            translate_query_to_english(state["original_query"], llm_client),
        )
        logger.info("Query rewrite: %r -> %r", state["original_query"], rewritten)
        if english:
            logger.info("English retrieval query: %r", english)
        return {"rewritten_query": rewritten, "english_query": english}

    async def retrieve_node(state: GraphState) -> dict:
        if weaviate_client is None:
            logger.warning("Weaviate is not available — answering without any document context")
            return {"chunks": []}
        if not weaviate_client.collections.exists(CHUNK_COLLECTION_NAME):
            # Distinguished from "found nothing" on purpose: this is
            # almost always "no documents have been ingested yet", and
            # silently degrading to the general-knowledge fallback makes
            # that look like a retrieval quality problem instead.
            logger.warning(
                "Weaviate collection %r does not exist — no documents have been ingested yet",
                CHUNK_COLLECTION_NAME,
            )
            return {"chunks": []}

        collection = weaviate_client.collections.get(CHUNK_COLLECTION_NAME)
        query = state["rewritten_query"] or state["original_query"]
        settings = get_settings()

        # Per-request overrides fall back to the configured defaults, so
        # omitting them behaves exactly as it did before they existed.
        limit = state["limit"] if state["limit"] is not None else 30
        alpha = state["alpha"] if state["alpha"] is not None else 0.5
        top_k = state["top_k"] if state["top_k"] is not None else settings.RERANK_TOP_K
        do_rerank = state["rerank"] if state["rerank"] is not None else settings.RERANK_ENABLED

        try:
            embed_query_fn = get_embedder().embed_query
            candidates = hybrid_search(collection, query, embed_query_fn, limit=limit, alpha=alpha)

            # Second pass in English, merged by rank. A non-English query
            # embeds into the same BGE-M3 space as the English documents,
            # but does NOT reliably surface the same chunks an English
            # query would — see app/rag/query_translation.py for the
            # measurement that motivated this.
            english_query = state["english_query"]
            if english_query:
                english_candidates = hybrid_search(
                    collection, english_query, embed_query_fn, limit=limit, alpha=alpha
                )
                candidates = merge_by_rank(candidates, english_candidates, limit=len(candidates))

            if do_rerank:
                reranked = rerank_chunks(query, candidates, get_reranker().score, top_k=top_k)
                chunks = [r.chunk for r in reranked]
            else:
                # Stage 1 already returns highest-score-first, so taking
                # the head is the same selection reranking would make,
                # minus the cross-encoder pass. See RERANK_ENABLED in
                # app/core/config.py for why this is the default.
                chunks = candidates[:top_k]
        except EmbeddingModelUnavailableError:
            logger.exception("Embedding/reranking unavailable during graph retrieval")
            chunks = []
        return {"chunks": chunks}

    async def generate_node(state: GraphState) -> dict:
        result = await generate_answer(state["original_query"], state["chunks"], llm_client)
        return {"answer": result.answer, "citations": result.citations, "was_answerable": result.was_answerable}

    async def general_fallback_node(state: GraphState) -> dict:
        # A question `classify` judged company-related, but the documents
        # had nothing for it (or the answer failed verification). Answer
        # from general knowledge, with a note so it can't be mistaken for
        # policy. Citations stay empty: nothing here is sourced.
        #
        # `verified is False` distinguishes the two ways in — the answer
        # existed but couldn't be confirmed, versus never existing.
        failed_verification = state.get("verified") is False
        # Both ways into this node are company questions — that's what
        # `classify` decided before retrieval ran — so the restrained
        # prompt applies to both.
        answer = await generate_general_answer(state["original_query"], llm_client)
        note = _UNVERIFIED_NOTE if failed_verification else _NOT_IN_DOCUMENTS_NOTE
        response = note + answer
        return {
            "response": response,
            "answer": response,
            "citations": [],
            "answer_source": "unverified_fallback" if failed_verification else "general_fallback",
        }

    async def verify_node(state: GraphState) -> dict:
        context = build_context(state["chunks"])
        verification = await verify_answer(state["answer"], context, llm_client)

        # On failure this used to return a fixed "I could not verify this"
        # and stop, leaving the user with nothing actionable. Now it
        # records the verdict and routes on to a general-knowledge answer
        # — the ungrounded answer is still discarded (that part was
        # right), but the user gets something useful instead of a wall.
        if not verification.supported:
            return {
                "verified": False,
                "verification_reason": verification.reasoning,
                "citations": [],
            }

        return {
            "verified": True,
            "verification_reason": verification.reasoning,
            "response": state["answer"],
            "answer_source": "rag",
        }

    def _route_after_classify(state: GraphState) -> str:
        if state["classification"] == QueryClassification.GENERAL.value:
            return "general"
        if state["classification"] == QueryClassification.SMALL_TALK.value:
            return "small_talk"
        return "rewrite"

    def _route_after_generate(state: GraphState) -> str:
        return "verify" if state["was_answerable"] else "general_fallback"

    def _route_after_verify(state: GraphState) -> str:
        # A verified answer is done. An unverified one falls through to a
        # general-knowledge answer rather than dead-ending.
        return END if state["verified"] else "general_fallback"

    graph = StateGraph(GraphState)
    graph.add_node("classify", classify_node)
    graph.add_node("general", general_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("general_fallback", general_fallback_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"general": "general", "small_talk": "small_talk", "rewrite": "rewrite"},
    )
    graph.add_edge("general", END)
    graph.add_edge("small_talk", END)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges(
        "generate", _route_after_generate, {"verify": "verify", "general_fallback": "general_fallback"}
    )
    graph.add_edge("general_fallback", END)
    graph.add_conditional_edges(
        "verify", _route_after_verify, {END: END, "general_fallback": "general_fallback"}
    )

    return graph.compile()
