"""LangGraph RAG workflow — plan.md's "Final V1 Pipeline", extended so
the user always gets a real answer:

START -> classify -> SMALL_TALK    -> conversational reply         -> END
                  -> GENERAL       -> general-knowledge answer      -> END
                  -> RAG_REQUIRED  -> rewrite -> retrieve -> generate
                                       -> answerable? -> verify
                                                          -> supported   -> END
                                                          -> unsupported -> RETRY generate (once)
                                                          -> still bad   -> general answer -> END
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
  verification could not confirm it even after ONE regeneration with the
  verifier's objection fed back in. Only then is it discarded. Without
  that retry, a whole grounded answer with citations was thrown away
  over a single unsupported sentence.

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
    "> ⚠️ **Not from your company's documents.**\n"
    "> I couldn't find this in the uploaded policies, so the answer below is "
    "**general information from the internet** — it is NOT your company's "
    "policy and the figures in it are not yours. Confirm anything that "
    "matters with HR.\n\n"
)

# How many times `generate` may run for one question, including the
# first. 2 = one original attempt plus one correction pass. See
# `_route_after_verify`.
_MAX_GENERATION_ATTEMPTS = 2

_UNVERIFIED_NOTE = (
    "> ⚠️ **Not from your company's documents.**\n"
    "> Your documents do mention this, but I couldn't confirm the details well "
    "enough to quote them as policy, so I've discarded that draft. The answer "
    "below is **general information from the internet** — not your company's "
    "policy. Worth asking again in different words, since the material does "
    "appear to be there; otherwise check with HR.\n\n"
)
# Both notes are prepended to an answer produced by the UNRESTRAINED
# prompt in `general_answer.py` (see its docstring for the risk that
# accepts). They are kept DISTINCT rather than merged because they call
# for different next actions: "found nothing" means ask HR, whereas
# "found it but couldn't verify it" means rephrase and try again — the
# material is in the corpus. Collapsing them would lose that, and a
# person told to go to HR will not retry a question that would have
# worked.

# Sent for anything classified GENERAL. Deliberately fixed text: no LLM
# call happens on this path at all, so there is nothing for a crafted
# message to influence and no API cost to a misuse attempt. Worded to
# explain the scope and redirect rather than just say no — the earlier
# version ("I can't help with unrelated topics like general knowledge,
# math, or coding questions") read as a rebuke.
_OUT_OF_SCOPE_ANSWER = (
    "I'm the assistant for our company's policies and documents — things like company policy, "
    "IT rules, conduct, and joining or exit processes. I'm not set up to work through maths, "
    "coding, or travel, so I'd only guess at those.\n\n"
    "Ask me anything about how things work here."
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
        # History is passed so "what is my name?" right after "my name is
        # Jainam" can be answered. Without it this node had no way to see
        # the previous turn, and the question fell to GENERAL's fixed
        # refusal — which recited the maths/coding/travel scope at someone
        # who had just introduced themselves.
        reply = await generate_small_talk_reply(
            state["original_query"], llm_client, history=state["history"]
        )
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
        # On a retry the verifier's objection is passed back in, so the
        # model is told exactly which claim to drop rather than being
        # asked to try again blind.
        attempt = state["generation_attempts"] + 1
        objection = state["verification_reason"] if state["generation_attempts"] else None
        if objection:
            logger.info("Regenerating (attempt %d) after: %s", attempt, objection)

        # History goes in as CONTEXT ONLY — it tells the model who is
        # asking and what a follow-up refers to. `answer_generator`'s
        # prompt is explicit that it can never supply or override a
        # policy fact: the documents are the record, and a user saying
        # "I think casual leave is 20 days" does not make it 20.
        result = await generate_answer(
            state["original_query"],
            state["chunks"],
            llm_client,
            objection=objection,
            history=state["history"],
        )
        return {
            "answer": result.answer,
            "citations": result.citations,
            "was_answerable": result.was_answerable,
            "generation_attempts": attempt,
        }

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
            # Citations are deliberately NOT cleared here. The answer may
            # be regenerated and pass, in which case its citations must
            # still be there — and clearing them made retrieval metrics
            # read 0 for every rejected answer, which looked like a
            # retrieval failure when retrieval had worked fine.
            return {
                "verified": False,
                "verification_reason": verification.reasoning,
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
        """Done, retry once, or give up.

        The retry budget is ONE. Each attempt costs a generate + a verify
        call against a Groq token quota, and a claim rejected twice is
        usually one the model believes — a third attempt tends to return
        the same sentence in new words rather than dropping it.

        Throwing away a whole grounded answer with citations because ONE
        sentence was unsupported is the failure this fixes. Observed
        2026-09-08: a leave-policy answer built from 10 real chunks was
        discarded entirely over an invented "26 weeks paid maternity
        leave", and replaced by a general-knowledge reply with no
        citations at all.
        """
        if state["verified"]:
            return END
        if state["generation_attempts"] < _MAX_GENERATION_ATTEMPTS:
            return "generate"
        logger.warning(
            "Answer still unverified after %d attempts — falling back. Last objection: %s",
            state["generation_attempts"],
            state["verification_reason"],
        )
        return "general_fallback"

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
    # The one cycle in this graph: verify -> generate. Bounded by
    # `generation_attempts` in `_route_after_verify`; without that bound
    # LangGraph would loop until its recursion limit.
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {END: END, "generate": "generate", "general_fallback": "general_fallback"},
    )

    return graph.compile()
