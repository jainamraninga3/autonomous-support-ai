"""Grounded answer generation (plan.md sections 23-25).

Sends the original user query + retrieved context to the existing
`LLMClient` abstraction (Groq / `openai/gpt-oss-120b` when configured,
`StubLLMClient` echo otherwise — see `app/llm/base.py`). No changes to
that abstraction were needed: the grounded-answer instructions, context,
and question are all composed into the single `message` string its
`generate_reply()` already accepts.

NOT implemented here (later phases, per plan.md): query classification
(section 14) — this assumes the caller already decided RAG applies —
query rewriting (section 15), and answer verification (section 26).
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import LLMClient
from app.rag.generation.context_builder import Citation, build_citations, build_context
from app.rag.retrieval.hybrid_search import RetrievedChunk

logger = get_logger(__name__)

_NO_CONTEXT_ANSWER = "The available information does not contain an answer to this question."

# A deterministic, easy-to-reproduce token rather than requiring the LLM
# to reproduce a full sentence verbatim (unreliable) — `was_answerable`
# below is a substring check against this, not an exact-string match.
# Callers (the graph's generate->verify/general_fallback routing) use
# `was_answerable=False` as the signal to fall back to a disclosed
# general-knowledge answer instead of just refusing outright.
_NO_CONTEXT_SENTINEL = "NOT_FOUND_IN_CONTEXT"

_GROUNDED_ANSWER_PROMPT = """You are a support assistant. Answer the user's question using ONLY the context below.

Rules:
- Do not invent information that is not present in the context.
- If the context does not contain enough information to answer the question, respond with EXACTLY the single word {sentinel} and nothing else — no explanation, no apology.
- Do not use external knowledge unless explicitly allowed.
- Preserve important numbers, dates, names, and conditions exactly as given in the context.
- You may refer to sources by their [Source N] label; do not invent page numbers or source names.

Context:
{context}

Question: {query}

Answer:"""


@dataclass(frozen=True)
class RAGAnswer:
    """A generated answer plus the citations it's grounded in."""

    answer: str
    citations: list[Citation]
    was_answerable: bool  # False when there were no chunks to answer from at all


async def generate_answer(query: str, chunks: list[RetrievedChunk], llm_client: LLMClient) -> RAGAnswer:
    """Generate a grounded answer to `query` using `chunks` as context.

    If `chunks` is empty, skips the LLM call entirely and returns a
    fixed "not found" answer (plan.md section 24: "If the context does
    not contain enough information, clearly state that the information
    could not be found") — there's nothing to ground an answer in, so
    there's no reason to spend an LLM call on it.
    """
    if not chunks:
        logger.info("No chunks retrieved for query — skipping LLM call")
        return RAGAnswer(answer=_NO_CONTEXT_ANSWER, citations=[], was_answerable=False)

    context = build_context(chunks)
    citations = build_citations(chunks)
    prompt = _GROUNDED_ANSWER_PROMPT.format(context=context, query=query, sentinel=_NO_CONTEXT_SENTINEL)

    answer_text = await llm_client.generate_reply(prompt)
    if _NO_CONTEXT_SENTINEL in answer_text:
        logger.info("Retrieved chunks did not contain an answer for this query")
        return RAGAnswer(answer=_NO_CONTEXT_ANSWER, citations=[], was_answerable=False)
    return RAGAnswer(answer=answer_text, citations=citations, was_answerable=True)
