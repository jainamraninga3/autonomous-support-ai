"""Query rewriting (plan.md sections 15-17): improve a query for
retrieval while strictly preserving its meaning.

The rewritten query is used ONLY for retrieval — the *original*
question is what's sent to the LLM when generating the final answer
(plan.md section 17: "Do NOT replace the original query"). Callers must
keep both around separately, the same way `GraphState` does
(`original_query` vs `rewritten_query`).
"""

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

_REWRITE_PROMPT = """Rewrite the search query below to improve retrieval quality.

Rules:
- Preserve the original meaning, intent, numbers, dates, names, negation, comparisons, and constraints exactly.
- Fix spelling and grammar only.
- Expand abbreviations only when their meaning is certain.
- Never invent facts or add assumptions.
- Never answer the question.
- Return ONLY the rewritten query, nothing else — no explanation, no quotes.

Original query: {query}

Rewritten query:"""


async def rewrite_query(query: str, llm_client: LLMClient) -> str:
    """Return a retrieval-optimized rewrite of `query`.

    Falls back to the original query verbatim if the LLM's response
    looks unreliable — empty, or suspiciously long (a sign it answered
    the question instead of rewriting it, despite the prompt's explicit
    "never answer" rule). A bad rewrite should never end up worse than
    no rewrite at all.
    """
    raw = await llm_client.generate_reply(_REWRITE_PROMPT.format(query=query))
    rewritten = raw.strip().strip('"')

    if not rewritten or len(rewritten) > len(query) * 4:
        logger.warning("Query rewrite looked unreliable — falling back to the original query")
        return query
    return rewritten
