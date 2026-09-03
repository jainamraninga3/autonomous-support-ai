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
from app.rag.classification import format_history

_HISTORY_TEMPLATE = '''Recent conversation, for resolving references only:
{history}

'''

logger = get_logger(__name__)

_REWRITE_PROMPT = """Rewrite the search query below to improve retrieval quality.

{history_block}Rules:
- Preserve the original meaning, intent, numbers, dates, names, negation, comparisons, and constraints exactly.
- Resolve pronouns and references from the conversation above, if any. A follow-up like "where is \
it located?" or "and for new joiners?" is unsearchable on its own — replace "it"/"that"/"they" with \
the thing being referred to, so the rewrite stands alone as a search query.
- Fix spelling and grammar only.
- Expand abbreviations only when their meaning is certain.
- Never invent facts or add assumptions.
- Never answer the question.
- Keep the rewrite in the EXACT SAME language as the original query — never translate it, \
even if you think another language would retrieve better (e.g. a Hindi/Gujarati/Marathi query \
must be rewritten in that same language, not English).
- Return ONLY the rewritten query, nothing else — no explanation, no quotes.

Original query: {query}

Rewritten query:"""


async def rewrite_query(
    query: str,
    llm_client: LLMClient,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Return a retrieval-optimized rewrite of `query`.

    `history` is recent `(role, content)` turns; it lets a follow-up
    resolve its own subject ("where is it located?" -> "where is Amnex
    Infotechnologies located?"). Without it such a question retrieves
    essentially at random, because "it" matches nothing.

    Falls back to the original query verbatim if the LLM's response looks
    unreliable — empty, or suspiciously long (a sign it answered the
    question instead of rewriting it, despite the prompt's explicit
    "never answer" rule). A bad rewrite should never end up worse than no
    rewrite at all.

    The length guard scales with the history: resolving a two-word
    follow-up legitimately produces a much longer query, so the original
    `len(query) * 4` would have rejected exactly the rewrites that matter
    most.
    """
    history_text = format_history(history)
    history_block = _HISTORY_TEMPLATE.format(history=history_text) if history_text else ""

    raw = await llm_client.generate_reply(
        _REWRITE_PROMPT.format(query=query, history_block=history_block)
    )
    rewritten = raw.strip().strip('"')

    max_length = max(len(query) * 4, 200) if history_text else len(query) * 4
    if not rewritten or len(rewritten) > max_length:
        logger.warning("Query rewrite looked unreliable — falling back to the original query")
        return query
    return rewritten
