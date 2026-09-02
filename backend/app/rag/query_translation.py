"""Translate a non-English query to English, for a second retrieval pass.

Why this exists: BGE-M3 embeds many languages into one vector space, so
a Hindi query does retrieve from an English document — but not reliably
the SAME chunks an equivalent English query would. Measured on one
compound question against a single English policy PDF, the Hindi query
shared only 6 of 10 retrieved chunks with the English query and missed
the one stating the earned-leave carry-forward and encashment rules; the
answer then (correctly, given its context) told the reader the policy
was silent on both. Kannada shared 8 of 10.

So this is used to run retrieval a SECOND time in English and merge the
hits (`app/rag/retrieval/merge.py`) — recovering what the original-
language query missed, and giving BM25 something lexical to match on,
which it has none of for a Devanagari or Kannada query.

Note what this is NOT: the answer is still generated once, directly in
the user's language, from the original question (see
`app/rag/generation/answer_generator.py`). Nothing translates the
ANSWER — a translation pass over a finished answer is where numbers,
dates, and inclusive/exclusive thresholds get corrupted, and it would
happen after verification could catch it.
"""

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

# Returned when the query needs no translation, so the caller can skip
# the redundant second search instead of searching twice with the same
# text. A sentinel rather than a separate language-detection call:
# "is this already English?" is the only question we need answered, and
# the translation call can answer it for free.
ALREADY_ENGLISH = "ALREADY_ENGLISH"

_TRANSLATION_PROMPT = """Translate the search query below into English, for use as a document \
search query.

Rules:
- If the query is ALREADY entirely in English, respond with exactly {sentinel} and nothing else.
- A query written in Latin script but not actually English (e.g. romanized Hindi like "sick \
leave ek saal me kitni milti hai") is NOT English — translate it.
- Preserve the meaning, numbers, names, negation, and comparisons exactly. Do not answer the \
question, do not add anything, do not explain.
- Prefer the wording a company HR policy document would use.
- Return ONLY the English query, nothing else — no quotes, no commentary.

Query: {query}

English query:"""


async def translate_query_to_english(query: str, llm_client: LLMClient) -> str | None:
    """Return an English translation of `query`, or None if it needs none.

    Returns None (rather than raising or returning the original) on an
    unreliable response — an empty reply, the sentinel, or something so
    long it likely answered the question instead of translating it. None
    means "skip the second retrieval pass", so a bad translation costs
    nothing beyond the call: retrieval still runs in the original
    language exactly as it did before.
    """
    raw = await llm_client.generate_reply(_TRANSLATION_PROMPT.format(query=query, sentinel=ALREADY_ENGLISH))
    translated = raw.strip().strip('"')

    if not translated or ALREADY_ENGLISH in translated.upper():
        return None
    if len(translated) > len(query) * 5:
        logger.warning("Query translation looked unreliable — skipping the English retrieval pass")
        return None
    if translated == query:
        return None
    return translated
