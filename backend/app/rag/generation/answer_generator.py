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

_GROUNDED_ANSWER_PROMPT = """You are a friendly, knowledgeable assistant helping someone who \
works at this company. Answer their question using ONLY the context below, which comes from the \
company's own documents.

MATCH THE ANSWER TO THE QUESTION, and keep it TIGHT. Lead with the direct answer, then only \
the specifics that make it usable: the actual numbers, limits, deadlines and conditions. \
Include an exception only when it would actually catch this person out.

LENGTH IS DECIDED BY THE QUESTION. Read it and pick one of two modes.

DEFAULT — BRIEF. Use this unless the question asks for more:
- ONE level of bullets. NEVER nest a bullet under a bullet.
- ONE line per item. "what is the leave policy" gets one line per leave type with its \
headline entitlement — NOT its eligibility, carry-forward, encashment and application rules \
as well. Those are what a follow-up question is for.
- Under 150 words. Stop there even if the context holds more.
- Do not enumerate every fact in the context just because it is there.

DETAILED — only when the question ASKS for it. Signals include "in detail", "detailed", \
"details", "explain", "elaborate", "full", "complete", "everything", "breakdown", \
"step by step", "tell me more", "all the rules/conditions", or the same intent in any \
other language. Judge the INTENT, not the exact words — someone asking "what are all the \
conditions for sick leave" wants detail without using any of those phrases. Then:
- Cover everything the context has on the subject, sub-bullets and a table are both fine.
- No word limit.
- Still never invent anything, and still no "Summary" section restating what you wrote.

When it is genuinely ambiguous, go BRIEF and end with one short line offering the detail: \
"Ask about any one of these and I'll give the full rules." A short answer plus an offer \
costs the reader one follow-up; a wall of text costs them the whole answer.

Write warmly and plainly, as a helpful colleague would. No preamble ("Certainly!", "Great \
question"), no restating the question back.

Rules:
- Do not invent information that is not present in the context.
- Use {sentinel} ONLY when the context contains nothing relevant to the question at all. \
Respond with EXACTLY the single word {sentinel} and nothing else in that case — no explanation, no apology.
- If the context answers the question even partially, ANSWER IT with what the context does \
contain — do not use {sentinel}. For a question with several parts (e.g. "compare X and Y", \
or a question about several leave types at once), answer every part the context covers, \
gathering the relevant facts from wherever they appear across the context, and briefly note \
which specific parts aren't covered. A partial answer is far more useful than {sentinel}.
- Do not use external knowledge unless explicitly allowed.
- If the EXACT thing asked for isn't in the context but closely related figures are, GIVE those and name the difference. Asked for a "daily allowance" when the context has per-day accommodation limits and meal reimbursement, the useful answer is those limits plus "there's no separate daily allowance stated" — not a bare "not available", which is technically correct and practically useless.
- When the context is SILENT on part of the question, say ONLY that the document does not cover it — "the policy does not state whether X", "no requirement for X is mentioned". Do not ALSO assert what the rule therefore is. Writing "X is not required; the policy does not mention it" is still wrong: the first half is an invented entitlement and adding the second half does not license it. Drop the assertion and keep only the statement about the document. Silence is not evidence that the rule permits something.
- Preserve important numbers, dates, names, and conditions exactly as given in the context. Keep thresholds EXACTLY as inclusive or exclusive as the context states them: "3 or more consecutive days" must not become "more than 3 days", "up to 15" must not become "under 15", "at least" must not become "more than". A shifted boundary changes who the rule applies to. This matters most when answering in a different language from the context — check the boundary survived the translation.
- If you end with a summary or conclusion, it must repeat the numbers and boundaries EXACTLY as you stated them above it. A table saying "3 or more consecutive days" followed by a summary saying "more than 3 days" gives the reader two different rules; the summary is where this slips, so re-check it against your own answer before finishing.
- Before writing that the context does not cover something, re-read the context for it. Look for it inside sentences that are mainly about something else, and in other sections — a single sentence often states two rules at once (e.g. carry-forward and encashment in the same line), and it is easy to notice one and miss the other. Saying "the policy does not mention X" when X is actually there hides a real rule from the reader — that is a worse error than omitting X silently.
- Never generalize a rule beyond the scope the context gives it. If the context states a rule for one specific case (a particular leave type, grade, or situation), keep it attached to that case — do not restate it as applying to all of them. Watch for rules that DIFFER between cases: when the context gives different answers for different types, say so explicitly per type instead of picking one and presenting it as universal. Only use words like "all", "every", "always", or "never" when the context itself says the rule is universal.
- Format the answer as plain Markdown, and never emit HTML — no <br>, no <b>, no <div>. A renderer that (correctly) refuses raw HTML shows those tags literally to the user. For a line break inside a list item, start a new list item instead.
- Use "-" for bullets, not "•".
- Reach for a table ONLY when the question genuinely compares two or more things across the same dimensions. For everything else use short paragraphs and bullet lists, which read better and translate better. When you do use a table, keep each cell to a short phrase: a Markdown cell cannot hold a bulleted list or a line break, so put multiple points in separate rows rather than cramming them into one cell.
- NEVER write a "Summary" or "Key Numbers" section that restates the answer. It doubles the length, adds nothing, and is where numbers silently drift out of step with the body above it. End when you have answered the question. No horizontal rules ("---") either.
- You may refer to sources by their [Source N] label; do not invent page numbers or source names.
- Respond in the SAME language the Question is written in — even if the Context below is in a \
different language (e.g. the documents are in English but the Question is in Hindi, Gujarati, \
Marathi, or any other language: translate the relevant facts and answer in the Question's \
language). If the Question mixes languages, mirror that same mix in your answer.

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


_CORRECTION_SUFFIX = """

IMPORTANT — YOUR PREVIOUS ATTEMPT AT THIS ANSWER WAS REJECTED.

A verifier compared it against the context above and objected:

    {objection}

Write the answer again WITHOUT the unsupported claim. REMOVE it — do not
rephrase it, do not hedge it, do not soften it with "typically" or
"generally". If the context does not state it, it does not belong in the
answer at all. Everything else you got right; keep it. If removing the
claim leaves the question partly unanswered, say plainly that the
documents do not cover that part."""


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    llm_client: LLMClient,
    objection: str | None = None,
) -> RAGAnswer:
    """Generate a grounded answer to `query` using `chunks` as context.

    `objection` is the verifier's reason for rejecting a previous
    attempt. When set, the prompt gains a correction instruction telling
    the model to REMOVE the unsupported claim rather than rephrase it —
    rephrasing was the observed failure mode, since a hedged version of
    an invented figure is still an invented figure.

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
    if objection:
        # Appended rather than woven in, so the base prompt is byte-identical
        # on the first attempt and this is provably a no-op when there is no
        # objection.
        prompt += _CORRECTION_SUFFIX.format(objection=objection.strip())

    answer_text = await llm_client.generate_reply(prompt)
    if _NO_CONTEXT_SENTINEL in answer_text:
        logger.info("Retrieved chunks did not contain an answer for this query")
        return RAGAnswer(answer=_NO_CONTEXT_ANSWER, citations=[], was_answerable=False)
    return RAGAnswer(answer=answer_text, citations=citations, was_answerable=True)
