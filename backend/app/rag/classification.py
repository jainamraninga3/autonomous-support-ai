"""Query classification (plan.md section 14): decide how to answer a
message — from the company's documents, as conversation, or not at all.

LLM-based rather than a hardcoded keyword list: the distinction is
semantic, and the project's `LLMClient` abstraction is already available
for exactly this kind of small classification call.

GENERAL means "politely decline this" — an off-topic question gets a
fixed message and is never sent to the LLM at all.

This boundary was briefly removed and then restored. Removing it turned
the assistant into a general-purpose one, which is a misuse channel for
a company tool: staff can run maths, homework, and coding work through
the company's API budget, and every message becomes an
LLM-prompt-injection surface. Not calling the LLM at all is the only
version of that boundary that cannot be argued with by the message
itself. The cost is that a genuinely curious off-topic question gets a
redirect rather than an answer — an accepted trade.

Questions about the ASSISTANT itself are not off-topic: they go to
SMALL_TALK and are answered.
"""

from enum import Enum

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

_CLASSIFICATION_PROMPT = """You are routing a message inside a company support assistant. The assistant answers \
ONLY from the company's own internal documents (HR policies, leave rules, employee handbooks, IT \
policies, procedures, etc.). It is not a general-purpose assistant.

Decide which of three routes fits the message below.

RAG_REQUIRED — the message asks about the company, its employees, its workplace, or any \
policy/procedure/benefit/entitlement/rule an employee might have (leave, holidays, pay, \
conduct, IT, onboarding, exit, the company's own business and what it does, etc.) — \
EVEN IF:
- it's phrased casually, with typos, in another language, or as a hypothetical
- it doesn't say "policy" or name a document
- you could guess a plausible generic answer without looking anything up
The test is topic (is this about this company or its workplace?), never "could I answer without a \
lookup?" — assume you know nothing about THIS company until you check its documents.

SMALL_TALK — the message is conversation, not a question with an answer to look up: a greeting \
("hi", "good morning"), a thank-you, a goodbye, or a question about the ASSISTANT ITSELF ("who are \
you?", "what can you do?", "are you a bot?").

GENERAL — anything that is not about this company or workplace: general knowledge, maths, \
homework, coding or programming help, geography, current affairs, other companies, translation \
requests, creative writing, and so on. **TRAVEL is GENERAL** — trips, flights, hotels, \
itineraries, travel booking and travel reimbursement are all out of scope, by an explicit \
product decision on 2026-09-08, even though they are workplace-adjacent and used to be \
RAG_REQUIRED. **META-QUESTIONS ABOUT THE KNOWLEDGE BASE / DOCUMENT INVENTORY ARE ALSO GENERAL** — \
questions asking about the presence, count, list, or metadata of documents/PDFs in the knowledge \
base or database (e.g. "is Code of Conduct in this knowledge base?", "how many PDFs/documents are \
in the knowledge base?", "what documents are uploaded?", "list all documents in the system"). The \
assistant answers policy content questions (e.g. "what is the code of conduct on gifts?"), NOT \
meta-questions about what documents exist in the knowledge base index or file repository. These \
are OUT OF SCOPE and get politely declined, so route them here rather than trying to force them \
into the documents. Be decisive: a maths, coding, or knowledge base inventory request is GENERAL \
even if framed around work.

{history_block}Examples:
"What happens if an employee exhausts their sick leave?" -> RAG_REQUIRED
"Is there a leave policy for jury duty?" -> RAG_REQUIRED (retrieval decides whether it's covered)
"matrnity leave weeks bio mother" -> RAG_REQUIRED (workplace topic despite typos)
"what does our company do?" -> RAG_REQUIRED (about THIS company)
"who are you?" -> SMALL_TALK (about the assistant)
"thanks, that helps!" -> SMALL_TALK
"hi, how many sick leaves do I get?" -> RAG_REQUIRED (the greeting is incidental)
"is Code of Conduct in this knowledge base?" -> GENERAL (meta-question about knowledge base inventory/index)
"how many PDF documents are in the knowledge base?" -> GENERAL (meta-question about stored document count)
"what documents are uploaded in the knowledge base?" -> GENERAL (meta-question about stored files)
"what is 2+2" -> GENERAL (maths)
"write me a Python function to sort a list" -> GENERAL (coding task)
"write a SQL query to join two tables" -> GENERAL (technical task, not a policy question)
"who is the CEO of Amazon?" -> GENERAL (a different company)
"what is the travel policy?" -> GENERAL (travel is out of scope)
"can i claim my flight ticket?" -> GENERAL (travel reimbursement is travel)
"translate this paragraph into French" -> GENERAL (not a policy question)

Respond with EXACTLY one word: RAG_REQUIRED, SMALL_TALK, or GENERAL. No explanation.

Message: {query}"""

_HISTORY_TEMPLATE = """Recent conversation, for context. Use it to resolve what the message REFERS \
to — a follow-up like "where is it located?" or "and for new joiners?" inherits its subject from \
here, and must be classified by that resolved subject, not by the words alone:
{history}

"""


def format_history(history: list[tuple[str, str]] | None, max_chars: int = 1200) -> str:
    """Render recent turns as `role: content` lines, newest last.

    Truncated per message: history is here to identify what a pronoun
    points at, and a long earlier answer would otherwise dominate the
    prompt it's meant to be supporting.
    """
    if not history:
        return ""
    lines = []
    for role, content in history:
        text = " ".join(content.split())
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"  {role}: {text}")
    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = rendered[-max_chars:]
    return rendered


class QueryClassification(str, Enum):
    GENERAL = "GENERAL"
    SMALL_TALK = "SMALL_TALK"
    RAG_REQUIRED = "RAG_REQUIRED"


async def classify_query(
    query: str,
    llm_client: LLMClient,
    history: list[tuple[str, str]] | None = None,
) -> QueryClassification:
    """Classify `query` as RAG_REQUIRED, SMALL_TALK, or GENERAL.

    `history` is recent `(role, content)` turns, used only to resolve
    what a follow-up refers to.

    Defaults to RAG_REQUIRED on any ambiguous/unexpected LLM response —
    checking the documents unnecessarily is a much cheaper mistake than
    answering a company question from general knowledge.

    SMALL_TALK is matched first because it's the narrowest category, so a
    response naming it is unambiguous even if the model padded its reply.
    """
    history_text = format_history(history)
    history_block = _HISTORY_TEMPLATE.format(history=history_text) if history_text else ""

    raw = await llm_client.generate_reply(
        _CLASSIFICATION_PROMPT.format(query=query, history_block=history_block)
    )
    normalized = raw.strip().upper()

    if "SMALL_TALK" in normalized:
        return QueryClassification.SMALL_TALK
    if "GENERAL" in normalized and "RAG_REQUIRED" not in normalized:
        return QueryClassification.GENERAL
    return QueryClassification.RAG_REQUIRED
