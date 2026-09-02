"""Query classification (plan.md section 14): decide whether a query
needs retrieval at all, or can go straight to the LLM.

LLM-based rather than a hardcoded keyword list — the distinction
("what is 2+2?" vs "what does the leave policy say?") is semantic, and
the project's `LLMClient` abstraction is already available for exactly
this kind of small classification call.
"""

from enum import Enum

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

_CLASSIFICATION_PROMPT = """You are the front gate of a company support assistant that only answers from the company's own internal documents (HR policies, leave rules, employee handbooks, procedures, etc.).

Classify the message below as RAG_REQUIRED, SMALL_TALK, or GENERAL.

RAG_REQUIRED — the question is about the company, its employees, its workplace, or any policy/procedure/benefit/entitlement an employee might have (leave, holidays, pay, conduct, IT, onboarding, etc.) — EVEN IF:
- it's phrased casually, with typos, in another language, or as a hypothetical/scenario
- it doesn't explicitly say "policy" or name a document
- you personally could guess a plausible generic answer without looking anything up
The test is topic (is this about company/workplace matters?), never "could I answer this without a lookup?" — assume you know nothing about this specific company until you check its documents.

SMALL_TALK — the message is ONLY a conversational pleasantry with no question in it at all: a greeting ("hi", "hello", "good morning"), a thank-you, a goodbye, or "how are you". Nothing is being asked. If the message greets you AND then asks something ("hi, how many sick leaves do I get?"), that is NOT SMALL_TALK — classify it by the question part.

GENERAL — the message asks something that has nothing to do with the company or workplace at all: general trivia, math, coding help, geography, jokes, unrelated companies, or anything else that is clearly off-topic for a company support assistant.

Examples:
"What happens if an employee exhausts their sick leave?" -> RAG_REQUIRED (workplace/leave topic)
"Is there a leave policy for jury duty?" -> RAG_REQUIRED (workplace/leave topic, even if not covered, that's for retrieval to determine)
"matrnity leave weeks bio mother" -> RAG_REQUIRED (workplace/leave topic despite typos)
"what is 2+2" -> GENERAL (pure math, no workplace connection)
"What does Amazon do?" -> GENERAL (unrelated company, no connection to this workplace)
"Write me a Python function to sort a list" -> GENERAL (generic coding request, no workplace connection)
"Hi" -> SMALL_TALK (a bare greeting, nothing is being asked)
"thanks, that helps!" -> SMALL_TALK (a pleasantry, nothing is being asked)
"hi, how many sick leaves do I get?" -> RAG_REQUIRED (the greeting is incidental — there is a real workplace question)

Respond with EXACTLY one word: GENERAL, SMALL_TALK, or RAG_REQUIRED. No explanation.

Question: {query}"""


class QueryClassification(str, Enum):
    GENERAL = "GENERAL"
    SMALL_TALK = "SMALL_TALK"
    RAG_REQUIRED = "RAG_REQUIRED"


async def classify_query(query: str, llm_client: LLMClient) -> QueryClassification:
    """Classify `query` as GENERAL, SMALL_TALK, or RAG_REQUIRED.

    Defaults to RAG_REQUIRED on any ambiguous/unexpected LLM response —
    retrieving unnecessarily is a much cheaper mistake than skipping
    retrieval when it was actually needed.

    SMALL_TALK is checked first because it's the narrowest category (a
    message with no question in it at all), so a response naming it is
    unambiguous even if the model padded its answer.
    """
    raw = await llm_client.generate_reply(_CLASSIFICATION_PROMPT.format(query=query))
    normalized = raw.strip().upper()

    if "SMALL_TALK" in normalized:
        return QueryClassification.SMALL_TALK
    if "GENERAL" in normalized and "RAG_REQUIRED" not in normalized:
        return QueryClassification.GENERAL
    return QueryClassification.RAG_REQUIRED
