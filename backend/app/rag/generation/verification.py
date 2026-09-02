"""Answer verification (plan.md section 26): check whether a generated
answer is actually supported by the retrieved context, rather than
trusting the generation step unconditionally.
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

_VERIFICATION_PROMPT = """You are checking whether an AI-generated answer is supported by the context it was given — not whether it's a good answer, only whether its factual claims are grounded in the context below.

Mark UNSUPPORTED only when the answer actually contradicts the context, or states a specific fact (a number, date, name, condition, or entitlement) that the context does not contain at all. That is the ONLY reason to reject an answer.

Do NOT mark UNSUPPORTED for any of these — all of them are expected and fine:
- The answer combines facts drawn from several different parts of the context, or reorganises them (e.g. a comparison table, or a summary spanning multiple sections). Facts do not need to appear together in one place, or in the same order as the answer presents them.
- The answer paraphrases, condenses, or reformats the context instead of quoting it.
- The answer is written in a DIFFERENT language from the context (e.g. English context, Hindi/Gujarati/Marathi answer) — judge semantic support across languages, not literal text matching.
- The answer omits some detail present in the context, or is shorter/less complete than it could be. Incompleteness is not a grounding failure.
- The answer adds neutral connective wording, headings, or structure that carries no factual claim of its own.
- The answer states that the context does NOT cover something, or does not mention some detail (e.g. "the policy does not specify X", "no rule is stated for Y") AND the context indeed does not contain it. A truthful statement about what is ABSENT from the context is not a factual claim that needs grounding — it is the answer being honest about its own limits.

DO mark UNSUPPORTED when the answer claims the context is silent about something the context ACTUALLY STATES. Check each such claim against the context before accepting it: telling a reader "the policy does not mention X" when the policy does mention X is just as wrong as inventing a fact, and it hides a real rule from them.

Also mark UNSUPPORTED when the answer changes an inclusive threshold into an exclusive one or vice versa — "3 or more days" becoming "more than 3 days", "up to 30" becoming "over 30", "at least" becoming "more than". These read as harmless rewordings but change who the rule applies to.

When in doubt, answer SUPPORTED.

Context:
{context}

Answer to check:
{answer}

Work through this before answering:
1. List every claim in the answer that says the context does NOT cover something ("not stated", "not mentioned", "no information", "the document is silent"). For EACH one, search the context for that topic. If you find it — even inside a sentence mainly about something else, or in a different section — the claim is false and the verdict is UNSUPPORTED.
2. Check every number, date, and threshold in the answer against the context, including inclusive/exclusive boundaries.
3. If the answer contains a summary or conclusion section, check it against the answer's own body: a summary that restates a rule differently from the body above it (a different number, or a shifted boundary) is UNSUPPORTED.

Respond with EXACTLY one word on the first line: SUPPORTED or UNSUPPORTED.
Then, on the next line, a one-sentence reason naming the specific claim if you rejected it."""


@dataclass(frozen=True)
class AnswerVerification:
    supported: bool
    reasoning: str


async def verify_answer(answer: str, context: str, llm_client: LLMClient) -> AnswerVerification:
    """Ask the LLM to judge whether `answer` is grounded in `context`.

    Defaults to `supported=True` on an ambiguous verdict — verification
    here is a safety net on top of the grounded-answer generation
    prompt, not the primary grounding mechanism; a verifier that's too
    eager to flag good answers as unsupported would do more harm than
    the rare unsupported answer it misses.
    """
    if not context.strip():
        return AnswerVerification(supported=False, reasoning="No context was retrieved to verify against.")

    raw = await llm_client.generate_reply(_VERIFICATION_PROMPT.format(context=context, answer=answer))
    lines = raw.strip().splitlines()
    verdict = lines[0].strip().upper() if lines else ""
    reasoning = lines[1].strip() if len(lines) > 1 else raw.strip()

    supported = "UNSUPPORTED" not in verdict
    if not supported:
        # Logged at WARNING because this discards an already-generated
        # answer and replaces it with a refusal — an over-eager verifier
        # is otherwise completely invisible from the outside.
        logger.warning("Verification rejected the generated answer: %s", reasoning)
    else:
        logger.info("Verification passed: %s", reasoning)
    return AnswerVerification(supported=supported, reasoning=reasoning)
