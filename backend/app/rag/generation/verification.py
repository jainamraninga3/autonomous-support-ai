"""Answer verification (plan.md section 26): check whether a generated
answer is actually supported by the retrieved context, rather than
trusting the generation step unconditionally.
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)

_VERIFICATION_PROMPT = """You are checking whether an AI-generated answer is fully supported by the context it was given — not whether it's a good answer, only whether every claim in it is grounded in the context below.

Context:
{context}

Answer to check:
{answer}

Respond with EXACTLY one word on the first line: SUPPORTED or UNSUPPORTED.
Then, on the next line, a one-sentence reason."""


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
    return AnswerVerification(supported=supported, reasoning=reasoning)
