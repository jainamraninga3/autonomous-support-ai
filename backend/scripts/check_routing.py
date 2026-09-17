"""Check query routing against the REAL LLM, without the stack running.

The unit suite stubs the LLM, so it can prove the routing code works but
never that the routing PROMPT works — and the prompt is where the bugs
have been. This calls Groq for real and asserts each case lands on the
route it should.

    python -m scripts.check_routing

Needs GROQ_API_KEY (from backend/.env). Nothing else: no Postgres, no
Weaviate, no Redis, no running backend. Roughly one second per case.

Add a case whenever a routing bug is found, so the next prompt edit has
to keep it fixed. Cases that are NOT about the bug being fixed matter
just as much — a prompt edit that repairs one route usually breaks
another, and that is exactly what this catches.
"""

import asyncio
import sys

from scripts._console import fix_windows_console_encoding

fix_windows_console_encoding()

from app.api.dependencies import get_llm_client
from app.rag.classification import classify_query

_INTRO = [
    ("user", "my name is Jainam"),
    ("assistant", "Nice to meet you, Jainam! Let me know if there's anything I can help you with."),
]
_SICK_LEAVE = [
    ("user", "what is the sick leave policy"),
    ("assistant", "7 days per calendar year, carried forward up to 15 days."),
]

# (question, history, expected route)
CASES = [
    # Answerable from the conversation itself. Regression: "what is my
    # name" right after an introduction was declined as off-topic, which
    # the GENERAL definition won because it came later in the prompt.
    ("what is my name", _INTRO, "SMALL_TALK"),
    ("what did I just ask you?", _INTRO, "SMALL_TALK"),
    ("can you repeat that?", _SICK_LEAVE, "SMALL_TALK"),
    # Ordinary conversation.
    ("hi", None, "SMALL_TALK"),
    ("thanks, that helps!", None, "SMALL_TALK"),
    ("who are you?", None, "SMALL_TALK"),
    # Off-topic. These must NOT be collateral damage of the above.
    ("what is 2+2", None, "GENERAL"),
    ("write me a python function to sort a list", None, "GENERAL"),
    ("what is the travel policy?", None, "GENERAL"),
    ("how many documents are in the knowledge base?", None, "GENERAL"),
    # Company questions, including one that only resolves via history.
    ("how many sick leaves do i get", None, "RAG_REQUIRED"),
    ("and for new joiners?", _SICK_LEAVE, "RAG_REQUIRED"),
    ("what does our company do?", None, "RAG_REQUIRED"),
    ("matrnity leave weeks bio mother", None, "RAG_REQUIRED"),
]


async def _run() -> int:
    llm_client = get_llm_client()
    failures = 0
    for query, history, expected in CASES:
        actual = (await classify_query(query, llm_client, history=history)).value
        if actual != expected:
            failures += 1
        marker = "ok  " if actual == expected else "FAIL"
        print(f"{marker}  {query[:45]:47} expected={expected:13} got={actual}")

    print()
    if failures:
        print(f"{failures} of {len(CASES)} cases routed wrong.")
    else:
        print(f"All {len(CASES)} cases routed correctly.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
