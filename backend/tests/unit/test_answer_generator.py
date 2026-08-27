"""Unit tests for `generate_answer`, with a fake LLM client — no real
Groq call needed.
"""

import pytest

from app.rag.generation.answer_generator import generate_answer
from app.rag.retrieval.hybrid_search import RetrievedChunk


class _FakeLLMClient:
    def __init__(self, reply: str = "the answer") -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    async def generate_reply(self, message: str) -> str:
        self.last_prompt = message
        return self.reply


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="doc-1",
        document_name="policy.pdf",
        version=1,
        chunk_index=0,
        start_page=1,
        end_page=1,
        content="Employees get 20 days of annual leave.",
        score=0.9,
    )


@pytest.mark.asyncio
async def test_generate_answer_returns_llm_reply_and_citations() -> None:
    llm = _FakeLLMClient(reply="You get 20 days.")
    result = await generate_answer("How many leave days?", [_chunk()], llm)

    assert result.answer == "You get 20 days."
    assert result.was_answerable is True
    assert len(result.citations) == 1
    assert result.citations[0].document_name == "policy.pdf"


@pytest.mark.asyncio
async def test_generate_answer_includes_context_and_query_in_prompt() -> None:
    llm = _FakeLLMClient()
    await generate_answer("How many leave days?", [_chunk()], llm)

    assert "Employees get 20 days of annual leave." in llm.last_prompt
    assert "How many leave days?" in llm.last_prompt
    assert "[Source 1]" in llm.last_prompt


@pytest.mark.asyncio
async def test_generate_answer_skips_llm_call_when_no_chunks() -> None:
    llm = _FakeLLMClient()
    result = await generate_answer("Anything?", [], llm)

    assert result.was_answerable is False
    assert result.citations == []
    assert llm.last_prompt is None  # LLM was never called


@pytest.mark.asyncio
async def test_generate_answer_not_answerable_when_llm_reports_not_found_in_context() -> None:
    """Chunks WERE retrieved, but the LLM correctly determined they don't
    answer the question and emitted the sentinel per the prompt's
    instructions — `was_answerable=False` is the signal the graph uses
    to fall back to a disclosed general-knowledge answer instead of a
    bare refusal."""
    llm = _FakeLLMClient(reply="NOT_FOUND_IN_CONTEXT")
    result = await generate_answer("What is the WFH leave policy?", [_chunk()], llm)

    assert result.was_answerable is False
    assert result.citations == []
    assert result.answer == "The available information does not contain an answer to this question."
