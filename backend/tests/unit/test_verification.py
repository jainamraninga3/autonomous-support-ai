"""Unit tests for answer verification — fake LLM client, no real model."""

import pytest

from app.llm.base import LLMClient
from app.rag.generation.verification import verify_answer


class _FakeLLMClient(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def generate_reply(self, message: str) -> str:
        return self.reply


@pytest.mark.asyncio
async def test_verify_answer_recognizes_supported() -> None:
    result = await verify_answer(
        "Employees get 20 days of leave.",
        "Employees are entitled to 20 days of leave.",
        _FakeLLMClient("SUPPORTED\nThe number matches the context exactly."),
    )
    assert result.supported is True
    assert "matches" in result.reasoning


@pytest.mark.asyncio
async def test_verify_answer_recognizes_unsupported() -> None:
    result = await verify_answer(
        "Employees get 30 days of leave.",
        "Employees are entitled to 20 days of leave.",
        _FakeLLMClient("UNSUPPORTED\nThe context says 20, not 30."),
    )
    assert result.supported is False
    assert "20" in result.reasoning


@pytest.mark.asyncio
async def test_verify_answer_with_no_context_is_never_supported_without_calling_the_llm() -> None:
    class _BoomLLMClient(LLMClient):
        async def generate_reply(self, message: str) -> str:
            raise AssertionError("should not call the LLM when there's no context to verify against")

    result = await verify_answer("some answer", "   ", _BoomLLMClient())
    assert result.supported is False
