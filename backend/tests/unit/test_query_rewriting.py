"""Unit tests for query rewriting — fake LLM client, no real model."""

import pytest

from app.llm.base import LLMClient
from app.rag.query_rewriting import rewrite_query


class _FakeLLMClient(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def generate_reply(self, message: str) -> str:
        return self.reply


@pytest.mark.asyncio
async def test_rewrite_query_returns_the_rewritten_text() -> None:
    result = await rewrite_query("wat is leave policy", _FakeLLMClient("What is the leave policy?"))
    assert result == "What is the leave policy?"


@pytest.mark.asyncio
async def test_rewrite_query_strips_surrounding_quotes() -> None:
    result = await rewrite_query("leave policy", _FakeLLMClient('"What is the leave policy?"'))
    assert result == "What is the leave policy?"


@pytest.mark.asyncio
async def test_rewrite_query_falls_back_to_original_on_empty_reply() -> None:
    result = await rewrite_query("original query", _FakeLLMClient("   "))
    assert result == "original query"


@pytest.mark.asyncio
async def test_rewrite_query_falls_back_to_original_when_reply_looks_like_an_answer() -> None:
    original = "leave policy"
    suspiciously_long_reply = "Employees are entitled to twenty days of annual leave per calendar year " * 3
    result = await rewrite_query(original, _FakeLLMClient(suspiciously_long_reply))
    assert result == original
