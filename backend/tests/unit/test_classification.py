"""Unit tests for query classification — fake LLM client, no real model."""

import pytest

from app.llm.base import LLMClient
from app.rag.classification import QueryClassification, classify_query


class _FakeLLMClient(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def generate_reply(self, message: str) -> str:
        return self.reply


@pytest.mark.asyncio
async def test_classify_query_recognizes_general() -> None:
    result = await classify_query("what is 2 + 2?", _FakeLLMClient("GENERAL"))
    assert result == QueryClassification.GENERAL


@pytest.mark.asyncio
async def test_classify_query_recognizes_rag_required() -> None:
    result = await classify_query("what does the leave policy say?", _FakeLLMClient("RAG_REQUIRED"))
    assert result == QueryClassification.RAG_REQUIRED


@pytest.mark.asyncio
async def test_classify_query_defaults_to_rag_required_on_ambiguous_reply() -> None:
    result = await classify_query("some question", _FakeLLMClient("I'm not sure, maybe both?"))
    assert result == QueryClassification.RAG_REQUIRED
