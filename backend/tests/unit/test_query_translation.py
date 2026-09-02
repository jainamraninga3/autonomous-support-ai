"""Unit tests for `translate_query_to_english`, with a fake LLM client."""

import pytest

from app.rag.query_translation import ALREADY_ENGLISH, translate_query_to_english


class _FakeLLMClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    async def generate_reply(self, message: str) -> str:
        self.calls.append(message)
        return self.reply


@pytest.mark.asyncio
async def test_returns_the_translation_for_a_non_english_query() -> None:
    llm = _FakeLLMClient("How many days of sick leave per year?")

    result = await translate_query_to_english("एक साल में कितनी सिक लीव मिलती है?", llm)

    assert result == "How many days of sick leave per year?"


@pytest.mark.asyncio
async def test_returns_none_for_an_already_english_query() -> None:
    """None means "skip the second retrieval pass" — searching twice with
    the same English text would just cost a Weaviate round trip to find
    the identical chunks."""
    llm = _FakeLLMClient(ALREADY_ENGLISH)

    assert await translate_query_to_english("How many sick leaves do I get?", llm) is None


@pytest.mark.asyncio
async def test_returns_none_on_an_empty_reply() -> None:
    assert await translate_query_to_english("कुछ सवाल", _FakeLLMClient("   ")) is None


@pytest.mark.asyncio
async def test_returns_none_when_the_reply_is_implausibly_long() -> None:
    """A reply far longer than the query usually means the model answered
    the question instead of translating it. Falling back to None degrades
    to single-pass retrieval rather than searching with a paragraph."""
    llm = _FakeLLMClient("Sick leave " * 200)

    assert await translate_query_to_english("सिक लीव?", llm) is None


@pytest.mark.asyncio
async def test_strips_surrounding_quotes() -> None:
    llm = _FakeLLMClient('"How many days of sick leave per year?"')

    result = await translate_query_to_english("सिक लीव कितनी?", llm)

    assert result == "How many days of sick leave per year?"
