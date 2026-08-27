"""Unit tests for GroqLLMClient's multi-key fallback and error
translation.

On a rate limit, a Groq SDK error must not simply bubble up (or fail the
request) — the client should retry the same request against the next
configured key, and only raise once every key is exhausted. Any other
Groq error should surface immediately as one of our own `AppException`
subclasses (so it maps to a clean, diagnosable HTTP status) instead of
an opaque 500. See docs/PROJECT_LOG.md: this was the actual root cause
of `rag_chat_test` runs failing with generic `internal_error` 500s — a
Groq daily token quota being exhausted (`groq.RateLimitError`).
"""

import httpx
import pytest
from groq import APIError, RateLimitError as GroqRateLimitError

from app.core.exceptions import RateLimitError, ServiceUnavailableError
from app.llm.base import GroqLLMClient


def _fake_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


def _rate_limit_error() -> GroqRateLimitError:
    return GroqRateLimitError("rate limit reached", response=_fake_response(429), body=None)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _ScriptedCompletions:
    """Each call to `create()` pops the next item from `script` — either
    an exception to raise or a reply string to return."""

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeCompletionResponse(item)


def _client_with_scripts(scripts: list[list]) -> GroqLLMClient:
    """One script (list of exceptions/replies) per configured key,
    consumed in order as that key is called."""
    client = GroqLLMClient(api_keys=[f"key-{i}" for i in range(len(scripts))], model="openai/gpt-oss-120b")
    for groq_client, script in zip(client._clients, scripts, strict=True):
        groq_client.chat.completions = _ScriptedCompletions(script)
    return client


@pytest.mark.asyncio
async def test_first_key_succeeds_without_touching_others() -> None:
    client = _client_with_scripts([["the answer"], [_rate_limit_error()]])

    reply = await client.generate_reply("hello")

    assert reply == "the answer"
    assert client._clients[1].chat.completions.call_count == 0


@pytest.mark.asyncio
async def test_falls_through_to_next_key_on_rate_limit() -> None:
    client = _client_with_scripts([[_rate_limit_error()], ["fallback answer"]])

    reply = await client.generate_reply("hello")

    assert reply == "fallback answer"


@pytest.mark.asyncio
async def test_sticky_index_skips_dead_key_on_next_call() -> None:
    client = _client_with_scripts([[_rate_limit_error()], ["first call", "second call"]])

    first = await client.generate_reply("hello")
    second = await client.generate_reply("hello again")

    assert first == "first call"
    assert second == "second call"
    # Key 0 was only ever tried once (the initial failure) — later calls
    # go straight to key 1 instead of re-trying a known-dead key.
    assert client._clients[0].chat.completions.call_count == 1


@pytest.mark.asyncio
async def test_all_keys_rate_limited_raises_app_rate_limit_error() -> None:
    client = _client_with_scripts([[_rate_limit_error()], [_rate_limit_error()]])

    with pytest.raises(RateLimitError):
        await client.generate_reply("hello")


@pytest.mark.asyncio
async def test_non_rate_limit_api_error_raises_immediately_without_trying_other_keys() -> None:
    groq_exc = APIError("boom", request=httpx.Request("POST", "https://api.groq.com/x"), body=None)
    client = _client_with_scripts([[groq_exc], ["never reached"]])

    with pytest.raises(ServiceUnavailableError):
        await client.generate_reply("hello")

    assert client._clients[1].chat.completions.call_count == 0


def test_constructor_rejects_empty_key_list() -> None:
    with pytest.raises(ValueError):
        GroqLLMClient(api_keys=[], model="openai/gpt-oss-120b")
