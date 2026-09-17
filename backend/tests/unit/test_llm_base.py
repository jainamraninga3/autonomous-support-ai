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

import groq
import httpx
import pytest
from groq import APIError, RateLimitError as GroqRateLimitError

from app.core.exceptions import RateLimitError, ServiceUnavailableError
from app.llm import base
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


def _rate_limit_error_with_wait(message: str) -> GroqRateLimitError:
    """A 429 whose body states how long to wait, the way Groq's does."""
    return GroqRateLimitError(message, response=_fake_response(429), body=None)


@pytest.mark.asyncio
async def test_waits_out_a_per_minute_limit_instead_of_failing(monkeypatch) -> None:
    """Groq meters tokens-per-MINUTE and tokens-per-DAY under the same
    429. A per-minute limit clears in seconds, and failing the user's
    question over a 4-second wait is absurd — observed live on
    2026-09-16, mid-run, with `Limit 8000, Used 7956 ... try again in
    4.4775s`.
    """
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", _fake_sleep)

    client = _client_with_scripts([
        [_rate_limit_error_with_wait(
            "Rate limit reached ... tokens per minute (TPM): Limit 8000, Used 7956, "
            "Requested 641. Please try again in 4.4775s."
        ), "recovered"],
    ])

    assert await client.generate_reply("hello") == "recovered"
    assert slept, "a 4.5s limit must be waited out, not raised"
    assert 4.4 < slept[0] < 5.0, slept


@pytest.mark.asyncio
async def test_does_not_wait_out_a_daily_limit(monkeypatch) -> None:
    """A per-DAY limit says minutes, not seconds. Sleeping on it turns a
    fast, clear error into a hung request, so it must fail immediately."""
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", _fake_sleep)

    client = _client_with_scripts([
        [_rate_limit_error_with_wait(
            "Rate limit reached ... tokens per day (TPD): Limit 200000, Used 199235, "
            "Requested 1557. Please try again in 5m42.144s."
        )],
    ])

    with pytest.raises(RateLimitError):
        await client.generate_reply("hello")
    assert slept == [], "a 5-minute wait must not be slept on"


def test_retry_after_parses_both_shapes() -> None:
    assert base._retry_after_seconds(
        _rate_limit_error_with_wait("Please try again in 4.4775s.")
    ) == pytest.approx(4.4775)
    assert base._retry_after_seconds(
        _rate_limit_error_with_wait("Please try again in 5m42.144s.")
    ) == pytest.approx(342.144)
    assert base._retry_after_seconds(_rate_limit_error()) is None


@pytest.mark.asyncio
async def test_a_timeout_is_retried_not_raised(monkeypatch) -> None:
    """A timeout says nothing about the request or the key — the same call
    may succeed a moment later. Observed 2026-09-17: a 45-turn regression
    run died at turn 45 on `Groq API error: Request timed out.`, because
    timeouts fell into the generic APIError branch whose reasoning ("a
    property of the REQUEST, another key would not help") is right for a
    bad model name and wrong for this.
    """
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", _fake_sleep)

    timeout = groq.APITimeoutError(
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )
    client = _client_with_scripts([[timeout, "recovered"]])

    assert await client.generate_reply("hello") == "recovered"
    assert slept, "a timeout must be waited out and retried"


@pytest.mark.asyncio
async def test_a_bad_request_is_still_raised_immediately(monkeypatch) -> None:
    """The retry must NOT swallow genuine request errors. A bad model name
    fails identically on every key, so cycling just multiplies latency."""
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", _fake_sleep)

    bad = APIError("model not found", request=httpx.Request("POST", "https://api.groq.com/x"), body=None)
    client = _client_with_scripts([[bad, "never reached"]])

    with pytest.raises(ServiceUnavailableError):
        await client.generate_reply("hello")
    assert slept == []
