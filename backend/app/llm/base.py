"""LLM provider abstraction.

This decouples the rest of the application from any specific LLM vendor.
`GroqLLMClient` is the first real provider; `StubLLMClient` (echo) remains
available for tests/dev environments that don't have a `GROQ_API_KEY`.
"""

from abc import ABC, abstractmethod

import groq
from groq import AsyncGroq

from app.core.exceptions import RateLimitError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract interface that every LLM provider implementation must follow."""

    @abstractmethod
    async def generate_reply(self, message: str) -> str:
        """Generate a reply for a given user message."""
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """Placeholder LLM client used when no real provider is configured.

    Does not call any external LLM. Exists purely to prove the
    Service -> LLM abstraction boundary works end to end, and as a
    dependency-free fallback for local dev / tests.
    """

    async def generate_reply(self, message: str) -> str:
        return f"Echo: {message}"


class GroqLLMClient(LLMClient):
    """LLM client backed by the Groq API (OpenAI-compatible chat completions).

    Accepts one or more API keys. On a rate limit (429 — e.g. a daily
    token quota exhausted, see docs/PROJECT_LOG.md) it automatically
    retries the same request against the next key instead of failing the
    request. `_current_index` is "sticky": once a key is found to work,
    later calls start there directly rather than re-trying already-dead
    keys from the front every time.
    """

    def __init__(self, api_keys: list[str], model: str) -> None:
        if not api_keys:
            raise ValueError("GroqLLMClient requires at least one API key")
        self.model = model
        self._clients = [AsyncGroq(api_key=key) for key in api_keys]
        self._current_index = 0

    async def generate_reply(self, message: str) -> str:
        last_exc: Exception | None = None

        for offset in range(len(self._clients)):
            index = (self._current_index + offset) % len(self._clients)
            client = self._clients[index]
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": message}],
                )
            except groq.RateLimitError as exc:
                logger.warning(
                    "Groq key #%d rate-limited for model=%s, trying next key: %s", index, self.model, exc
                )
                last_exc = exc
                continue
            except groq.APIError as exc:
                # Not a rate limit — a different key wouldn't help, so
                # surface immediately rather than burning through the
                # rest of the keys for nothing.
                logger.error("Groq API error for model=%s: %s", self.model, exc)
                raise ServiceUnavailableError(f"Groq API error: {exc}") from exc

            self._current_index = index
            reply = response.choices[0].message.content
            if not reply:
                logger.warning("Groq returned an empty reply for model=%s", self.model)
                return ""
            return reply

        # Surfaced as a real error the caller can see and log, rather
        # than falling through to the app's generic 500 handler — this
        # specific error (a Groq daily/per-minute token quota being
        # exhausted) was previously indistinguishable from any other
        # unhandled exception until someone dug through the server's own
        # logs. See docs/PROJECT_LOG.md.
        raise RateLimitError(
            f"All {len(self._clients)} configured Groq API key(s) are rate-limited: {last_exc}"
        ) from last_exc
