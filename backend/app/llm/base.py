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

    Accepts one or more API keys and moves to the next one whenever the
    failure is a property of the KEY rather than of the request:

    - **429 rate limit** — e.g. a token quota exhausted.
    - **401 / 403 rejected key** — revoked, or from the wrong account.

    Anything else (a bad model name, a malformed request) fails
    immediately, because another key would fail identically and trying
    them all just multiplies the latency of a certain failure.

    `_current_index` is "sticky": once a key is found to work, later
    calls start there directly rather than re-trying already-dead keys
    from the front every time. That is what keeps a revoked primary key
    to a single wasted round trip per process, instead of one on every
    request.
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
            except (groq.AuthenticationError, groq.PermissionDeniedError) as exc:
                # A REVOKED OR WRONG KEY IS PER-KEY, so skip it exactly
                # like a rate limit rather than failing the request.
                #
                # This block used to be absent, and both of these fell
                # into the generic `APIError` handler below on the
                # reasoning that "a different key wouldn't help". That is
                # true for a bad model name or a malformed request; it is
                # exactly BACKWARDS for an authentication error, which is
                # a statement about ONE credential and says nothing about
                # the others.
                #
                # Observed 2026-09-08: `GROQ_API_KEY` had been revoked
                # while `GROQ_API_1/_2/_3` all still worked, and every
                # request failed with a 401 — three good keys sitting
                # unused behind one dead one, because the primary is
                # tried first. WARNING rather than ERROR: the request is
                # about to succeed on another key, so this is a
                # configuration problem to notice, not a failure.
                logger.warning(
                    "Groq key #%d was rejected (%s) for model=%s — skipping to the "
                    "next key. Remove or replace it: a dead key costs a wasted "
                    "round trip on every cold start.",
                    index,
                    type(exc).__name__,
                    self.model,
                )
                last_exc = exc
                continue
            except groq.APIError as exc:
                # Everything else — a bad model name, a malformed
                # request, a server-side fault. These are properties of
                # the REQUEST, not of the key, so another key genuinely
                # would not help and trying the rest just multiplies the
                # latency of a guaranteed failure.
                logger.error("Groq API error for model=%s: %s", self.model, exc)
                raise ServiceUnavailableError(f"Groq API error: {exc}") from exc

            self._current_index = index
            reply = response.choices[0].message.content
            if not reply:
                logger.warning("Groq returned an empty reply for model=%s", self.model)
                return ""
            return reply

        # Every key was skipped. Surfaced as a real error the caller can
        # see and log, rather than falling through to the app's generic
        # 500 handler — a Groq token quota being exhausted was previously
        # indistinguishable from any other unhandled exception until
        # someone dug through the server's own logs. See
        # docs/PROJECT_LOG.md.
        #
        # WHICH error matters, because the two need different actions: a
        # rate limit means wait, a rejected key means go and fix the
        # configuration. Reporting an auth failure as "rate-limited"
        # would send someone off to check quotas that are perfectly fine.
        if isinstance(last_exc, groq.RateLimitError):
            raise RateLimitError(
                f"All {len(self._clients)} configured Groq API key(s) are rate-limited: {last_exc}"
            ) from last_exc
        raise ServiceUnavailableError(
            f"All {len(self._clients)} configured Groq API key(s) were rejected "
            f"(revoked, or from the wrong account): {last_exc}"
        ) from last_exc
