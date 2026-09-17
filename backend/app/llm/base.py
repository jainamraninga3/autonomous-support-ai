"""LLM provider abstraction.

This decouples the rest of the application from any specific LLM vendor.
`GroqLLMClient` is the first real provider; `StubLLMClient` (echo) remains
available for tests/dev environments that don't have a `GROQ_API_KEY`.
"""

import asyncio
import re
from abc import ABC, abstractmethod

import groq
from groq import AsyncGroq

from app.core.exceptions import RateLimitError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Groq meters two different things under the same 429, and they need
# opposite responses. Tokens-per-MINUTE clears in seconds — failing the
# user's question over a 4-second wait is absurd. Tokens-per-DAY does
# not clear for hours, and sleeping on it just converts a fast, clear
# error into a hung request. The retry-after value is what separates
# them, so the wait itself decides whether to wait.
_MAX_RETRY_WAIT_SECONDS = 20.0
# Timeouts and 5xx carry no retry-after, so they get a fixed short backoff.
_TRANSIENT_RETRY_SECONDS = 2.0
_MAX_TOTAL_WAIT_SECONDS = 45.0

# "Please try again in 4.4775s" / "in 5m42.144s"
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def _retry_after_seconds(exc: Exception) -> float | None:
    """How long Groq says to wait, or None if it did not say.

    Prefers the `retry-after` header; falls back to the sentence in the
    error body, which is the only place the sub-second value appears.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    raw = header.get("retry-after")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass

    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        minutes, seconds = match.groups()
        return float(seconds) + (60.0 * int(minutes) if minutes else 0.0)
    return None


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
        waited = 0.0
        while True:
            reply, last_exc, retry_after = await self._try_every_key(message)
            if last_exc is None:
                return reply

            # Every key refused. A per-MINUTE limit says "4.5s" and is
            # worth sleeping through; a per-DAY limit says "5m42s" and is
            # not. Anything without a stated wait is not slept on either.
            retryable = isinstance(
                last_exc,
                (groq.RateLimitError, groq.APITimeoutError,
                 groq.APIConnectionError, groq.InternalServerError),
            )
            if (
                retryable
                and retry_after is not None
                and retry_after <= _MAX_RETRY_WAIT_SECONDS
                and waited + retry_after <= _MAX_TOTAL_WAIT_SECONDS
            ):
                pause = retry_after + 0.25  # clear the window, don't race it
                logger.info(
                    "All Groq keys are rate-limited for %.1fs — waiting and retrying "
                    "(%.1fs of %.1fs budget used)",
                    retry_after,
                    waited,
                    _MAX_TOTAL_WAIT_SECONDS,
                )
                await asyncio.sleep(pause)
                waited += pause
                continue

            self._raise_for(last_exc, waited)

    async def _try_every_key(
        self, message: str
    ) -> tuple[str, Exception | None, float | None]:
        """One pass over all keys. Returns (reply, last error, shortest wait).

        `last error` is None when a key succeeded. The shortest wait is
        the soonest any key says it will accept traffic again — waiting
        longer than that helps nobody.
        """
        last_exc: Exception | None = None
        soonest_retry: float | None = None

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
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None and (soonest_retry is None or retry_after < soonest_retry):
                    soonest_retry = retry_after
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
            except (groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError) as exc:
                # TRANSIENT. A timeout, a dropped connection or a 5xx says
                # nothing about the request or the key — the same call may
                # well succeed a moment later, so it is retried like a rate
                # limit rather than failing the user's question.
                #
                # Observed 2026-09-17: a 45-turn regression run died at turn
                # 45 on `Groq API error: Request timed out.` These fell into
                # the generic `APIError` branch below, whose reasoning ("a
                # property of the REQUEST, another key would not help") is
                # correct for a bad model name and simply wrong for a
                # timeout. No stated retry-after exists for these, so
                # `_TRANSIENT_RETRY_SECONDS` is used instead.
                logger.warning(
                    "Groq key #%d hit a transient failure (%s) for model=%s, trying next key: %s",
                    index, type(exc).__name__, self.model, exc,
                )
                last_exc = exc
                if soonest_retry is None or _TRANSIENT_RETRY_SECONDS < soonest_retry:
                    soonest_retry = _TRANSIENT_RETRY_SECONDS
                continue
            except groq.APIError as exc:
                # Everything else — a bad model name, a malformed
                # request. These are properties of
                # the REQUEST, not of the key, so another key genuinely
                # would not help and trying the rest just multiplies the
                # latency of a guaranteed failure.
                logger.error("Groq API error for model=%s: %s", self.model, exc)
                raise ServiceUnavailableError(f"Groq API error: {exc}") from exc

            self._current_index = index
            reply = response.choices[0].message.content
            if not reply:
                logger.warning("Groq returned an empty reply for model=%s", self.model)
                return "", None, None
            return reply, None, None

        # Every key was skipped. The caller decides whether the failure is
        # worth waiting out; see `generate_reply`.
        return "", last_exc, soonest_retry

    def _raise_for(self, last_exc: Exception, waited: float) -> None:
        """Surface an exhausted rotation as a real, actionable error.

        Rather than falling through to the app's generic 500 handler — a
        Groq token quota being exhausted was previously indistinguishable
        from any other unhandled exception until someone dug through the
        server's own logs. See docs/PROJECT_LOG.md.

        WHICH error matters, because the two need different actions: a
        rate limit means wait, a rejected key means go and fix the
        configuration. Reporting an auth failure as "rate-limited" would
        send someone off to check quotas that are perfectly fine.
        """
        if isinstance(last_exc, groq.RateLimitError):
            waited_note = f" after waiting {waited:.1f}s" if waited else ""
            raise RateLimitError(
                f"All {len(self._clients)} configured Groq API key(s) are "
                f"rate-limited{waited_note}: {last_exc}"
            ) from last_exc
        raise ServiceUnavailableError(
            f"All {len(self._clients)} configured Groq API key(s) were rejected "
            f"(revoked, or from the wrong account): {last_exc}"
        ) from last_exc
