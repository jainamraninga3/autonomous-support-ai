"""LLM provider abstraction.

This decouples the rest of the application from any specific LLM vendor.
`GroqLLMClient` is the first real provider; `StubLLMClient` (echo) remains
available for tests/dev environments that don't have a `GROQ_API_KEY`.
"""

from abc import ABC, abstractmethod

from groq import AsyncGroq

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
    """LLM client backed by the Groq API (OpenAI-compatible chat completions)."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = AsyncGroq(api_key=api_key)

    async def generate_reply(self, message: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
        )
        reply = response.choices[0].message.content
        if not reply:
            logger.warning("Groq returned an empty reply for model=%s", self.model)
            return ""
        return reply
