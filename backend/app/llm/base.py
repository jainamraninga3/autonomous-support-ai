"""LLM provider abstraction.

This decouples the rest of the application from any specific LLM vendor.
A real provider (OpenAI, Anthropic, local model, etc.) will implement this
interface in a later phase. For Phase 1 only the abstraction and a stub
implementation exist so the API -> Service -> LLM flow can be exercised.
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface that every LLM provider implementation must follow."""

    @abstractmethod
    async def generate_reply(self, message: str) -> str:
        """Generate a reply for a given user message."""
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """Placeholder LLM client used until a real provider is wired in.

    Does not call any external LLM. Exists purely to prove the
    Service -> LLM abstraction boundary works end to end.
    """

    async def generate_reply(self, message: str) -> str:
        return f"Echo: {message}"
