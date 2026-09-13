"""Formatter utility for session conversation context."""

from typing import Any


class SessionContextBuilder:
    """Converts message records (from Redis dicts or SQLAlchemy models)
    into the list-of-tuples format expected by LangGraph graph state."""

    @staticmethod
    def build_history_tuples(messages: list[Any]) -> list[tuple[str, str]]:
        """Format messages into [(role, content), ...] tuples.

        Supports dicts (`{"role": "user", "content": "..."}`) and ORM Message instances.
        """
        history: list[tuple[str, str]] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            if role and content:
                history.append((role, content))
        return history
