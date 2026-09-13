"""Unit tests for SessionContextBuilder."""

import pytest
from app.services.session_context_builder import SessionContextBuilder


class DummyMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


def test_build_history_tuples_from_dicts() -> None:
    dict_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    tuples = SessionContextBuilder.build_history_tuples(dict_messages)
    assert tuples == [("user", "Hello"), ("assistant", "Hi there!")]


def test_build_history_tuples_from_objects() -> None:
    obj_messages = [
        DummyMessage("user", "What is the policy?"),
        DummyMessage("assistant", "The policy states..."),
    ]
    tuples = SessionContextBuilder.build_history_tuples(obj_messages)
    assert tuples == [("user", "What is the policy?"), ("assistant", "The policy states...")]


def test_build_history_tuples_empty() -> None:
    assert SessionContextBuilder.build_history_tuples([]) == []
