"""Unit tests for the full LangGraph workflow (classify -> general | RAG
-> verify), using a scripted fake LLM client and a fake Weaviate client
— no real model or Weaviate connection needed.

Replaces the old placeholder-graph test: `build_graph()` used to compile
a single echo node just to prove LangGraph worked at all; it now takes
real dependencies and runs the full plan.md pipeline.
"""

from dataclasses import dataclass

import pytest

from app.graph.state import initial_state
from app.graph.workflow import build_graph
from app.llm.base import LLMClient


@dataclass
class _FakeMetadata:
    score: float


@dataclass
class _FakeObject:
    properties: dict
    metadata: _FakeMetadata


@dataclass
class _FakeHybridResponse:
    objects: list


class _FakeQueryHandle:
    def __init__(self, objects: list) -> None:
        self._objects = objects

    def hybrid(self, **kwargs):
        return _FakeHybridResponse(objects=self._objects)


class _FakeCollection:
    def __init__(self, objects: list) -> None:
        self.query = _FakeQueryHandle(objects)


class _FakeCollections:
    def __init__(self, exists: bool, collection: _FakeCollection) -> None:
        self._exists = exists
        self._collection = collection

    def exists(self, name: str) -> bool:
        return self._exists

    def get(self, name: str):
        return self._collection


class _FakeWeaviateClient:
    def __init__(self, exists: bool = True, objects: list | None = None) -> None:
        self.collections = _FakeCollections(exists, _FakeCollection(objects or []))


def _fake_object(index: int, score: float) -> _FakeObject:
    return _FakeObject(
        properties={
            "chunk_id": f"chunk-{index}",
            "document_id": "doc-1",
            "document_name": "policy.pdf",
            "version": 1,
            "chunk_index": index,
            "start_page": 1,
            "end_page": 1,
            "content": f"content {index}",
        },
        metadata=_FakeMetadata(score=score),
    )


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 for _ in texts]


class _ScriptedLLMClient(LLMClient):
    """Returns a canned reply based on a distinctive substring in the
    prompt — each of the four prompt types this graph sends
    (classification, rewrite, grounded-answer, verification) has one."""

    def __init__(self, responses: dict[str, str], default: str = "a general reply") -> None:
        self.responses = responses
        self.default = default
        self.calls: list[str] = []

    async def generate_reply(self, message: str) -> str:
        self.calls.append(message)
        for marker, reply in self.responses.items():
            if marker in message:
                return reply
        return self.default


@pytest.mark.asyncio
async def test_general_query_skips_retrieval_entirely() -> None:
    llm = _ScriptedLLMClient({"GENERAL or RAG_REQUIRED": "GENERAL"})
    graph = build_graph(llm_client=llm, weaviate_client=None)

    result = await graph.ainvoke(initial_state("what is 2 + 2?"))

    assert result["classification"] == "GENERAL"
    assert result["response"] == "a general reply"
    assert result["chunks"] == []
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_rag_query_with_supported_answer_returns_the_generated_answer(monkeypatch) -> None:
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.graph.workflow.get_reranker", lambda: _FakeReranker())

    llm = _ScriptedLLMClient(
        {
            "GENERAL or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "rewritten leave policy question",
            "SUPPORTED or UNSUPPORTED": "SUPPORTED\nEvery claim matches the context.",
        },
        default="Employees get 20 days of leave. [Source 1]",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert result["classification"] == "RAG_REQUIRED"
    assert result["rewritten_query"] == "rewritten leave policy question"
    assert result["verified"] is True
    assert result["response"] == "Employees get 20 days of leave. [Source 1]"
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_rag_query_with_unsupported_answer_returns_the_refusal(monkeypatch) -> None:
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.graph.workflow.get_reranker", lambda: _FakeReranker())

    llm = _ScriptedLLMClient(
        {
            "GENERAL or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "rewritten query",
            "SUPPORTED or UNSUPPORTED": "UNSUPPORTED\nThe answer invents a number not in the context.",
        },
        default="some ungrounded answer",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert result["verified"] is False
    assert "could not fully verify" in result["response"]


@pytest.mark.asyncio
async def test_rag_query_with_no_weaviate_client_returns_not_found_without_crashing() -> None:
    llm = _ScriptedLLMClient({"GENERAL or RAG_REQUIRED": "RAG_REQUIRED", "Rewritten query:": "rewritten query"})
    graph = build_graph(llm_client=llm, weaviate_client=None)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert result["chunks"] == []
    assert result["verified"] is False
    assert "does not contain an answer" in result["response"]
