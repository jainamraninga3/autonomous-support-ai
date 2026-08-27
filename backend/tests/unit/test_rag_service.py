"""Unit tests for RagQueryService — fake Weaviate client, fake embedder/
reranker (monkeypatched), fake LLM client. No real Weaviate connection or
model needed.
"""

from dataclasses import dataclass

import pytest

from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.services.rag_service import RagQueryService


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


class _FakeLLMClient:
    """Returns a fixed, realistic-looking answer rather than echoing
    `message` — echoing would include the grounded-answer prompt's own
    instruction text (which mentions the NOT_FOUND_IN_CONTEXT sentinel
    by name) in the reply, falsely tripping `was_answerable` detection."""

    async def generate_reply(self, message: str) -> str:
        return "Employees are entitled to 20 days of annual leave. [Source 1]"


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 for _ in texts]


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


@pytest.mark.asyncio
async def test_ask_raises_not_found_when_collection_missing() -> None:
    service = RagQueryService(weaviate_client=_FakeWeaviateClient(exists=False), llm_client=_FakeLLMClient())

    with pytest.raises(NotFoundError):
        await service.ask("what is the leave policy?")


@pytest.mark.asyncio
async def test_ask_raises_service_unavailable_when_weaviate_client_missing() -> None:
    service = RagQueryService(weaviate_client=None, llm_client=_FakeLLMClient())

    with pytest.raises(ServiceUnavailableError):
        await service.ask("what is the leave policy?")


@pytest.mark.asyncio
async def test_ask_returns_grounded_answer_with_citations(monkeypatch) -> None:
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.services.rag_service.get_reranker", lambda: _FakeReranker())

    service = RagQueryService(
        weaviate_client=_FakeWeaviateClient(objects=[_fake_object(0, 0.9)]),
        llm_client=_FakeLLMClient(),
    )

    result = await service.ask("what is the leave policy?", do_rerank=True, top_k=5)

    assert result.was_answerable is True
    assert result.answer == "Employees are entitled to 20 days of annual leave. [Source 1]"
    assert len(result.citations) == 1
    assert result.citations[0].document_name == "policy.pdf"


@pytest.mark.asyncio
async def test_ask_without_rerank_never_loads_the_reranker(monkeypatch) -> None:
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: _FakeEmbedder())

    def _boom():
        raise AssertionError("reranker should not be loaded when do_rerank=False")

    monkeypatch.setattr("app.services.rag_service.get_reranker", _boom)

    service = RagQueryService(
        weaviate_client=_FakeWeaviateClient(objects=[_fake_object(0, 0.9)]),
        llm_client=_FakeLLMClient(),
    )

    result = await service.ask("what is the leave policy?", do_rerank=False, top_k=5)

    assert result.was_answerable is True


@pytest.mark.asyncio
async def test_ask_with_no_matching_chunks_skips_llm_and_reports_unanswerable(monkeypatch) -> None:
    # embed_query_fn is called to build the hybrid-search request regardless
    # of how many results come back, so this still needs a fake embedder —
    # otherwise it would try to load the real BGE-M3 model.
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.services.rag_service.get_reranker", lambda: _FakeReranker())

    service = RagQueryService(
        weaviate_client=_FakeWeaviateClient(objects=[]),
        llm_client=_FakeLLMClient(),
    )

    result = await service.ask("unrelated question?")

    assert result.was_answerable is False
    assert result.citations == []
