"""Unit tests for `hybrid_search`, with a fake Weaviate collection and a
fake query embedder — no real Weaviate connection or model needed.
"""

from dataclasses import dataclass, field

from app.rag.retrieval.hybrid_search import hybrid_search


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
        self.last_call_kwargs: dict = {}

    def hybrid(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeHybridResponse(objects=self._objects)


class _FakeCollection:
    def __init__(self, objects: list) -> None:
        self.query = _FakeQueryHandle(objects)


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


def _fake_embed_query(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def test_hybrid_search_maps_results_in_order() -> None:
    collection = _FakeCollection([_fake_object(0, 0.9), _fake_object(1, 0.5)])

    results = hybrid_search(collection, "leave policy", _fake_embed_query, limit=30, alpha=0.5)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-0"
    assert results[0].score == 0.9
    assert results[1].document_name == "policy.pdf"


def test_hybrid_search_passes_query_vector_and_params_through() -> None:
    collection = _FakeCollection([])
    hybrid_search(collection, "leave policy", _fake_embed_query, limit=10, alpha=0.7)

    kwargs = collection.query.last_call_kwargs
    assert kwargs["query"] == "leave policy"
    assert kwargs["vector"] == [0.1, 0.2, 0.3]
    assert kwargs["limit"] == 10
    assert kwargs["alpha"] == 0.7


def test_hybrid_search_empty_query_short_circuits_without_calling_weaviate() -> None:
    collection = _FakeCollection([_fake_object(0, 0.9)])

    results = hybrid_search(collection, "   ", _fake_embed_query, limit=30)

    assert results == []
    assert collection.query.last_call_kwargs == {}  # never called


def test_hybrid_search_no_results_returns_empty_list() -> None:
    collection = _FakeCollection([])
    assert hybrid_search(collection, "no matches", _fake_embed_query) == []
