"""Unit tests for `write_chunks_to_weaviate`, with a fake Weaviate collection
and a fake embedder — no real Weaviate connection or model needed.
"""

import uuid
from dataclasses import dataclass, field

import pytest

from app.rag.ingestion.vector_writer import write_chunks_to_weaviate

DOCUMENT_ID = uuid.uuid4()


@dataclass
class _FakeInsertResult:
    has_errors: bool = False
    errors: dict = field(default_factory=dict)


class _FakeDataHandle:
    def __init__(self) -> None:
        self.inserted_objects: list = []
        self.result = _FakeInsertResult()

    def insert_many(self, objects):
        self.inserted_objects.extend(objects)
        return self.result


class _FakeCollection:
    def __init__(self) -> None:
        self.data = _FakeDataHandle()


def _fake_chunks(n: int) -> list[dict]:
    return [
        {"index": i, "text": f"chunk {i}", "token_count": 10, "start_page": 1, "end_page": 1} for i in range(n)
    ]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3] for _ in texts]


def test_write_chunks_returns_count_written() -> None:
    collection = _FakeCollection()
    written = write_chunks_to_weaviate(
        collection=collection,
        document_id=DOCUMENT_ID,
        document_name="policy.pdf",
        version=1,
        chunks=_fake_chunks(3),
        embed_texts_fn=_fake_embed,
    )
    assert written == 3
    assert len(collection.data.inserted_objects) == 3


def test_write_chunks_empty_list_is_a_noop() -> None:
    collection = _FakeCollection()
    written = write_chunks_to_weaviate(
        collection=collection,
        document_id=DOCUMENT_ID,
        document_name="policy.pdf",
        version=1,
        chunks=[],
        embed_texts_fn=_fake_embed,
    )
    assert written == 0
    assert collection.data.inserted_objects == []


def test_chunk_uuid_is_deterministic_across_calls() -> None:
    collection_a = _FakeCollection()
    collection_b = _FakeCollection()
    chunks = _fake_chunks(2)

    write_chunks_to_weaviate(collection_a, DOCUMENT_ID, "policy.pdf", 1, chunks, _fake_embed)
    write_chunks_to_weaviate(collection_b, DOCUMENT_ID, "policy.pdf", 1, chunks, _fake_embed)

    uuids_a = [obj.uuid for obj in collection_a.data.inserted_objects]
    uuids_b = [obj.uuid for obj in collection_b.data.inserted_objects]
    assert uuids_a == uuids_b  # same document_id/version/index -> same uuid -> upsert, not duplicate


def test_write_chunks_raises_on_vector_count_mismatch() -> None:
    collection = _FakeCollection()
    with pytest.raises(ValueError):
        write_chunks_to_weaviate(
            collection=collection,
            document_id=DOCUMENT_ID,
            document_name="policy.pdf",
            version=1,
            chunks=_fake_chunks(3),
            embed_texts_fn=lambda texts: [[0.1, 0.2]],  # wrong count on purpose
        )


def test_write_chunks_raises_when_weaviate_reports_errors() -> None:
    collection = _FakeCollection()
    collection.data.result = _FakeInsertResult(has_errors=True, errors={0: "boom"})
    with pytest.raises(RuntimeError):
        write_chunks_to_weaviate(
            collection=collection,
            document_id=DOCUMENT_ID,
            document_name="policy.pdf",
            version=1,
            chunks=_fake_chunks(1),
            embed_texts_fn=_fake_embed,
        )
