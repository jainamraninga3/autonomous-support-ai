"""Unit tests for `read_chunks` and `is_already_embedded` — fake Weaviate client, no real
Weaviate connection needed.
"""

import contextlib
import uuid
from types import SimpleNamespace

import pytest

from app.rag.ingestion.embedder import is_already_embedded


def _fake_session_factory():
    """A session factory whose sessions are inert — the repository is
    monkeypatched, so nothing actually touches a database."""

    @contextlib.asynccontextmanager
    async def _session():
        yield object()

    return _session

DOCUMENT_ID = uuid.uuid4()


class _FakeQueryHandle:
    def __init__(self, objects: list) -> None:
        self._objects = objects

    def fetch_objects(self, filters, limit):
        return SimpleNamespace(objects=self._objects)


class _FakeCollection:
    def __init__(self, objects: list) -> None:
        self.query = _FakeQueryHandle(objects)


class _FakeCollectionsHandle:
    def __init__(self, exists: bool, objects: list) -> None:
        self._exists = exists
        self._collection = _FakeCollection(objects)

    def exists(self, name):
        return self._exists

    def get(self, name):
        return self._collection


class _FakeWeaviateClient:
    def __init__(self, exists: bool = True, objects: list | None = None) -> None:
        self.collections = _FakeCollectionsHandle(exists=exists, objects=objects or [])


def test_is_already_embedded_true_when_objects_found() -> None:
    client = _FakeWeaviateClient(exists=True, objects=[object()])
    assert is_already_embedded(client, DOCUMENT_ID, 1) is True


def test_is_already_embedded_false_when_no_objects_found() -> None:
    client = _FakeWeaviateClient(exists=True, objects=[])
    assert is_already_embedded(client, DOCUMENT_ID, 1) is False


def test_is_already_embedded_false_when_collection_missing() -> None:
    client = _FakeWeaviateClient(exists=False, objects=[object()])
    assert is_already_embedded(client, DOCUMENT_ID, 1) is False


def test_is_already_embedded_false_when_client_is_none() -> None:
    assert is_already_embedded(None, DOCUMENT_ID, 1) is False


@pytest.mark.asyncio
async def test_read_chunks_returns_postgres_rows_in_document_order(monkeypatch) -> None:
    """Chunk text lives in PostgreSQL, not on disk. It used to be read
    from a JSON file whose path was stored in the database, which meant a
    deleted directory silently destroyed the text."""
    from app.rag.ingestion import embedder

    rows = [
        SimpleNamespace(chunk_index=0, text="first", token_count=5, start_page=1, end_page=1),
        SimpleNamespace(chunk_index=1, text="second", token_count=7, start_page=1, end_page=2),
    ]

    class _FakeRepository:
        def __init__(self, session) -> None:
            pass

        async def get_version(self, document_id, version):
            return SimpleNamespace(id=uuid.uuid4(), version=version)

        async def get_chunks(self, document_version_id):
            return rows

    monkeypatch.setattr(embedder, "DocumentRepository", _FakeRepository)

    chunks = await embedder.read_chunks(_fake_session_factory(), uuid.uuid4(), 1)

    assert chunks == [
        {"index": 0, "text": "first", "token_count": 5, "start_page": 1, "end_page": 1},
        {"index": 1, "text": "second", "token_count": 7, "start_page": 1, "end_page": 2},
    ]


@pytest.mark.asyncio
async def test_read_chunks_raises_for_a_missing_version(monkeypatch) -> None:
    from app.rag.ingestion import embedder

    class _FakeRepository:
        def __init__(self, session) -> None:
            pass

        async def get_version(self, document_id, version):
            return None

    monkeypatch.setattr(embedder, "DocumentRepository", _FakeRepository)

    with pytest.raises(FileNotFoundError):
        await embedder.read_chunks(_fake_session_factory(), uuid.uuid4(), 99)
