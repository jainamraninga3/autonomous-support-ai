"""Unit tests for `is_already_embedded` — fake Weaviate client, no real
Weaviate connection needed.
"""

import uuid
from types import SimpleNamespace

from app.rag.ingestion.embedder import is_already_embedded

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
