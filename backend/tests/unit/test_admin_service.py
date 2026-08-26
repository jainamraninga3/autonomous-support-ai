"""Unit tests for AdminService's reset operations — fake session/Weaviate
client, no real database or Weaviate connection needed.
"""

from dataclasses import dataclass

import pytest

from app.core.exceptions import ServiceUnavailableError
from app.services.admin_service import AdminService


@dataclass
class _FakeExecuteResult:
    rowcount: int = 0


class _FakeSession:
    def __init__(self) -> None:
        self.executed = []
        self.committed = False

    async def execute(self, statement):
        self.executed.append(statement)
        return _FakeExecuteResult(rowcount=2)

    async def commit(self):
        self.committed = True


class _FakeCollectionsHandle:
    def __init__(self, exists: bool) -> None:
        self._exists = exists
        self.deleted = False
        self.created = False

    def exists(self, name):
        return self._exists

    def delete(self, name):
        self.deleted = True
        self._exists = False

    def create(self, **kwargs):
        self.created = True
        self._exists = True


class _FakeWeaviateClient:
    def __init__(self, exists: bool = True) -> None:
        self.collections = _FakeCollectionsHandle(exists=exists)


@pytest.mark.asyncio
async def test_reset_postgres_deletes_documents_and_chat_sessions() -> None:
    session = _FakeSession()
    service = AdminService(session=session, weaviate_client=None)

    details = await service.reset_postgres()

    assert len(session.executed) == 2
    assert session.committed is True
    assert len(details) == 2


def test_reset_vector_store_deletes_existing_collection() -> None:
    client = _FakeWeaviateClient(exists=True)
    service = AdminService(session=None, weaviate_client=client)

    details = service.reset_vector_store()

    assert client.collections.deleted is True
    assert client.collections.created is True
    assert len(details) == 1


def test_reset_vector_store_creates_when_missing() -> None:
    client = _FakeWeaviateClient(exists=False)
    service = AdminService(session=None, weaviate_client=client)

    details = service.reset_vector_store()

    assert client.collections.deleted is False
    assert client.collections.created is True
    assert len(details) == 1


def test_reset_vector_store_raises_when_weaviate_unavailable() -> None:
    service = AdminService(session=None, weaviate_client=None)

    with pytest.raises(ServiceUnavailableError):
        service.reset_vector_store()
