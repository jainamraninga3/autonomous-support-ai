"""Unit tests for DocumentService — fake repository/Weaviate client and
monkeypatched module-level ingestion/embedding functions. No real
database, Weaviate, or PDF parsing needed.
"""

import uuid
from types import SimpleNamespace

import pytest

from app.core.exceptions import BadRequestError, NotFoundError
from app.rag.ingestion.pipeline import IngestionError, IngestionResult
from app.services.document_service import DocumentService

DOCUMENT_ID = uuid.uuid4()


def _fake_result(**overrides) -> IngestionResult:
    defaults = dict(
        document_id=DOCUMENT_ID, version=1, is_duplicate=False, page_count=1, chunk_count=3
    )
    defaults.update(overrides)
    return IngestionResult(**defaults)


class _FakeRepository:
    def __init__(self, document=None, version=None) -> None:
        self.document = document
        self.version = version
        self.deleted = []

    async def get_by_id(self, document_id):
        return self.document

    async def get_version(self, document_id, version):
        return self.version

    async def get_active_version(self, document_id):
        return self.version

    async def get_by_id_with_versions(self, document_id):
        return self.document

    async def delete_document(self, document):
        self.deleted.append(document)


class _FakeCollectionsHandle:
    def __init__(self, exists: bool = True) -> None:
        self._exists = exists
        self.deleted_filters = []

    def exists(self, name):
        return self._exists

    def get(self, name):
        return SimpleNamespace(data=SimpleNamespace(delete_many=lambda where: self.deleted_filters.append(where)))


class _FakeWeaviateClient:
    def __init__(self, exists: bool = True) -> None:
        self.collections = _FakeCollectionsHandle(exists=exists)


def _make_service(tmp_path, repository=None, weaviate_client=None) -> DocumentService:
    service = DocumentService(session=object(), weaviate_client=weaviate_client, session_factory=lambda: None)
    if repository is not None:
        service.repository = repository
    return service


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.document_service.get_settings", lambda: SimpleNamespace(DOCUMENTS_DIR=str(tmp_path))
    )
    service = _make_service(tmp_path)

    with pytest.raises(BadRequestError):
        await service.upload(b"not empty", "notes.txt")


@pytest.mark.asyncio
async def test_upload_rejects_empty_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.document_service.get_settings", lambda: SimpleNamespace(DOCUMENTS_DIR=str(tmp_path))
    )
    service = _make_service(tmp_path)

    with pytest.raises(BadRequestError):
        await service.upload(b"", "policy.pdf")


@pytest.mark.asyncio
async def test_upload_passes_the_bytes_through_and_writes_nothing_to_disk(tmp_path, monkeypatch) -> None:
    """Uploads go to PostgreSQL only — the service must hand the raw bytes
    to the pipeline rather than saving a file first. A stray file on disk
    is exactly the state this design removed."""
    seen = {}

    async def fake_ingest_pdf(file_bytes, filename, session):
        seen["bytes"] = file_bytes
        seen["filename"] = filename
        return _fake_result()

    monkeypatch.setattr("app.services.document_service.ingest_pdf", fake_ingest_pdf)
    service = _make_service(tmp_path)

    result = await service.upload(b"%PDF-1.4 header", "policy.pdf")

    assert result.document_id == DOCUMENT_ID
    assert seen == {"bytes": b"%PDF-1.4 header", "filename": "policy.pdf"}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_upload_wraps_ingestion_error(tmp_path, monkeypatch) -> None:
    async def raising_ingest_pdf(file_bytes, filename, session):
        raise IngestionError("bad pdf")

    monkeypatch.setattr("app.services.document_service.ingest_pdf", raising_ingest_pdf)
    service = _make_service(tmp_path)

    with pytest.raises(BadRequestError):
        await service.upload(b"%PDF-1.4 header", "policy.pdf")


@pytest.mark.asyncio
async def test_ingest_raises_not_found_for_missing_document(tmp_path) -> None:
    repository = _FakeRepository(document=None)
    service = _make_service(tmp_path, repository=repository)

    with pytest.raises(NotFoundError):
        await service.ingest(DOCUMENT_ID, version=None, force=False)


@pytest.mark.asyncio
async def test_ingest_skips_when_already_embedded(tmp_path, monkeypatch) -> None:
    document = SimpleNamespace(id=DOCUMENT_ID, name="policy.pdf")
    version = SimpleNamespace(version=1, storage_path="x.json")
    repository = _FakeRepository(document=document, version=version)
    monkeypatch.setattr("app.services.document_service.is_already_embedded", lambda *a, **k: True)

    def _boom(*args, **kwargs):
        raise AssertionError("should not re-embed without force=True")

    monkeypatch.setattr("app.services.document_service.embed_document", _boom)
    service = _make_service(tmp_path, repository=repository)

    resolved_version, count, already_embedded = await service.ingest(DOCUMENT_ID, version=None, force=False)

    assert resolved_version == 1
    assert count == 0
    assert already_embedded is True


@pytest.mark.asyncio
async def test_ingest_force_reembeds_even_if_already_embedded(tmp_path, monkeypatch) -> None:
    document = SimpleNamespace(id=DOCUMENT_ID, name="policy.pdf")
    version = SimpleNamespace(version=1, storage_path="x.json")
    repository = _FakeRepository(document=document, version=version)
    monkeypatch.setattr("app.services.document_service.is_already_embedded", lambda *a, **k: True)

    async def fake_embed_document(session_factory, document_id, version, weaviate_client):
        return 5

    monkeypatch.setattr("app.services.document_service.embed_document", fake_embed_document)
    service = _make_service(tmp_path, repository=repository)

    resolved_version, count, already_embedded = await service.ingest(DOCUMENT_ID, version=None, force=True)

    assert count == 5
    assert already_embedded is False


@pytest.mark.asyncio
async def test_ingest_embeds_when_not_yet_embedded(tmp_path, monkeypatch) -> None:
    document = SimpleNamespace(id=DOCUMENT_ID, name="policy.pdf")
    version = SimpleNamespace(version=1, storage_path="x.json")
    repository = _FakeRepository(document=document, version=version)
    monkeypatch.setattr("app.services.document_service.is_already_embedded", lambda *a, **k: False)

    async def fake_embed_document(session_factory, document_id, version, weaviate_client):
        return 3

    monkeypatch.setattr("app.services.document_service.embed_document", fake_embed_document)
    service = _make_service(tmp_path, repository=repository)

    resolved_version, count, already_embedded = await service.ingest(DOCUMENT_ID, version=None, force=False)

    assert count == 3
    assert already_embedded is False


@pytest.mark.asyncio
async def test_delete_document_removes_weaviate_chunks_and_the_row(tmp_path) -> None:
    """Deleting a document has to clear BOTH stores: the Postgres row (its
    versions, chunk text, and stored PDF bytes cascade) and the embedded
    chunks in Weaviate. Leaving the Weaviate side behind would keep the
    deleted document answering questions."""
    document = SimpleNamespace(id=DOCUMENT_ID, versions=[SimpleNamespace(version=1)])
    repository = _FakeRepository(document=document)
    weaviate_client = _FakeWeaviateClient(exists=True)
    service = _make_service(tmp_path, repository=repository, weaviate_client=weaviate_client)

    await service.delete_document(DOCUMENT_ID)

    assert repository.deleted == [document]
    assert len(weaviate_client.collections.deleted_filters) == 1


@pytest.mark.asyncio
async def test_delete_document_raises_not_found_for_missing_document(tmp_path) -> None:
    repository = _FakeRepository(document=None)
    service = _make_service(tmp_path, repository=repository)

    with pytest.raises(NotFoundError):
        await service.delete_document(DOCUMENT_ID)


@pytest.mark.asyncio
async def test_ingest_prunes_earlier_versions_from_weaviate(tmp_path, monkeypatch) -> None:
    """Chunk objects are keyed by (document_id, version, chunk_index), so
    embedding a NEW version writes new objects instead of replacing the
    old ones. Nothing cleaned those up, and retrieval doesn't filter by
    version — so stale text kept competing with current text in every
    search and the collection grew on each re-process."""
    document = SimpleNamespace(id=DOCUMENT_ID, name="policy.pdf")
    version = SimpleNamespace(version=2)
    repository = _FakeRepository(document=document, version=version)
    monkeypatch.setattr("app.services.document_service.is_already_embedded", lambda *a, **k: False)

    async def fake_embed_document(session_factory, document_id, version, weaviate_client):
        return 4

    monkeypatch.setattr("app.services.document_service.embed_document", fake_embed_document)

    weaviate_client = _FakeWeaviateClient(exists=True)
    service = _make_service(tmp_path, repository=repository, weaviate_client=weaviate_client)

    await service.ingest(DOCUMENT_ID, version=None, force=False)

    # One delete_many issued, scoped to this document and excluding the
    # version we just embedded.
    assert len(weaviate_client.collections.deleted_filters) == 1


@pytest.mark.asyncio
async def test_a_failed_prune_does_not_fail_a_successful_embed(tmp_path, monkeypatch) -> None:
    """The embedding worked. Raising here would report the whole ingest as
    failed when the only casualty is leftover stale chunks — which is the
    state it was already in."""
    document = SimpleNamespace(id=DOCUMENT_ID, name="policy.pdf")
    repository = _FakeRepository(document=document, version=SimpleNamespace(version=1))
    monkeypatch.setattr("app.services.document_service.is_already_embedded", lambda *a, **k: False)

    async def fake_embed_document(*args, **kwargs):
        return 4

    monkeypatch.setattr("app.services.document_service.embed_document", fake_embed_document)

    class _ExplodingCollections:
        def exists(self, name):
            return True

        def get(self, name):
            raise RuntimeError("weaviate went away")

    weaviate_client = SimpleNamespace(collections=_ExplodingCollections())
    service = _make_service(tmp_path, repository=repository, weaviate_client=weaviate_client)

    _, count, already = await service.ingest(DOCUMENT_ID, version=None, force=False)

    assert count == 4
    assert already is False
