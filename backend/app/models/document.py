"""Documents, their versions, and the chunks a version is split into.

**Reconstructed 2026-09-07** — see `base.py` for why these files were
missing from git.

The three tables in one sentence: a **Document** is one exact file's
content, a **DocumentVersion** is one ingest of that content (carrying
the source PDF bytes), and a **DocumentChunk** is one retrievable
passage of a version.

**`content_hash` is unique, and that is what makes a Document identity
rather than a filename.** Re-uploading the same bytes under a different
name finds the existing row instead of creating a duplicate; editing a
file's bytes produces a different hash and therefore a NEW Document, not
a new version of the old one. Versions exist for re-ingesting the SAME
bytes — after a chunk-size change, for instance.

Rebuilt against `migrations/versions/20260901_0001_initial_schema.py`.
Note which indexes exist: only `document_chunks.document_version_id` has
one. Adding others here without a matching migration would make
`alembic check` report drift.
"""

import uuid

from sqlalchemy import Boolean, ForeignKey, Integer, LargeBinary, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """One exact file content, identified by its hash."""

    __tablename__ = "documents"

    __table_args__ = (UniqueConstraint("content_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # NOT unique: two files with the same name but different content are
    # two documents. The hash is the identity, not this.
    name: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(Base, TimestampMixin):
    """One ingest of a document's content.

    `pdf_bytes` holds the source PDF in the database rather than on disk.
    That is what makes re-chunking possible after a `CHUNK_SIZE_TOKENS`
    change without needing the original upload, and it is why this
    project has no filesystem state to back up or lose.

    `is_active` marks the version retrieval should use. Exactly one
    version per document is expected to be active;
    `DocumentRepository.create_version` deactivates the previous one in
    the same transaction as it inserts the new one.
    """

    __tablename__ = "document_versions"

    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_document_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    pdf_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    page_count: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped["Document"] = relationship(back_populates="versions")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="version_row", cascade="all, delete-orphan"
    )


class DocumentChunk(Base, TimestampMixin):
    """One retrievable passage of a document version.

    The chunk TEXT lives here, not in Weaviate. Weaviate holds the
    vectors and a `chunk_id` pointing back at this row, so PostgreSQL
    stays the single source of truth for content and a Weaviate rebuild
    never loses text.

    `start_page`/`end_page` are what a citation is built from — a chunk
    can span a page boundary, so both are stored rather than one page
    number.
    """

    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_version_id", "chunk_index", name="uq_document_chunk_index"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Indexed: every read of this table filters on it, and this is the
    # one index migration 0001 creates on this table.
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    start_page: Mapped[int] = mapped_column(Integer)
    end_page: Mapped[int] = mapped_column(Integer)

    version_row: Mapped["DocumentVersion"] = relationship(back_populates="chunks")
