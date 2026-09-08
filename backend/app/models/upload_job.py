"""Upload job tracking — schema only, nothing writes to it yet.

**Reconstructed 2026-09-07** — see `base.py` for why these files were
missing from git.

This table exists for a background/async ingestion path that has not
been built: today `POST /api/v1/documents` ingests synchronously and
returns when the work is done. It is kept because migration 0001 creates
the table, so removing the model would make `alembic check` report drift
against a table that really is there.

Two details that match 0001 and are easy to "fix" wrongly:
- `document_id` is NULLABLE and has **no foreign key**. A job is created
  before its document exists — that is the point of a job — and a job
  that failed may never get one at all. Adding an FK would make the
  failure case unrecordable.
- `status` is plain TEXT with no CHECK constraint, unlike
  `memories.status` elsewhere in this project's history. Nothing reads
  it yet, so there is no vocabulary to enforce; add the constraint in
  the same change that starts using it.
"""

import uuid

from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class UploadJob(Base, TimestampMixin):
    """One document upload, for the not-yet-built async ingestion path."""

    __tablename__ = "upload_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    status: Mapped[str] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
