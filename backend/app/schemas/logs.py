"""Schemas for the log-tail endpoint."""

from pydantic import BaseModel, Field


class LogLine(BaseModel):
    """One parsed line from the application log."""

    raw: str = Field(description="The complete log line, exactly as written.")
    timestamp: str | None = Field(default=None, description="ISO timestamp, if the line had one.")
    level: str | None = Field(
        default=None,
        description="DEBUG / INFO / WARNING / ERROR / CRITICAL, if the line had one.",
    )
    logger: str | None = Field(default=None, description="Logger name, e.g. 'app.graph.workflow'.")
    message: str | None = Field(default=None, description="The message body without the prefix fields.")


class LogTailResponse(BaseModel):
    """A slice of the log file, plus the cursor to continue from.

    Cursor-based rather than "last N lines" so a poller never re-shows
    lines it already has and never misses lines written between polls.
    """

    lines: list[LogLine]
    next_offset: int = Field(
        description=(
            "Byte offset to send as `offset` on the next request to get only "
            "what was written after this response."
        )
    )
    file_size: int = Field(description="Current size of the log file in bytes.")
    truncated: bool = Field(
        default=False,
        description=(
            "True when the requested offset was past the end of the file, which "
            "means the log rotated (RotatingFileHandler, 5MB x 3) and reading "
            "restarted from the beginning. A client should not treat the "
            "resulting jump in content as data loss on its side."
        ),
    )
