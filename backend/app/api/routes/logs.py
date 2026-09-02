"""Read the application log over HTTP, so a UI can show what the backend
is actually doing.

Cursor-based polling rather than SSE/WebSocket on purpose: a byte offset
is stateless, survives a page reload, cannot leak a long-lived connection
per browser tab, and behaves identically through any dev server or proxy.
Polling once a second is cheap — each request is a seek and a read of
whatever was appended.

Reads `backend/logs/app.log`, written by `configure_logging()`'s
RotatingFileHandler (5MB x 3 backups). Only the live file is served; the
rotated `.1`/`.2`/`.3` backups are not, so a rotation is reported to the
client via `truncated` instead of silently skipping content.

Dev-only, like the admin router: unauthenticated, and log lines can
contain anything the app logged. Don't expose this publicly.
"""

import re

from fastapi import APIRouter, Query

from app.core.logging import LOG_FILE
from app.schemas.logs import LogLine, LogTailResponse

router = APIRouter(prefix="/logs", tags=["logs"])

# Matches the format set in `app/core/logging.py`:
#   2026-09-01T13:09:40+0000 level=INFO logger=app.main message=...
# Falls back to the raw line when it doesn't match — third-party libraries
# (uvicorn, sqlalchemy, sentence-transformers) log in their own formats and
# must still be shown, not dropped.
_LINE_PATTERN = re.compile(
    r"^(?P<timestamp>\S+)\s+level=(?P<level>\w+)\s+logger=(?P<logger>\S+)\s+message=(?P<message>.*)$",
    re.DOTALL,
)

# Cap per response so a client that has been away for a long time (or asks
# with offset=0 against a 5MB file) gets a bounded payload rather than
# megabytes in one go. It will simply catch up over the next few polls.
_MAX_BYTES_PER_READ = 256 * 1024


def _parse(raw: str) -> LogLine:
    match = _LINE_PATTERN.match(raw)
    if not match:
        return LogLine(raw=raw)
    return LogLine(
        raw=raw,
        timestamp=match.group("timestamp"),
        level=match.group("level"),
        logger=match.group("logger"),
        message=match.group("message"),
    )


@router.get("/tail", response_model=LogTailResponse)
async def tail_logs(
    offset: int = Query(
        default=-1,
        description=(
            "Byte offset to read from. Send -1 (the default) on the first call to "
            "start at the END of the file — i.e. only new activity, not the whole "
            "history. Then send back the `next_offset` from each response."
        ),
    ),
    max_bytes: int = Query(
        default=_MAX_BYTES_PER_READ,
        ge=1024,
        le=_MAX_BYTES_PER_READ,
        description="Maximum bytes to return in one response.",
    ),
) -> LogTailResponse:
    """Return log content written after `offset`."""
    if not LOG_FILE.exists():
        return LogTailResponse(lines=[], next_offset=0, file_size=0)

    file_size = LOG_FILE.stat().st_size

    # First call: start at the end, so a freshly-opened UI shows what
    # happens next instead of replaying up to 5MB of history.
    if offset < 0:
        return LogTailResponse(lines=[], next_offset=file_size, file_size=file_size)

    truncated = False
    if offset > file_size:
        # The file shrank — RotatingFileHandler rolled it over. Start again
        # from the beginning rather than reading past the end forever.
        offset = 0
        truncated = True

    read_size = min(max_bytes, file_size - offset)
    if read_size <= 0:
        return LogTailResponse(lines=[], next_offset=offset, file_size=file_size, truncated=truncated)

    with LOG_FILE.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(read_size)

    # A read can stop mid-line. Keep the incomplete tail out of this
    # response and rewind the cursor so it arrives whole next time —
    # otherwise a long line gets split across two responses and both
    # halves render as garbage.
    if read_size == max_bytes and not chunk.endswith(b"\n"):
        cut = chunk.rfind(b"\n")
        if cut != -1:
            chunk = chunk[: cut + 1]

    next_offset = offset + len(chunk)
    text = chunk.decode("utf-8", errors="replace")
    lines = [_parse(line) for line in text.splitlines() if line.strip()]

    return LogTailResponse(
        lines=lines,
        next_offset=next_offset,
        file_size=file_size,
        truncated=truncated,
    )
