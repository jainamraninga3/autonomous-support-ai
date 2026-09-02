"""Tests for the log-tail endpoint's cursor semantics.

The parsing is easy; the cursor is where the bugs would be — a poller
that re-shows lines it already has, misses lines, or splits a line across
two responses all look like corrupted output in the UI.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes import logs as logs_route


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    """Point the route at a temporary log file instead of the real one."""
    path = tmp_path / "app.log"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(logs_route, "LOG_FILE", path)
    return path


async def _tail(**params):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/logs/tail", params=params)
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_default_offset_starts_at_the_end_and_returns_nothing(log_file) -> None:
    """A freshly-opened UI must show what happens NEXT, not replay up to
    5MB of history."""
    log_file.write_text("old line 1\nold line 2\n", encoding="utf-8")

    body = await _tail()

    assert body["lines"] == []
    assert body["next_offset"] == log_file.stat().st_size


@pytest.mark.asyncio
async def test_returns_only_what_was_appended_after_the_cursor(log_file) -> None:
    log_file.write_text("first\n", encoding="utf-8")
    cursor = (await _tail())["next_offset"]

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("second\n")

    body = await _tail(offset=cursor)

    assert [line["raw"] for line in body["lines"]] == ["second"]

    # Polling again with the new cursor yields nothing — no duplicates.
    assert (await _tail(offset=body["next_offset"]))["lines"] == []


@pytest.mark.asyncio
async def test_parses_the_app_log_format_into_fields(log_file) -> None:
    log_file.write_text(
        "2026-09-01T13:09:40+0000 level=INFO logger=app.main message=Starting up\n",
        encoding="utf-8",
    )

    body = await _tail(offset=0)
    line = body["lines"][0]

    assert line["level"] == "INFO"
    assert line["logger"] == "app.main"
    assert line["message"] == "Starting up"


@pytest.mark.asyncio
async def test_keeps_lines_that_do_not_match_the_app_format(log_file) -> None:
    """uvicorn, sqlalchemy and sentence-transformers log in their own
    formats. Dropping them would hide exactly the output that matters when
    something is wrong."""
    log_file.write_text("INFO:     Application startup complete.\n", encoding="utf-8")

    line = (await _tail(offset=0))["lines"][0]

    assert line["raw"] == "INFO:     Application startup complete."
    assert line["level"] is None


@pytest.mark.asyncio
async def test_offset_past_the_end_restarts_and_flags_truncation(log_file) -> None:
    """RotatingFileHandler rolls the file over at 5MB, so the cursor can
    end up past the end. Reading must restart rather than returning
    nothing forever, and the client must be told why the content jumped."""
    log_file.write_text("after rotation\n", encoding="utf-8")

    body = await _tail(offset=999_999)

    assert body["truncated"] is True
    assert [line["raw"] for line in body["lines"]] == ["after rotation"]


@pytest.mark.asyncio
async def test_missing_log_file_is_not_an_error(log_file) -> None:
    """The file doesn't exist until the app logs something. The UI should
    show an empty console, not an error."""
    log_file.unlink()

    body = await _tail(offset=0)

    assert body == {"lines": [], "next_offset": 0, "file_size": 0, "truncated": False}


@pytest.mark.asyncio
async def test_a_partial_final_line_is_held_back_until_it_is_complete(log_file) -> None:
    """A read capped by max_bytes can land mid-line. Returning the half
    would render as garbage and the other half would arrive next poll, so
    the cursor is rewound to the last newline instead."""
    log_file.write_text("aaaa\nbbbb\ncccc", encoding="utf-8")  # no trailing newline

    body = await _tail(offset=0, max_bytes=1024 + 0)
    # Whole file fits, so nothing is held back here...
    assert [line["raw"] for line in body["lines"]] == ["aaaa", "bbbb", "cccc"]

    # ...but when the cap forces a cut, the incomplete tail is excluded and
    # the cursor stops at the last newline.
    body = await _tail(offset=0, max_bytes=1024)
    assert body["next_offset"] <= log_file.stat().st_size
