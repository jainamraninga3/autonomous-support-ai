"""Shared CLI helper: make stdout/stderr safe for non-ASCII output on Windows.

Windows terminals often default to a legacy codepage (e.g. cp1252) for
Python's stdout/stderr, which raises `UnicodeEncodeError` and crashes
the script the moment any printed text contains a character outside
that codepage — which real LLM output frequently does (curly quotes,
em-dashes, CJK-style brackets, etc.). Reconfiguring to UTF-8 with
`errors="replace"` avoids the crash; anything truly unrenderable in the
terminal shows as `?` instead of killing the process.
"""

import sys


def fix_windows_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # not all stream types support reconfigure (e.g. when redirected) — best effort
