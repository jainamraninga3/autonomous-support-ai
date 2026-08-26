"""Basic text cleaning for extracted PDF page text.

Deliberately simple for this pass: whitespace normalization only.
Header/footer detection (plan.md section 6) needs cross-page frequency
analysis to do well and is left for a follow-up rather than guessed at
here.
"""

import re

_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted PDF text.

    - Collapses runs of 3+ blank lines down to a single blank line.
    - Strips trailing whitespace at the end of each line.
    - Collapses runs of spaces/tabs into a single space.
    - Strips leading/trailing whitespace overall.
    """
    text = _TRAILING_SPACES.sub("\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()
