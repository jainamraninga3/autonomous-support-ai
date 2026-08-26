"""Content hashing, used for document deduplication (plan.md section 27)."""

import hashlib
from pathlib import Path


def hash_file(path: Path) -> str:
    """Return the sha256 hex digest of a file's raw bytes."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
