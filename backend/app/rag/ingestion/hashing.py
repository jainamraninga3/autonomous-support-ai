"""Content hashing, used for document deduplication (plan.md section 27)."""

import hashlib


def hash_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()
