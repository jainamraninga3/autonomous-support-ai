"""Dense embedding generation (plan.md section 11: BAAI BGE-M3).

`sentence-transformers` (and its `torch` dependency) is imported lazily,
inside `Embedder._load_model()`, not at module import time. This lets
the rest of the app import this module freely without requiring torch
to be installed, and without triggering the ~2.3GB BGE-M3 download until
something actually asks to embed text.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModelUnavailableError(Exception):
    """Raised when the embedding model's dependencies aren't installed."""


class Embedder:
    """Wraps a sentence-transformers model to produce dense embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelUnavailableError(
                "sentence-transformers (and torch) must be installed to generate "
                f"embeddings with '{self.model_name}'. Install them, then retry — "
                "the first call will also download the model weights."
            ) from exc

        logger.info("Loading embedding model '%s' (first use downloads the weights)", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text, in the same order."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single search query."""
        return self.embed_texts([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    """Return a process-wide cached `Embedder` for the configured model."""
    settings = get_settings()
    return Embedder(model_name=settings.EMBEDDING_MODEL_NAME)
