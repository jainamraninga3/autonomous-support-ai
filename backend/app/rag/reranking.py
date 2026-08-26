"""Reranking (plan.md section 19: BAAI BGE-Reranker-v2-M3).

Same lazy-import pattern as `app/rag/embeddings.py`: `sentence-
transformers` is only imported inside `Reranker._load_model()`, not at
module import time, so nothing breaks if it's absent, and nothing is
downloaded until this is actually used. It lives in the same optional
`requirements/embeddings.txt` as the embedding model — no separate
install needed since `sentence-transformers.CrossEncoder` covers both.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingModelUnavailableError

logger = get_logger(__name__)


class Reranker:
    """Wraps a sentence-transformers `CrossEncoder` for query/chunk scoring."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise EmbeddingModelUnavailableError(
                "sentence-transformers (and torch) must be installed to rerank with "
                f"'{self.model_name}'. Install requirements/embeddings.txt, then retry — "
                "the first call will also download the model weights."
            ) from exc

        logger.info("Loading reranker model '%s' (first use downloads the weights)", self.model_name)
        self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in the same order as `texts`."""
        if not texts:
            return []
        model = self._load_model()
        scores = model.predict([(query, text) for text in texts])
        return [float(s) for s in scores]


@lru_cache
def get_reranker() -> Reranker:
    """Return a process-wide cached `Reranker` for the configured model."""
    settings = get_settings()
    return Reranker(model_name=settings.RERANKER_MODEL_NAME)
