"""Unit tests for the embedding wrapper's behavior without a real model.

`requirements/embeddings.txt` (`sentence-transformers`/`torch`) is
optional and may or may not be installed depending on the environment —
these tests don't assume either way. The dependency-missing test forces
the lazy import to fail via monkeypatching `builtins.__import__`, so it
stays meaningful (and doesn't flake) regardless of whether
sentence-transformers actually happens to be installed here.
"""

import builtins

import pytest

from app.rag.embeddings import Embedder, EmbeddingModelUnavailableError


def test_embed_texts_empty_list_short_circuits_without_loading_model() -> None:
    embedder = Embedder(model_name="BAAI/bge-m3")
    assert embedder.embed_texts([]) == []
    assert embedder._model is None


def test_embed_texts_raises_clear_error_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated: sentence-transformers not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    embedder = Embedder(model_name="BAAI/bge-m3")
    with pytest.raises(EmbeddingModelUnavailableError, match="sentence-transformers"):
        embedder.embed_texts(["some text"])
