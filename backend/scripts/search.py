"""CLI entry point for hybrid search (+ optional reranking) over
ingested/embedded documents.

Usage (from backend/, with the venv active):
    python -m scripts.search "your query here" [--limit 30] [--alpha 0.5]
    python -m scripts.search "your query here" --rerank [--top-k 10]

Stage 1 (plan.md section 18): hybrid search returns the raw Top-`limit`
candidates. Stage 2, with `--rerank`: BGE-Reranker-v2-M3 narrows those
down to a relevance-based Top-`top-k`-max. No query classification/
rewriting, context construction, or LLM answer generation yet — that's
later. Requires `requirements/embeddings.txt` installed to embed the
query (and, with `--rerank`, to score candidates).
"""

import argparse
import sys

from scripts._console import fix_windows_console_encoding

fix_windows_console_encoding()

from app.database.vector_store import CHUNK_COLLECTION_NAME, get_weaviate_client
from app.rag.embeddings import EmbeddingModelUnavailableError, get_embedder
from app.rag.reranking import get_reranker
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.rerank import rerank_chunks


def main(query: str, limit: int, alpha: float, do_rerank: bool, top_k: int) -> int:
    embed_query_fn = get_embedder().embed_query

    client = get_weaviate_client()
    try:
        if not client.collections.exists(CHUNK_COLLECTION_NAME):
            print(
                f"Collection '{CHUNK_COLLECTION_NAME}' doesn't exist yet — embed a document first.",
                file=sys.stderr,
            )
            return 1
        collection = client.collections.get(CHUNK_COLLECTION_NAME)
        try:
            candidates = hybrid_search(collection, query, embed_query_fn, limit=limit, alpha=alpha)
        except EmbeddingModelUnavailableError as exc:
            print(f"Embedding failed: {exc}", file=sys.stderr)
            return 1
    finally:
        client.close()

    if not candidates:
        print("No results.")
        return 0

    if not do_rerank:
        for rank, chunk in enumerate(candidates, start=1):
            print(f"[{rank}] score={chunk.score:.4f} {chunk.document_name} (pages {chunk.start_page}-{chunk.end_page})")
            print(f"    {_preview(chunk.content)}")
        return 0

    try:
        reranked = rerank_chunks(query, candidates, get_reranker().score, top_k=top_k)
    except EmbeddingModelUnavailableError as exc:
        print(f"Reranking failed: {exc}", file=sys.stderr)
        return 1

    for rank, item in enumerate(reranked, start=1):
        c = item.chunk
        print(
            f"[{rank}] rerank_score={item.rerank_score:.4f} (hybrid_score={c.score:.4f}) "
            f"{c.document_name} (pages {c.start_page}-{c.end_page})"
        )
        print(f"    {_preview(c.content)}")
    return 0


def _preview(text: str, max_len: int = 200) -> str:
    return text[:max_len] + ("..." if len(text) > max_len else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid search (+ optional reranking) over ingested document chunks.")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=30, help="Stage 1 Top-N candidates (plan.md default: 30).")
    parser.add_argument("--alpha", type=float, default=0.5, help="0.0=BM25 only, 1.0=vector only.")
    parser.add_argument("--rerank", action="store_true", help="Apply Stage 2 reranking (BGE-Reranker-v2-M3).")
    parser.add_argument("--top-k", type=int, default=10, help="Max results kept after reranking.")
    args = parser.parse_args()
    sys.exit(main(args.query, args.limit, args.alpha, args.rerank, args.top_k))
