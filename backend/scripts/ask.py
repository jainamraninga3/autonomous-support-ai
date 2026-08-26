"""CLI entry point for the full retrieval -> grounded-answer flow.

Usage (from backend/, with the venv active):
    python -m scripts.ask "your question" [--limit 30] [--alpha 0.5]
    python -m scripts.ask "your question" --rerank [--top-k 10]

Pipeline: hybrid search (Stage 1) -> optional rerank (Stage 2) ->
context construction -> grounded answer from the configured LLM (Groq
if GROQ_API_KEY is set, else the echo stub — see
`app/api/dependencies.get_llm_client`).

NOT implemented (later phases, see plan.md): query classification
(this always treats the input as a RAG query), query rewriting (the
raw question is embedded/searched as-is), and answer verification
(the answer is returned as generated, unchecked).
"""

import argparse
import asyncio
import sys

from scripts._console import fix_windows_console_encoding

fix_windows_console_encoding()

from app.api.dependencies import get_llm_client
from app.database.vector_store import CHUNK_COLLECTION_NAME, get_weaviate_client
from app.rag.embeddings import EmbeddingModelUnavailableError, get_embedder
from app.rag.generation.answer_generator import generate_answer
from app.rag.reranking import get_reranker
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.rerank import rerank_chunks


async def _run(query: str, limit: int, alpha: float, do_rerank: bool, top_k: int) -> int:
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
            if do_rerank:
                reranked = rerank_chunks(query, candidates, get_reranker().score, top_k=top_k)
                chunks = [r.chunk for r in reranked]
            else:
                chunks = candidates[:top_k]
        except EmbeddingModelUnavailableError as exc:
            print(f"Embedding/reranking failed: {exc}", file=sys.stderr)
            return 1
    finally:
        client.close()

    result = await generate_answer(query, chunks, get_llm_client())

    print(result.answer)
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            page_range = f"page {c.start_page}" if c.start_page == c.end_page else f"pages {c.start_page}-{c.end_page}"
            print(f"  [{c.label}] {c.document_name} ({page_range})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve context and generate a grounded answer.")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=30, help="Stage 1 Top-N candidates.")
    parser.add_argument("--alpha", type=float, default=0.5, help="0.0=BM25 only, 1.0=vector only.")
    parser.add_argument("--rerank", action="store_true", help="Apply Stage 2 reranking before answering.")
    parser.add_argument("--top-k", type=int, default=10, help="Max chunks used as context.")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.query, args.limit, args.alpha, args.rerank, args.top_k)))
