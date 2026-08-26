"""
Config for the standalone RAG chat test harness.

This folder (rag_chat_test/) is completely separate from the main
backend app — it only talks to it over HTTP, exactly like any other
client of POST /api/v1/chat. Nothing here imports or modifies any file
under backend/app.
"""

import os
from pathlib import Path

# --- Target API (the actual chat endpoint being tested) ---
CHAT_API_URL = "http://localhost:8000/api/v1/chat"

# --- Retrieval/reranking comparison ---
# /api/v1/chat runs one fixed pipeline (classify -> rewrite -> retrieve ->
# rerank -> generate -> verify) with no way to change retrieval settings
# per request. /api/v1/rag/ask exposes the retrieval knobs directly
# (alpha = BM25-vs-vector weighting, rerank on/off, top_k), so for each
# question we additionally run it through this endpoint once per variant
# below — giving several differently-retrieved answers to the SAME
# question, each independently scored by the judge LLM, so you can see
# which reranking/retrieval strategy actually produces the best answer.
RAG_ASK_API_URL = "http://localhost:8000/api/v1/rag/ask"
ENABLE_RERANK_VARIANTS = True

RERANK_VARIANTS = [
    {"name": "no_rerank__hybrid_0.5", "rerank": False, "alpha": 0.5, "limit": 30, "top_k": 10},
    {"name": "rerank__hybrid_0.5", "rerank": True, "alpha": 0.5, "limit": 30, "top_k": 10},
    {"name": "rerank__vector_only_1.0", "rerank": True, "alpha": 1.0, "limit": 30, "top_k": 10},
    {"name": "rerank__bm25_only_0.0", "rerank": True, "alpha": 0.0, "limit": 30, "top_k": 10},
]

# Per-question timeout. Set to None to wait indefinitely for a response
# (no timeout at all — the runner will sit on one question until the API
# answers, however long that takes). If set to a number of seconds, the
# request is aborted after that long, the question is marked TIMEOUT, and
# the runner moves on to the next question.
REQUEST_TIMEOUT_SECONDS = None

# --- Evaluator LLM (Groq) — used to judge each answer's quality ---
# Prefer an environment variable if set, otherwise fall back to the key
# given directly for this test harness. Override by setting GROQ_API_KEY
# in your shell before running the script.
GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "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",
)
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_EVAL_TIMEOUT_SECONDS = 30

# --- Files ---
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "input" / "questions.json"
OUTPUT_DIR = BASE_DIR / "output"

# If True, every question reuses the conversation_id returned by the
# previous question (one continuous conversation). If False (default),
# every question starts a brand-new conversation — the right choice for
# independent quality testing of each question.
CHAIN_CONVERSATIONS = False
