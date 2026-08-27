# Autonomous Support AI — Backend

FastAPI + PostgreSQL + Weaviate + Groq, orchestrated by a real LangGraph
workflow implementing plan.md's full "Final V1 Pipeline": query
classification (general vs. RAG-required) → query rewriting → hybrid
search → BGE-Reranker-v2-M3 reranking → grounded answer generation with
citations → answer verification. `POST /api/v1/chat` runs this whole
graph; `POST /api/v1/rag/ask` is a lower-level endpoint that always runs
the RAG half directly (no classification), matching `scripts.ask`.
Documents are uploaded and ingested via the HTTP API (see "Documents API"
below) — upload a PDF to get a `document_id`, then use that id to embed
it into Weaviate whenever you're ready.

`requirements/embeddings.txt` (torch + sentence-transformers) is an
extra, opt-in install — not pulled in by default.

Scanned/image-based PDFs (no real text layer) are not supported —
extraction will come back with 0 chunks for those. Not part of this
project's scope; see `docs/PROJECT_LOG.md` if that changes later.

## Setup

```powershell
cd backend
.venv\Scripts\Activate.ps1
pip install -r requirements/dev.txt
copy .env.example .env   # then edit POSTGRES_PASSWORD, GROQ_API_KEY, etc.
```

## Multiple Groq API keys (automatic rate-limit fallback)

Set `GROQ_API_1`/`GROQ_API_2`/`GROQ_API_3` in `.env` alongside
`GROQ_API_KEY` (all optional — set only the ones you have). `GroqLLMClient`
(`app/llm/base.py`) tries `GROQ_API_KEY` first; if a call hits a rate
limit (`429` — e.g. a daily token quota exhausted), it automatically
retries the *same* request against the next configured key instead of
failing it, and remembers which key last worked so later calls start
there directly rather than re-trying dead keys every time. Only raises
(`429 rate_limited`, see Endpoints below) once every configured key is
rate-limited. A non-rate-limit Groq error (auth failure, 5xx, etc.)
surfaces immediately as `503 service_unavailable` — a different key
wouldn't fix those, so it doesn't burn through the rest of the list.

## Start PostgreSQL + Weaviate (Docker)

```powershell
docker compose up -d
```

Run from the repository root, where `docker-compose.yml` lives.

## Run the backend

```powershell
uvicorn app.main:app --reload
```

## Create database tables

No Alembic/migrations yet — tables are created directly from the ORM
models:

```powershell
python -m scripts.init_db
```

Re-run this after adding new models; it only creates missing tables, it
does not alter existing ones.

## Test

```powershell
pytest
```

## Ingest a PDF

```powershell
python -m scripts.ingest_document path\to\file.pdf
```

Extracts text, cleans it, and chunks it (fixed-size, ~100 tokens with
~20 token overlap — configurable via `CHUNK_SIZE_TOKENS` /
`CHUNK_OVERLAP_TOKENS`). Chunks are written as JSON under
`data/processed/<document_id>/v<version>.json`; the `documents` /
`document_versions` tables in PostgreSQL track identity and versioning.
Re-ingesting identical file content is a no-op (deduped by content
hash); pass `--force` to reprocess and create a new version.

## Embed a document into Weaviate

```powershell
pip install -r requirements/embeddings.txt   # one-time: torch + sentence-transformers
python -m scripts.embed_document <document_id>
```

Reads the JSON chunk output from the ingest step above, embeds each
chunk with BGE-M3 (`EMBEDDING_MODEL_NAME`, default `BAAI/bge-m3`), and
upserts it into Weaviate's `DocumentChunk` collection (created
automatically, `vectorizer: none` — vectors are supplied here, not
computed by Weaviate). Re-running is idempotent (same document/version/
chunk-index always maps to the same object, so it updates rather than
duplicates).

**Note:** `requirements/embeddings.txt` is intentionally NOT part of
`base.txt`/`dev.txt` — installing it pulls in `torch`, and the first
embed call downloads the ~2.3GB BGE-M3 model weights. Verified end to
end with a real model run: ingest → embed → 1024-dim vectors confirmed
in Weaviate.

**Windows note:** installing `torch` can fail with
`OSError: [WinError 206] The filename or extension is too long` —
`torch`'s wheel contains deeply nested license file paths that overflow
Windows' default `MAX_PATH` (260 chars), especially combined with an
already-long project path. Fix (one-time, admin PowerShell, no reboot —
just open a fresh non-admin terminal afterward):
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Search (hybrid, + optional reranking)

```powershell
python -m scripts.search "your query here" --limit 30 --alpha 0.5
python -m scripts.search "your query here" --rerank --top-k 5
```

Stage 1 (always): embeds the query with BGE-M3 and runs Weaviate's
hybrid (dense + BM25) search over `DocumentChunk`, printing the ranked
candidates with their scores. `--alpha` balances BM25 (`0.0`) against
vector search (`1.0`); `0.5` is a neutral default, not yet tuned
against an evaluation set.

Stage 2, with `--rerank`: reranks those candidates with
`BAAI/bge-reranker-v2-m3` down to a relevance-based Top-`--top-k`-max
(not forced to exactly that count). First use downloads the reranker
weights (~2.3GB, same `requirements/embeddings.txt` install as the
embedding model) — verified working end to end.

## Ask a question (full RAG pipeline)

```powershell
python -m scripts.ask "your question here"
python -m scripts.ask "your question here" --rerank --top-k 5
```

Runs hybrid search (+ optional reranking), builds a context block from
the retrieved chunks, and sends the original question + that context to
the configured LLM (Groq if `GROQ_API_KEY` is set, else the echo stub)
with a grounded-answer prompt: answer only from the given context, say
so if the answer isn't there, don't invent facts. Prints the answer
followed by a `Sources:` list built directly from retrieved metadata
(never from the LLM's own output — citations can't be hallucinated by
construction). If nothing is retrieved at all, skips the LLM call and
returns a fixed "not found" message.

No query classification (this always treats input as a RAG query), no
query rewriting (the raw question is embedded/searched as-is), and no
answer verification (the answer is returned as generated, unchecked) —
`scripts.ask` intentionally stays this simple; the full pipeline with
all three now lives in the LangGraph workflow instead (see below).

## Documents API

There is no auto-ingest-on-startup — documents are uploaded and
processed explicitly, via HTTP, in two steps:

1. `POST /api/v1/documents/upload` (multipart) — extracts, cleans, and
   chunks the PDF, writes it to PostgreSQL, and saves the file under
   `DOCUMENTS_DIR` (default `data/documents/`). Does **not** embed it.
   Returns a `document_id` (dedups by content hash — uploading identical
   bytes again returns the same id with `is_duplicate: true`).
2. `POST /api/v1/documents/{document_id}/ingest` — embeds that
   document's chunks into Weaviate (BGE-M3 + `requirements/embeddings.txt`
   required). Checks Weaviate first and skips re-embedding if this
   version is already there; pass `?force=true` to re-embed anyway
   (safe — upserts by a deterministic UUID, never duplicates).

Plus:

- `GET /api/v1/documents` — list every document, its versions, and
  whether each version is embedded (checked live against Weaviate).
- `GET /api/v1/documents/{document_id}` — same, for one document.
- `DELETE /api/v1/documents/{document_id}` — removes its Postgres rows,
  its stored chunk JSON, and any chunks it has in Weaviate.

## Admin: resetting data (development only)

- `POST /api/v1/admin/reset/postgres` — deletes all `documents` (and
  cascaded `document_versions`) and `chat_sessions` (and cascaded
  `messages`) rows. Does not touch Weaviate or files on disk.
- `POST /api/v1/admin/reset/vector-store` — deletes and recreates the
  Weaviate `DocumentChunk` collection, discarding every embedded chunk.
  Does not touch PostgreSQL.
- `POST /api/v1/admin/reset/all` — both of the above.

These are destructive and unauthenticated — this project has no auth
yet (see `docs/PROJECT_LOG.md`). Dev/test convenience only; don't expose
this router anywhere with real users.

## The full RAG workflow (LangGraph)

`app/graph/workflow.py`'s `build_graph()` compiles plan.md's complete
pipeline:

```
START -> classify -> GENERAL -> refuse (out of scope) -> END
                   -> RAG_REQUIRED -> rewrite -> retrieve (hybrid + rerank) -> generate -> [was_answerable?]
                                                                                   -> yes -> verify -> END
                                                                                   -> no  -> general_fallback -> END
```

- **Classify** (`app/rag/classification.py`) — an LLM call decides
  GENERAL vs. RAG_REQUIRED. Defaults to RAG_REQUIRED on any ambiguous
  reply (an unnecessary retrieval is cheaper than a skipped one). A
  GENERAL classification (off-topic — math, coding, general trivia,
  "what is 2+2", "what does Amazon do") is refused outright with a fixed
  message ("I'm a support assistant for our company's policies and
  documents...") — this bot only answers company-related questions.
  **The raw user message is never handed to the LLM to answer freely on
  this path** — a deliberate security boundary, not just a UX choice: it
  closes off the obvious way a user could otherwise turn this into a
  general-purpose assistant or attempt a prompt injection through an
  unconstrained "answer anything" call.
- **Rewrite** (`app/rag/query_rewriting.py`) — improves the query for
  retrieval while preserving its meaning; the *original* question (not
  the rewrite) is what's used to generate the final answer, per plan.md
  section 17. Falls back to the original query if the rewrite looks
  unreliable (empty, or suspiciously long).
- **Retrieve/generate** — the same hybrid search → rerank → grounded
  answer generation already used by `scripts.ask`/`/api/v1/rag/ask`. The
  grounded-answer prompt instructs the LLM to reply with an exact
  sentinel token when the retrieved context doesn't actually answer the
  question; `generate_answer()` detects that (or zero chunks retrieved
  at all) and reports `was_answerable=False`.
- **General fallback** (only reached when `was_answerable=False`) —
  answers from the LLM's own general knowledge instead of just refusing,
  but always prefixes the reply making that explicit ("This question
  isn't covered by our available documents, so here is a
  general-knowledge answer instead..."). `citations` stays empty (there
  is no source). This is narrower than it might sound: it only ever
  runs for a question `classify` already judged RAG_REQUIRED — it is
  not a way to route arbitrary trivia around retrieval, since
  `classify` already sent that elsewhere.
- **Verify** (`app/rag/generation/verification.py`, only reached when
  `was_answerable=True`) — an LLM call checks whether the generated
  answer is actually supported by the retrieved context. An unsupported
  verdict replaces the answer with a refusal message rather than
  returning it as-is — this path stays a hard refusal (not a
  general-knowledge fallback) since an answer that looked grounded but
  couldn't be verified is a hallucination risk, not a "not in our
  documents" case.

`POST /api/v1/chat` now runs this graph (see Endpoints below) instead of
calling the LLM directly.

## Endpoints

- `GET /health` — service, database, and vector store status
- `POST /api/v1/chat` — runs the full LangGraph workflow above (classify
  → refusal (off-topic), or rewrite → retrieve → generate →
  verify/general fallback), persisted to `chat_sessions` / `messages`
  in PostgreSQL. Body:
  `{"message": "...", "conversation_id": null}`. Returns
  `{"reply": "...", "conversation_id": "...", "citations": [...]}` —
  `citations` is empty for a general (non-RAG) reply. Uses Groq
  (`GROQ_API_KEY` + `GROQ_MODEL`) when a key is set; falls back to an
  echo stub (`StubLLMClient`) otherwise, so the app still runs without
  one (classification/rewriting/verification against the echo stub will
  behave oddly since it doesn't understand the prompts — fine for
  proving the flow wires together, not for real answers). See "Multiple
  Groq API keys" below for automatic rate-limit fallback.
- `POST /api/v1/rag/ask` — HTTP entry point onto the same pipeline as
  `scripts.ask`: hybrid search → optional rerank → grounded answer with
  citations, no classification/rewriting/verification. Body:
  `{"question": "...", "limit": 30, "alpha": 0.5, "rerank": true,
  "top_k": 10}` (all fields but `question` optional, defaults shown).
  Returns `{"answer": "...", "citations": [...], "was_answerable":
  true}`. Returns `404` if no documents have been ingested yet (the
  `DocumentChunk` collection doesn't exist), `503` if Weaviate isn't
  reachable or the embedding/reranking dependencies aren't installed.
  The Weaviate client is opened once at app startup (not reconnected per
  request) — see `app.main`'s `lifespan`.
- `POST /api/v1/documents/upload`, `POST /api/v1/documents/{id}/ingest`,
  `GET /api/v1/documents`, `GET /api/v1/documents/{id}`,
  `DELETE /api/v1/documents/{id}` — see "Documents API" above.
- `POST /api/v1/admin/reset/postgres`, `/vector-store`, `/all` — see
  "Admin: resetting data" above.

## Architecture

```
API (routes) -> Service -> Repository / Database
                        -> LangGraph workflow (app/graph) -> LLM abstraction -> LLM provider
                                                           -> RAG pipeline (app/rag)
```

Routes contain no business logic; all logic lives in `app/services`
(for `/api/v1/chat`, that means "run the graph and persist the result,"
not classification/retrieval logic itself — that lives in `app/graph`
and `app/rag`).
