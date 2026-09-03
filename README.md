# Autonomous Support AI

A production-oriented RAG support chatbot: it answers questions about your
company's own documents, in whatever language you ask, with citations —
and refuses, or clearly discloses, when the answer isn't in those
documents.

FastAPI + PostgreSQL + Weaviate + Groq, orchestrated by a LangGraph
workflow. The whole stack runs with one command.

```powershell
copy backend\.env.example backend\.env   # set GROQ_API_KEY + POSTGRES_PASSWORD
docker compose up --build
```

Then open <http://localhost:8000/docs>.

---

## Table of contents

- [What it does](#what-it-does)
- [How a question flows through it](#how-a-question-flows-through-it)
- [Running it](#running-it)
- [Configuration](#configuration)
- [Using it: upload, ingest, ask](#using-it-upload-ingest-ask)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Data and persistence — read before you delete anything](#data-and-persistence--read-before-you-delete-anything)
- [Database migrations](#database-migrations)
- [Tuning: chunk size, reranking, retrieval](#tuning-chunk-size-reranking-retrieval)
- [Tests and the evaluation harness](#tests-and-the-evaluation-harness)
- [CLI scripts](#cli-scripts)
- [Troubleshooting](#troubleshooting)
- [Current status and what's not built](#current-status-and-whats-not-built)

---

## What it does

**Answers only from your documents, with citations.** Retrieved chunks
carry their own document name and page span, and citations are built from
that metadata — never from the LLM's output, so a citation cannot be
hallucinated by construction.

**Multilingual, from English-only documents.** Ask in Hindi, Gujarati,
Marathi, Kannada, Hinglish, or English and the answer comes back in that
same language. There is no translation step in the answer path — BGE-M3
embeds query and document into one shared vector space. A non-English
question additionally retrieves with an English translation of itself and
merges the hits, because cross-lingual retrieval otherwise surfaces a
measurably different set of chunks (Hindi shared only 6/10 with English
before this; 8/10 after).

**Three honest failure modes instead of one confident guess:**

| situation | what you get |
| --- | --- |
| off-topic question ("what is 2+2") | fixed refusal, and your message is never sent to the LLM at all |
| a bare greeting ("hi", "thanks") | short conversational reply, no retrieval |
| a company question, nothing relevant found | general-knowledge answer that says plainly it is not from your documents, with no citations |
| answer generated but not supported by its own context | the answer is discarded and you're told so, rather than shown it |

**Verified, not just implemented.** Every claim above has been exercised
against the live API. `docs/PROJECT_LOG.md` records what was verified,
when, and what is still unproven.

---

## How a question flows through it

```
POST /api/v1/chat
        │
   [persist user message]
        │
     classify ──── GENERAL ──────→ fixed refusal (no LLM call)  ─────→ END
        │                          a real security boundary, not just UX
        ├───────── SMALL_TALK ───→ short conversational reply    ─────→ END
        │
        └───────── RAG_REQUIRED
                     │
                  rewrite         retrieval-only; typos/grammar, same language
                   + translate    English copy for a 2nd retrieval pass (concurrent)
                     │
                  retrieve        BGE-M3 embed → Weaviate hybrid (dense + BM25)
                     │            2 passes for non-English, merged by rank
                     │            reranker available but OFF by default
                     │
                  generate        ORIGINAL question + chunks → grounded answer
                     │            answers in the question's language
                     │
                was it answerable?
                     ├── no ────→ general_fallback (disclosed, no citations) → END
                     └── yes ───→ verify ──── supported ────→ the answer     → END
                                         └── unsupported ──→ "couldn't verify"
        │
   [persist reply] → response
```

The response carries the pipeline's internals so you can see *why* you got
what you got: `answer_source`, `rewritten_query`, `english_query`,
`retrieved_chunk_count`, `verified`, `verification_reason`. That
observability is not decoration — it is what identified two real bugs that
guesswork had missed.

`/api/v1/chat` is the only RAG endpoint. A second one
(`POST /api/v1/rag/ask`) used to exist for retrieval tuning; its
`limit` / `alpha` / `rerank` / `top_k` knobs are now optional fields on
the chat request, so there is one code path instead of two that had
drifted apart. Omit them for the configured defaults.

---

## Running it

### Requirements

- **Docker Desktop** — that's it for the default path.
- For running the backend outside Docker: **Python 3.11** and a
  virtualenv.
- A **Groq API key** (free tier works; note the 200,000 tokens/day
  limit). Up to four keys can be configured and the client falls through
  to the next one on a rate limit.
- **~8GB free disk.** The image is large (torch), and the models are
  ~2.3GB each.

### Option A — everything in Docker (recommended)

```powershell
copy backend\.env.example backend\.env
# edit backend\.env: set GROQ_API_KEY and POSTGRES_PASSWORD
docker compose up --build
```

Starts PostgreSQL, Weaviate, and the backend. The backend applies its own
database migrations on startup (`backend/entrypoint.sh` runs
`alembic upgrade head` before uvicorn), so there is no separate schema
step and no virtualenv.

- Swagger: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

**The first run is slow twice over.** The build downloads CPU-only torch
(~1GB), and then your first question or ingest downloads BAAI/bge-m3
(~2.3GB) at runtime. Both are cached afterwards — torch in the image
layer, model weights in the `model_cache` volume. If the first
`/api/v1/chat` call seems to hang, it is downloading the model; watch
`docker compose logs -f backend`.

```powershell
docker compose logs -f backend               # follow app logs
docker compose exec backend pytest           # run tests in the container
docker compose exec backend alembic current  # which migration is applied
docker compose restart backend               # after editing backend\.env
docker compose down                          # stop, KEEP data
docker compose down -v                       # stop, DELETE all data
```

To edit code without rebuilding, uncomment the bind mount and the
`--reload` command in the `backend` service in `docker-compose.yml`.

### Option B — backend locally, databases in Docker

For a debugger, or to use the CLI scripts.

```powershell
docker compose up -d postgres weaviate    # from the repo root
cd backend
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env                    # then edit it
alembic upgrade head
uvicorn app.main:app --reload
```

`.env` is written for this case (`POSTGRES_HOST=localhost`,
`POSTGRES_PORT=5433`); `docker-compose.yml` overrides both for the
containerised backend, so one file serves both. Don't "fix" those values
for Docker — you'll break the local path.

Stop the containerised backend first (`docker compose stop backend`) or
port 8000 will already be taken.

---

## Configuration

All settings come from `backend/.env` (see `.env.example`). Nothing is
hardcoded.

| variable | default | notes |
| --- | --- | --- |
| `POSTGRES_HOST` / `PORT` | `localhost` / `5433` | overridden to `postgres` / `5432` inside Docker |
| `POSTGRES_USER` / `PASSWORD` / `DB` | `postgres` / — / `rag_chatbot` | password required |
| `WEAVIATE_HOST` / `PORT` / `GRPC_PORT` | `localhost` / `8080` / `50051` | overridden to `weaviate` inside Docker |
| `GROQ_API_KEY` | — | tried first |
| `GROQ_API_1` / `_2` / `_3` | — | optional extra keys, tried in order on a 429 |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | |
| `CHUNK_SIZE_TOKENS` | `100` | see [Tuning](#tuning-chunk-size-reranking-retrieval) before changing |
| `CHUNK_OVERLAP_TOKENS` | `10` | must be less than chunk size |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | multilingual; 1024-dim |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | |
| `RERANK_ENABLED` | `false` | measured as not worth its latency on this corpus; also the default for the chat request's `rerank` field |
| `RERANK_TOP_K` | `10` | chunks passed to generation |
| `LOG_LEVEL` | `INFO` | |
| `DEBUG` | `true` | `true` echoes all SQL |

**Multiple Groq keys:** `GroqLLMClient` tries `GROQ_API_KEY`, and on a
rate limit (429 — e.g. the daily token quota) retries the *same* request
against the next configured key, remembering which key last worked so
later calls start there. It only fails once every key is rate-limited. A
non-rate-limit error (bad auth, 5xx) surfaces immediately as a 503,
because another key wouldn't fix it.

Without any Groq key the app still runs, falling back to an echo stub —
enough to prove the wiring, useless for real answers.

---

## Using it: upload, ingest, ask

Two explicit steps, because chunking is cheap and embedding is not.

**1. Upload** — `POST /api/v1/documents/upload` (multipart). Extracts,
cleans, and chunks the PDF, and writes the chunk text *and* the PDF's own
bytes to PostgreSQL. Returns a `document_id`. Deduplicates by content
hash. Does **not** embed.

**2. Ingest** — `POST /api/v1/documents/{document_id}/ingest`. Embeds that
document's chunks into Weaviate with BGE-M3. Skips work if this version is
already embedded; `?force=true` re-embeds anyway (safe — upserts by a
deterministic UUID, never duplicates).

**3. Ask** — `POST /api/v1/chat` with `{"message": "..."}`.

```powershell
$body = [Text.Encoding]::UTF8.GetBytes('{"message":"How many sick leaves do I get?"}')
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/chat" -Method Post `
  -ContentType "application/json; charset=utf-8" -Body $body | ConvertTo-Json -Depth 5
```

The UTF-8 byte encoding matters for non-English questions — PowerShell
otherwise mangles Devanagari/Gujarati/Kannada in the request body and
you'll be debugging the wrong thing.

**Scanned PDFs do not work.** Extraction is text-layer only (pypdf), no
OCR, so an image-based PDF ingests as zero chunks — in any language.

---

## API reference

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/health` | app + PostgreSQL + Weaviate connectivity |
| `POST` | `/api/v1/chat` | the full workflow; persists the conversation. Optional `limit`/`alpha`/`rerank`/`top_k` override retrieval per request |
| `POST` | `/api/v1/documents/upload` | extract + chunk + store (no embedding) |
| `POST` | `/api/v1/documents/{id}/ingest` | embed into Weaviate (`?force=true` to re-embed) |
| `GET` | `/api/v1/documents` | list documents, versions, and whether each is embedded |
| `GET` | `/api/v1/documents/{id}` | as above, one document |
| `DELETE` | `/api/v1/documents/{id}` | remove its Postgres rows **and** its Weaviate chunks |
| `POST` | `/api/v1/admin/reset/postgres` | delete all documents + chat history |
| `POST` | `/api/v1/admin/reset/vector-store` | drop and recreate the Weaviate collection |
| `POST` | `/api/v1/admin/reset/all` | both |

The admin reset endpoints are destructive and **unauthenticated** — this
project has no auth yet. Dev convenience only; never expose that router
where real users can reach it.

Full request/response detail, including every field on the chat response,
is in [backend/README.md](backend/README.md) and in Swagger.

---

## Project structure

```
autonomous-support-ai/
├── docker-compose.yml          postgres + weaviate + backend, one command
├── .gitattributes              forces LF on *.sh (a CRLF entrypoint breaks the container)
├── README.md                   this file
├── docs/
│   └── PROJECT_LOG.md          authoritative project state + full change history
├── frontend/                   not implemented yet
└── backend/
    ├── Dockerfile
    ├── entrypoint.sh           alembic upgrade head, then exec uvicorn
    ├── requirements.txt        ALL dependencies, single file
    ├── alembic.ini
    ├── .env.example
    ├── pytest.ini
    ├── migrations/             Alembic; env.py takes the DB URL from app config
    │   └── versions/
    ├── scripts/                CLI entry points (ask, search, ingest, embed)
    ├── tests/unit/             96 tests, no DB/network/model needed
    ├── rag_chat_test/          evaluation harness (talks to the live API over HTTP)
    │   ├── input/questions.json
    │   ├── main_script/        test_runner.py, config.py
    │   └── output/             per-run JSON reports
    └── app/
        ├── main.py             FastAPI app + lifespan (opens Weaviate once)
        ├── api/
        │   ├── dependencies.py DI wiring
        │   └── routes/         chat, rag, documents, admin, health
        ├── core/               config (Settings), logging, exceptions
        ├── database/           async engine/session, Weaviate client + collection
        ├── models/             SQLAlchemy: chat, document, document_chunk, upload_job
        ├── repositories/       data access only, no business logic
        ├── schemas/            Pydantic request/response models
        ├── services/           business logic: chat, rag, document, admin
        ├── llm/base.py         LLMClient abstraction + Groq multi-key client
        ├── graph/
        │   ├── state.py        GraphState (data only, never service handles)
        │   └── workflow.py     the LangGraph pipeline
        └── rag/
            ├── classification.py    GENERAL / SMALL_TALK / RAG_REQUIRED
            ├── query_rewriting.py   retrieval-only rewrite, never translates
            ├── query_translation.py English copy for the 2nd retrieval pass
            ├── embeddings.py        BGE-M3 loader
            ├── reranking.py         BGE-Reranker-v2-M3 loader
            ├── ingestion/           pdf_extractor, text_cleaner, chunker,
            │                        hashing, pipeline, embedder, vector_writer
            ├── retrieval/           hybrid_search, rerank, merge
            └── generation/          answer_generator, context_builder,
                                     small_talk, verification
```

`app/agents/`, `app/crew/`, `app/dspy/`, `app/mcp/`,
`app/observability/`, and `tests/integration/`, `tests/evaluation/` are
empty scaffolding for later phases.

**Layering:** routes contain no business logic.

```
routes → services → repositories → PostgreSQL
                  → LangGraph workflow (app/graph) → LLM abstraction → Groq
                                                   → RAG pipeline (app/rag) → Weaviate
```

---

## Data and persistence — read before you delete anything

**Nothing is stored on the filesystem.** An uploaded PDF ends up here:

```
your PDF's bytes
  → document_versions.pdf_bytes   (BYTEA column)
  → Postgres database "rag_chatbot"
  → /var/lib/postgresql/data      inside the postgres:16-alpine container
  → Docker named volume "postgres_data"
```

Chunk text sits alongside it in `document_chunks`; embeddings live in
Weaviate's `DocumentChunk` collection (volume `weaviate_data`); model
weights in `model_cache`. On Windows all three volumes are inside Docker
Desktop's WSL2 VM, not in the project tree.

| action | your PDFs, chunks, embeddings |
| --- | --- |
| `docker compose restart` / `stop` / `start` | survive |
| `docker compose down` | survive (named volumes aren't removed) |
| `docker compose down -v` | **destroyed** — `-v` deletes named volumes |
| `alembic downgrade base` | **destroyed** (drops the tables) |
| `POST /api/v1/admin/reset/*` | **destroyed** |

None of those warn you that source documents are going with them, and
there is no longer a copy of the PDF in the repo to fall back on. **Keep
your source files somewhere outside this project.**

The PDF bytes are stored specifically so a document can be re-chunked
after a `CHUNK_SIZE_TOKENS` change without needing the original file.
There is no `/rechunk` endpoint yet, so today that still means
delete + re-upload.

---

## Database migrations

Alembic is the single source of schema truth. The containerised backend
runs `alembic upgrade head` on every startup, so normally there's nothing
to do. From `backend/`:

```powershell
alembic upgrade head                              # apply pending migrations
alembic revision --autogenerate -m "what changed" # generate one from model changes
alembic current                                   # which revision is applied
alembic check                                     # do models match the database?
alembic downgrade -1                              # undo the last migration
```

`migrations/env.py` takes the database URL from `app.core.config`, not
from `alembic.ini`, so migrations can never run against a different
database than the app.

**Existing database from before Alembic?** If its schema already matches
the models, adopt it instead of migrating it:

```powershell
alembic stamp head
```

`upgrade head` would try to create tables that already exist and fail.

`scripts/init_db.py` is a deprecation notice now. It used
`Base.metadata.create_all`, which only creates *missing* tables — when a
model's columns changed it silently skipped that table and still reported
success, leaving the schema stale until an insert failed with
`column ... does not exist`. That happened during the storage rework and
is the reason Alembic is here.

---

## Tuning: chunk size, reranking, retrieval

### Chunk size — the setting most likely to break your answers

`CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` are **100/10**. Know what
that trades away, because it has already produced a wrong answer in this
project.

At ~100 tokens a policy sentence stating two rules can split across two
chunks, and a small overlap means neither neighbour holds it whole. Real
case from the leave policy: *"Accumulated earned leave may be carried
forward, each year, EL beyond 30 days is encashed by the Company in
January"* was severed, so a query that retrieved only the tail saw the
encashment rule and truthfully reported that carry-forward "is not stated
in the policy." Generation and verification were both behaving correctly
— the context was broken.

| size / overlap | chunks (2,000-token PDF) | chunks holding that sentence whole |
| --- | --- | --- |
| 600 / 100 | 4 | 2 |
| 100 / 20 | 25 | 1 |
| **100 / 10 (current)** | **23** | **1** |

**If answers start claiming your documents omit something they actually
state, raise chunk size or overlap first.** Parent-child chunking is the
proper fix — retrieve on small chunks, feed the LLM the larger parent —
and isn't built yet.

Chunking happens at **upload** time, so changing these does not affect
already-uploaded documents. Delete and re-upload to re-chunk.

### Reranking — off by default, deliberately

| variant | avg score | avg time | questions won |
| --- | --- | --- | --- |
| hybrid, no rerank | 8.00 | 16.1s | 5 |
| hybrid + rerank | 7.69 | 48.9s | 0 |
| vector-only + rerank | 8.38 | 58.8s | 4 |
| BM25-only + rerank | 7.75 | 57.6s | 0 |

With `RERANK_ENABLED=false`, retrieval takes the head of Stage 1's
already-ranked results up to `RERANK_TOP_K` — the same selection, minus
the cross-encoder pass. The model and code are untouched; set it to
`true` to restore it. A larger or more heterogeneous corpus could easily
flip this conclusion — re-measure before trusting it there.

The chat request has a per-request `rerank` field. Omit it (or send
`null`) to follow `RERANK_ENABLED`; send `true`/`false` to override for
one call. The first `true` downloads the ~2.3GB reranker model, which
looks like a hang for several minutes.

### `alpha`

`alpha` balances BM25 (`0.0`) against dense vector search (`1.0`); `0.5`
is a neutral default, not tuned. Adjustable per request on
`/api/v1/chat`. Note that a Devanagari or Kannada query has no lexical
overlap with an English document, so BM25 contributes little for those —
which is what the dual-language retrieval pass compensates for.

---

## Tests and the evaluation harness

**Unit tests** — 97, no database, network, or model needed. Weaviate,
Postgres, the LLM, and both ML models are faked.

```powershell
docker compose exec backend pytest    # in the container
pytest                                # or locally, venv active, from backend/
```

**Evaluation harness** (`backend/rag_chat_test/`) — a separate thing
entirely: it drives the **live** API over HTTP (it imports nothing from
`app/`), scores answers with an LLM judge, measures precision@k/recall@k,
checks that the scope rules are actually enforced, and A/B tests four
retrieval variants per question.

```powershell
cd backend\rag_chat_test\main_script
python test_runner.py
```

Questions live in `input/questions.json`; reports are written to
`output/`. This makes real Groq API calls against your keys and can take
minutes. Several findings in this project came from it rather than from
reading code — including the chunking bug above.

---

## CLI scripts

Run from `backend/` with the venv active. These need Option B (local)
setup, not Docker.

```powershell
python -m scripts.ingest_folder ..\policy            # upload AND ingest a whole folder
python -m scripts.ingest_folder --verify-only        # what's ingested, in both stores
python -m scripts.ingest_document path\to\file.pdf [--force]
python -m scripts.embed_document <document_id> [--version N]
python -m scripts.search "your query" [--limit 30] [--alpha 0.5] [--rerank [--top-k 10]]
python -m scripts.ask "your question" [--rerank] [--top-k 5]
```

`scripts.ask` is intentionally the simple version: no classification, no
rewriting, no verification. The full pipeline is the LangGraph workflow
behind `/api/v1/chat`.

---

## Troubleshooting

**First call hangs for minutes.** It's downloading a model (~2.3GB) —
`docker compose logs -f backend` will show which. Only the first time;
after that it's in the `model_cache` volume. If the model being
downloaded is the *reranker*, you almost certainly sent
`"rerank": true` — Swagger prefills it. Remove that field.

**`exec /app/entrypoint.sh: no such file or directory`.** Git converted
the script to CRLF. `.gitattributes` prevents it; if you hit it anyway,
re-save `entrypoint.sh` with LF endings.

**Port 8000 already in use.** A local uvicorn and the containerised
backend both want it. `docker compose stop backend`, or stop the local
one.

**`/health` reports `vector_store: unhealthy` in Docker.** The backend
resolves Weaviate by service name; check that
`docker-compose.yml`'s `WEAVIATE_HOST: weaviate` override is intact and
that the weaviate container is up.

**Non-English question comes back in English, or with mangled
characters.** Almost always the request body encoding, not the pipeline —
use the UTF-8 byte form shown in [Using it](#using-it-upload-ingest-ask).

**Answers say the policy doesn't mention something it does.** Chunking.
See [Tuning](#tuning-chunk-size-reranking-retrieval).

**`429 rate_limited`.** Every configured Groq key is rate-limited. The
free tier is 200,000 tokens/day; add more keys as `GROQ_API_1/_2/_3`.

**`torch` install fails on Windows with `[WinError 206] The filename or
extension is too long`.** Torch's wheel has deeply nested license paths
that overflow `MAX_PATH`. One-time fix in admin PowerShell (then open a
fresh terminal):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Not an issue inside the Linux container.

---

## Current status and what's not built

**V1 RAG pipeline complete and live-verified end to end** — ingestion,
embedding, hybrid search, reranking, grounded generation with citations,
classification, rewriting, verification, the disclosed general-knowledge
fallback, the off-topic refusal boundary, small talk, and multilingual
Q&A across five languages plus Hinglish.

**Not implemented yet:**

- OCR — scanned PDFs yield zero chunks
- table extraction, header/footer detection, contextual chunking,
  parent-child retrieval
- metadata filtering (ACL / tenant), tenant isolation, auth/authz
- conversation-history-aware retrieval — history *is* persisted, but the
  graph only ever sees the current message, so every turn is stateless
  and a follow-up like "and what about earned leave?" has no antecedent
- background ingestion job tracking (the `upload_jobs` table is schema
  only)
- a regenerate loop when verification fails (today the answer is simply
  discarded)
- a `/rechunk` endpoint
- rate limiting, tracing/metrics
- MCP, CrewAI, DSPy, frontend, Redis

**Known rough edges:** no timeout on the Groq call, so a network hang
blocks indefinitely; pypdf turns curly apostrophes and bullets into
U+FFFD in stored chunk text (cosmetic for retrieval, but it reaches the
LLM's context); Weaviate's `fusion_type` is left at the server default.

[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) is the authoritative record of
project state and every change, including what has and hasn't been
verified. Read it before making changes.
