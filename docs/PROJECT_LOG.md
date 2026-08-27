# Project Log

This file is the shared memory for this project across sessions, models,
and accounts. Anyone (any Claude session, any account) picking up this
project should read this file FIRST, before reading code, to understand
what exists, what doesn't, and why things are the way they are.

The full long-term spec lives at `../../plan.md` (RAG chatbot R&D spec —
do not implement it all at once; it's the destination, not the current
task). Path corrected 2026-08-25: this file itself was moved from the
repo root to `docs/PROJECT_LOG.md` at some point outside any logged
session (found relocated, alongside other repo cleanup — git initialized,
`Makefile` and several stray `__init__.py` files removed — none of that
reorg is this session's work, noting it here only so the path fix makes
sense).

---

## Rules for any AI assistant working on this project

1. **Read this whole file before making any change.** It tells you the
   current state so you don't rebuild, duplicate, or contradict prior work.
2. **After every change** — every file added, removed, renamed, or
   meaningfully edited; every dependency added/removed; every config or
   docker change — add a new entry at the TOP of the "Change Log" section
   below (reverse chronological, newest first). Do this immediately, in
   the same turn as the change, not "later."
3. **Update the "Current State" section in place** (don't append to it —
   overwrite the relevant part) whenever a change makes it stale. That
   section must always reflect reality right now, not history.
4. **Never delete past Change Log entries.** They're the audit trail.
   If something is later reverted, add a new entry saying so — don't
   erase the old one.
5. **Always write the "why"**, not just "what" — a one-line reason so a
   future session (or the user) can judge whether the change still makes
   sense.
6. Keep entries short: file paths, what changed, why. Not essays.

---

## Current State (authoritative — always keep this section accurate)

**Phase:** Foundation / initial setup. No RAG pipeline implemented yet.

**Stack in use right now:**
- Python 3.11, FastAPI, SQLAlchemy 2.x (async) + asyncpg
- PostgreSQL — runs via Docker (`docker-compose.yml`, service `postgres`)
- Weaviate — runs via Docker (`docker-compose.yml`, service `weaviate`),
  connection helper only, no collections/schema yet
- LangGraph — dependency added, minimal single-node compiled graph exists
  in `backend/app/graph/`, NOT yet wired into the chat endpoint
- LLM: abstraction exists (`backend/app/llm/base.py`). `GroqLLMClient`
  (via the `groq` SDK's `AsyncGroq`) is now wired in and used whenever
  `GROQ_API_KEY` is set; `StubLLMClient` (echoes input) remains as the
  automatic fallback when no key is set (local dev/tests without Groq
  access). **Verified live 2026-08-25** — user added their own
  `GROQ_API_KEY`; a real chat request returned a genuine Groq reply
  (not the stub echo) and persisted correctly to Postgres (see Change
  Log entry below).

**Working endpoints:**
- `GET /health` — reports app status + PostgreSQL + Weaviate connectivity
- `POST /api/v1/chat` — no longer calls the LLM directly; now runs the
  full LangGraph workflow (see below): classify -> general LLM reply, or
  rewrite -> retrieve -> generate -> verify. Still persists both the user
  message and the final reply to PostgreSQL (`chat_sessions` /
  `messages`) exactly as before; `conversation_id` semantics unchanged
  (session UUID, reuse to continue a session). Response gained a
  `citations` field (empty for a general reply). **Verified live
  2026-08-26** — both the GENERAL branch (2 Groq calls, no retrieval)
  and the RAG branch (4 Groq calls: classify/rewrite/generate/verify,
  correct grounded answer + citation, verification judged SUPPORTED)
  confirmed against real Groq/BGE-M3/Weaviate via the actual console log,
  not just response shape; persistence confirmed unaffected (see Change
  Log entry below).
- `POST /api/v1/rag/ask` — unchanged by the above; still always-RAG, no
  classification, matching `scripts.ask`. See its own entry further
  down.
- `POST /api/v1/documents/upload`, `POST /api/v1/documents/{id}/ingest`,
  `GET /api/v1/documents`, `GET /api/v1/documents/{id}` — replaced
  auto-ingest-on-startup, see below. **Verified live 2026-08-26**
  (upload/ingest/force-reingest/list/get all confirmed, followed by a
  real grounded `/api/v1/chat` answer proving retrieval works off the
  new path). `DELETE /api/v1/documents/{id}` wired + unit-tested but not
  yet live-verified (destructive, deferred).
- `POST /api/v1/admin/reset/postgres`, `/vector-store`, `/all` —
  destructive dev-only data reset. **Not yet live-verified.**

**Database schema:** SQLAlchemy 2.x models exist for `chat_sessions`,
`messages` (used by the chat endpoint), and `documents`,
`document_versions` (now actively used by the ingestion pipeline — see
below), and `upload_jobs` (still schema only, for the background/async
ingestion phase, not yet built). No Alembic yet — tables are created
directly via `python -m scripts.init_db` (`backend/scripts/init_db.py`),
which must be re-run after adding new models.

**Document ingestion pipeline (`backend/app/rag/ingestion/`):**
standalone module, NOT wired to Weaviate/embeddings yet. Flow: validate
file (must be `.pdf`, non-empty) → hash (sha256) → dedup check against
`documents.content_hash` → `pypdf` text extraction per page →
whitespace-only cleaning → fixed-size token chunking via `tiktoken`
(`cl100k_base` encoding as a model-agnostic stand-in for BGE-M3's own
tokenizer) at `CHUNK_SIZE_TOKENS`/`CHUNK_OVERLAP_TOKENS` (default
600/100, `app/core/config.py`) → chunks written as JSON to
`PROCESSED_DATA_DIR/<document_id>/v<version>.json` (default
`data/processed/`) → `documents`/`document_versions` rows created/
updated via `DocumentRepository`. Run via CLI:
`python -m scripts.ingest_document path/to/file.pdf` (`--force` to
reprocess and bump the version). **Schema interpretation** (not a schema
change): `Document.content_hash` is unique, so a `Document` row = one
exact file content; a `DocumentVersion` = one ingestion/processing run
over that content (`is_active` = the run currently used downstream).
Editing a source file's actual bytes produces a new `content_hash` and
therefore a new `Document`, not a new version of the old one — this is
a deliberate reading of the existing schema, not something enforced by
a DB constraint, so a different interpretation could be adopted later if
it turns out to be wrong. No table extraction, header/footer detection,
contextual chunking, or parent-child structure yet (plan.md sections 6,
9, 10) — this is the baseline fixed-size chunking slice only.

**Embedding + Weaviate write (`backend/app/rag/embeddings.py`,
`backend/app/rag/ingestion/vector_writer.py`, `backend/app/database/
vector_store.py`):** wired and **verified live end to end, including a
real BGE-M3 model run** (2026-08-25 — see Change Log). CLI:
`python -m scripts.embed_document <document_id> [--version N]`, reads
the JSON chunks written by `scripts.ingest_document`, embeds them with
BGE-M3, and upserts them (with 1024-dim vectors, confirmed via a live
query) into the `DocumentChunk` Weaviate collection (`vectorizer: none`,
properties per plan.md section 4.1; `section`/`parent_id`/`tenant_id`/
`document_type`/`access_level` exist in the schema for forward
compatibility but aren't populated by anything yet). `sentence-
transformers`/`torch` (`backend/requirements/embeddings.txt`) are now
installed in this environment — install required enabling Windows
`LongPathsEnabled` first (torch's wheel has deeply nested license
paths that overflow Windows' default `MAX_PATH` on a long project
path); see Change Log for the exact fix if this bites another machine.
Still deliberately NOT in `base.txt`/`dev.txt` (torch is large enough
that a routine `pip install -r requirements/dev.txt` shouldn't pull it
in unasked).

**Hybrid search + reranking (`backend/app/rag/retrieval/
hybrid_search.py`, `backend/app/rag/retrieval/rerank.py`,
`backend/app/rag/reranking.py`):** both stages wired and **verified
live with real models** (2026-08-25 — see Change Log, two entries).
Stage 1: dense vector search (BGE-M3 query embedding) + Weaviate's BM25
via its native `hybrid()` query (plan.md section 12); `alpha` (default
0.5) balances the two, not yet tuned against an evaluation set (none
exists yet). Returns the raw ranked Top-`limit` (default 30) candidates
with citation metadata (document name, page span). Stage 2 (`--rerank`
flag): `BAAI/bge-reranker-v2-m3` (CrossEncoder via sentence-transformers,
same `requirements/embeddings.txt` install as BGE-M3) rescores and
narrows to a relevance-based Top-`top_k`-max (default 10, NOT forced to
exactly that count — `score_threshold` param exists but has no default
yet, no evaluation set to derive one from). CLI: `python -m
scripts.search "query" [--limit 30] [--alpha 0.5] [--rerank
[--top-k 10]]`.

**Grounded answer generation (`backend/app/rag/generation/
context_builder.py`, `backend/app/rag/generation/answer_generator.py`):**
wired and **verified live with real Groq + real retrieved context**
(2026-08-25 — see Change Log). `build_context()`/`build_citations()`
(plan.md section 20) dedupe by `chunk_id` and label each chunk
`[Source N]`; citations are built ONLY from retrieved metadata, never
parsed from the LLM's output (plan.md section 25: "Do NOT hallucinate
page numbers"). `generate_answer()` composes a grounded-answer prompt
(plan.md section 24: answer only from context, say so if not found,
preserve numbers/dates/names) and sends it through the existing
`LLMClient` abstraction unchanged (Groq when `GROQ_API_KEY` is set) —
no new LLM-provider code was needed. Skips the LLM call entirely and
returns a fixed "not found" answer when zero chunks were retrieved at
all (distinct from the LLM itself declining despite having context —
both were observed live, see Change Log). CLI: `python -m scripts.ask
"question" [--rerank] [--top-k 10]` runs the whole pipeline (Stage 1 →
optional Stage 2 → context → answer) end to end.

**Query classification, rewriting, and answer verification (plan.md
sections 14, 15, 26) — implemented 2026-08-25, verified live 2026-08-26:**
- `backend/app/rag/classification.py` — `classify_query()`. LLM call,
  GENERAL vs RAG_REQUIRED, defaults to RAG_REQUIRED on any ambiguous
  reply. Confirmed live: "What is 2 + 2?" → GENERAL, "How many days of
  annual leave..." → RAG_REQUIRED, both correct.
- `backend/app/rag/query_rewriting.py` — `rewrite_query()`. LLM call;
  falls back to the original query verbatim if the rewrite looks
  unreliable. The *original* query (not the rewrite) is still what's
  sent to the LLM for the final answer, per plan.md section 17. Confirmed
  live (a real Groq call happens for this step on the RAG branch).
- `backend/app/rag/generation/verification.py` — `verify_answer()`. LLM
  call checking whether an answer is grounded in its context; an
  unsupported verdict causes the caller to substitute a refusal message.
  Confirmed live both ways as of 2026-08-26: SUPPORTED answers returned
  as-is (e.g. "7 days of Casual Leave per calendar year," grounded in
  4 real citations), AND the refusal path itself — asked "What is the
  process for applying for leave?" against the same embedded document
  twice, both times got the refusal message ("...could not fully verify
  that this answer is supported...") instead of a hallucinated answer.
  First live sighting of the UNSUPPORTED branch (previously only
  exercised via unit test with a scripted fake reply).
- All three are wired together in the LangGraph workflow (next section),
  NOT used by `scripts.ask`/`/api/v1/rag/ask`, which intentionally stay
  simple (always-RAG, no rewriting/verification) — see their own entries.

**Full LangGraph workflow (`backend/app/graph/`) — implemented
2026-08-25, verified live 2026-08-26:** `build_graph(llm_client,
weaviate_client)` in `workflow.py` compiles plan.md's actual pipeline
(previously just a placeholder echo node, kept only to prove LangGraph
compiled):
```
START -> classify -> GENERAL -> llm -> END
                   -> RAG_REQUIRED -> rewrite -> retrieve (hybrid+rerank) -> generate -> verify -> END
```
`GraphState` (`state.py`) carries `original_query`, `classification`,
`rewritten_query`, `chunks`, `citations`, `answer`, `verified`,
`response` — `initial_state(query)` builds a fully-populated starting
dict so no node/caller ever hits a missing key on a path that skips some
nodes (e.g. GENERAL never runs `retrieve`). Dependencies (LLM client,
Weaviate client) are captured via closures built inside `build_graph()`,
not threaded through state. This graph drives `POST /api/v1/chat` (see
above). **Confirmed live via the actual console log** (not just unit
tests against a scripted fake `LLMClient`): both branches produce the
exact call sequence the graph's structure predicts — see the Change Log
entry for the full trace. The regenerate-vs-refuse fork on an
UNSUPPORTED verdict has only been exercised by unit test so far, not
live (no real answer has failed verification yet in testing).

**Auto-ingestion on startup — REMOVED 2026-08-26 (user request), replaced
by an explicit Documents API.** It was implemented 2026-08-25 and later
live-verified, but then hit a real bug (a document could be archived as
"done" without ever being embedded — see the Change Log entry below) and
the user asked to move to explicit HTTP-driven ingestion instead of
folder-watching. See the next entry.

**Documents API (`backend/app/api/routes/documents.py`,
`backend/app/services/document_service.py`,
`backend/app/rag/ingestion/embedder.py`) — implemented 2026-08-26, NOT
yet live-verified:** replaces auto-ingest. Two explicit steps instead of
one automatic one:
1. `POST /api/v1/documents/upload` (multipart) — saves the file under
   `DOCUMENTS_DIR` (repurposed; no longer auto-scanned) and runs the
   existing `ingest_pdf()` (extract/clean/chunk/Postgres-write) inline.
   Returns `document_id`/`version`/`chunk_count`/`is_duplicate`. Does
   NOT embed.
2. `POST /api/v1/documents/{document_id}/ingest[?version=N&force=bool]`
   — embeds that version's chunks into Weaviate via
   `app/rag/ingestion/embedder.py`'s `embed_document()` (extracted from
   the old auto-ingest module, also now used by `scripts.embed_document`
   instead of its own duplicate copy). First checks
   `is_already_embedded()` (queries Weaviate directly by
   document_id+version — the actual source of truth, not a separately
   tracked flag that could drift) and skips re-embedding unless
   `force=true`; embedding is idempotent (deterministic-UUID upsert) so
   `force` is always safe.

Plus `GET /api/v1/documents` (list all, with each version's live
embedded status), `GET /api/v1/documents/{id}` (one document), and
`DELETE /api/v1/documents/{id}` (removes Postgres rows, stored chunk
JSON, and any Weaviate chunks for that document — `collection.data.
delete_many()` filtered by `document_id`).

Unit-tested with fakes (`DocumentRepository`/Weaviate client faked,
`ingest_pdf`/`embed_document`/`is_already_embedded` monkeypatched — see
`tests/unit/test_document_service.py`, `test_embedder.py`). **Confirmed
live 2026-08-26** (`curl.exe -F` for the multipart upload —
`Invoke-RestMethod` on Windows PowerShell 5.1 has no `-Form` support):
upload correctly recognized a re-uploaded file as a duplicate
(`is_duplicate: true`, same `document_id`); ingest without `force`
correctly skipped an already-embedded version (`already_embedded:
true`, 0 chunks written); ingest with `force=true` correctly re-embedded
anyway (7 chunks written, matching the original count); `GET
/api/v1/documents` and `GET /api/v1/documents/{id}` both correctly
report `is_embedded: true`; `POST /api/v1/chat` afterward returned a
real grounded answer, proving the new upload/ingest path feeds retrieval
exactly like the removed auto-ingest path did. `DELETE
/api/v1/documents/{id}` and the admin reset endpoints below are still
unverified live — destructive, deferred until the user is ready to
test them specifically.

**Admin reset endpoints (`backend/app/api/routes/admin.py`,
`backend/app/services/admin_service.py`) — implemented 2026-08-26, NOT
yet live-verified:** `POST /api/v1/admin/reset/postgres` deletes all
`documents` (cascades to `document_versions`) and `chat_sessions`
(cascades to `messages`) rows — does not touch Weaviate or disk files.
`POST /api/v1/admin/reset/vector-store` deletes + recreates the
`DocumentChunk` Weaviate collection — does not touch PostgreSQL.
`POST /api/v1/admin/reset/all` runs both. No auth — this project has
none yet; dev/test convenience only, documented as such in the README.
Unit-tested with fakes (`tests/unit/test_admin_service.py`).

**Architecture (must stay this shape):**
```
API (routes, no business logic) -> Service -> Repository / Database
                                            -> LangGraph workflow (app/graph) -> LLM abstraction -> provider
                                                                               -> RAG pipeline (app/rag)
```
`/api/v1/chat` now goes through the graph; `/api/v1/rag/ask` calls the
RAG pipeline functions directly (bypassing the graph, no
classification) — both are legitimate, intentionally different entry
points, not a case of one being stale.

**Explicitly NOT implemented yet** (do not build until asked — see
`../../plan.md` section "Deferred"/"V1 Features" for the full order):
table extraction, header/footer detection, contextual chunking,
parent-child retrieval, metadata filtering (ACL/tenant — the `filters`
param exists on `hybrid_search()` but nothing populates tenant/ACL
properties yet), conversation history feeding retrieval (each chat turn
is independent — the graph has no memory of earlier turns in the same
session yet, even though they're persisted), full background-ingestion
job tracking (`upload_jobs` is still schema-only — upload/ingest are
synchronous HTTP calls for now, not a background job queue), a
regenerate loop on a failed verification (current behavior: refuse, not
retry — plan.md's diagram allows either), evaluation, tracing/metrics,
rate limiting, auth/authz, tenant isolation, MCP, CrewAI, DSPy, frontend,
Redis.
(PDF ingestion, chunking, document dedup/versioning, BGE-M3 embeddings,
hybrid search, reranking, context construction, grounded answer
generation with citations, query classification, query rewriting, AND
answer verification ARE now implemented — see the entries above and the
Change Log. OCR was implemented on 2026-08-25 and then fully removed on
2026-08-26 at the user's request — see both entries; scanned/image-based
PDFs are simply not supported.)

**HTTP endpoint for the RAG pipeline:** `POST /api/v1/rag/ask`
(`backend/app/api/routes/rag.py` → `RagQueryService`
(`backend/app/services/rag_service.py`)) now exposes the same
retrieval → optional rerank → grounded-answer flow as `scripts.ask` over
HTTP — the CLI-only gap noted above is closed. **Not yet verified live**
(wired + unit-tested with fakes only — see Change Log entry below for
exactly what has/hasn't been exercised against the real stack). The
Weaviate client is opened once at app startup (`app.main`'s `lifespan`,
stored on `app.state.weaviate_client`) rather than per request — a
deliberate departure from the CLI scripts, which open/close a client per
invocation; that pattern doesn't scale to a long-lived server process.

**Key config:** `backend/app/core/config.py` (`Settings`) reads everything
from env vars — `backend/.env` (local, gitignored) /
`backend/.env.example` (committed template). Var names as of now:
`ENVIRONMENT`, `POSTGRES_HOST/PORT/USER/PASSWORD/DB`,
`WEAVIATE_HOST/PORT/GRPC_PORT`, `GROQ_API_KEY`, `GROQ_MODEL`.
`GROQ_API_KEY` is now set in `backend/.env` (user's own key, not
recorded anywhere in this log) and the live call is verified working.

**Folders that exist but are intentionally empty scaffolding** (for later
phases — do not fill them speculatively): `app/agents/`, `app/crew/`,
`app/dspy/`, `app/mcp/`, `app/rag/`, `app/models/`, `app/observability/`,
`mcp-servers/`, `frontend/`, most of `data/` and `docs/`.

**Docker:** `docker-compose.yml` at repo root runs ONLY `postgres` +
`weaviate`. The backend runs locally from `backend/.venv`, not in a
container, for this phase (a `Dockerfile` exists for later use but no
compose service currently builds it). Postgres is published on host port
**5433** (not the default 5432) — this machine has another, unrelated
Postgres already bound to 5432 (confirmed via `netstat`), so 5432 was
reassigned to avoid silently connecting to the wrong database. Weaviate
stays on its default ports (8080, 50051) — no conflict was found there.

---

## Change Log (newest first)

### 2026-08-26 — Multiple Groq API keys with automatic rate-limit fallback
**By:** Claude (Sonnet 5), this session.
**Why:** Immediately after diagnosing the Groq daily-token-quota 500s
(previous entry), user provided 3 additional Groq API keys and asked
for automatic fallback to the next key whenever the current one hits
its rate limit, keeping the existing key as the primary
(`GROQ_API_KEY`).

Changed:
- `backend/.env` (real, gitignored) — added `GROQ_API_1`/`_2`/`_3` with
  the 3 new keys; `GROQ_API_KEY` unchanged as the primary/first-tried
  key. `backend/.env.example` — added the same 3 vars, blank (template
  only, no real values), documented as optional.
- `backend/app/core/config.py` — `Settings` gained `GROQ_API_1`/`_2`/
  `_3` (each optional, default `""`) and a `groq_api_keys` property
  returning all configured keys in order with blanks dropped.
- `backend/app/llm/base.py` — `GroqLLMClient` now takes `api_keys:
  list[str]` (was a single `api_key: str`) and holds one `AsyncGroq`
  client per key. On `groq.RateLimitError` it retries the *same*
  request against the next key in the list instead of failing;
  `_current_index` is sticky (sticks to whichever key last worked, so a
  later call starts there directly rather than re-trying already-dead
  keys from the front every time). Only raises our `RateLimitError`
  (429) once every configured key has been tried and rate-limited on
  this call. A non-rate-limit `groq.APIError` still raises
  `ServiceUnavailableError` (503) immediately, without trying other
  keys — a different key wouldn't fix an auth failure or a Groq-side
  5xx.
- `backend/app/api/dependencies.py` — `get_llm_client()` now passes
  `settings.groq_api_keys` (the full list) instead of a single
  `GROQ_API_KEY`; falls back to `StubLLMClient` only if that list is
  entirely empty.
- `backend/tests/unit/test_llm_base.py` — rewritten for the new
  constructor signature; covers: first key succeeding without touching
  others, falling through on a rate limit, the sticky index skipping an
  already-dead key on a later call, raising once all keys are
  rate-limited, a non-rate-limit error raising immediately without
  trying other keys, and the empty-key-list constructor guard.
- `backend/README.md` — new "Multiple Groq API keys (automatic
  rate-limit fallback)" section; updated the `/api/v1/chat` endpoint
  doc to point to it.

Full suite run by the user: **73 passed** (up from 69 — 6 new
`test_llm_base.py` cases added, 2 old ones replaced for the new
constructor signature). **Not yet live-verified against the real Groq
API** — next step: exhaust (or simulate exhausting) the primary key and
confirm a real request actually falls through to `GROQ_API_1` rather
than just passing in unit tests with fakes.

### 2026-08-26 — Root cause of the `rag_chat_test` 500s found and fixed: Groq daily token quota exhausted, now surfaced cleanly instead of a generic 500
**By:** Claude (Sonnet 5), this session.
**Why:** With file logging in place (previous entry), the user re-ran
`rag_chat_test` and this time it failed 12/12 questions immediately —
the real traceback (now captured) shows the actual cause has been the
same the whole time:
```
groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate
limit reached for model `openai/gpt-oss-120b` ... on tokens per day
(TPD): Limit 200000, Used 199798, Requested 1513. Please try again in
9m26.352s. ...'}}
```
This is a hard daily token quota on the Groq free/on-demand tier for
this model (200,000 tokens/day), not a bug in retrieval, chunking, or
the graph. It explains the EARLIER run's exact pattern too (8 clean
successes, then 12 consecutive identical failures that never
recovered) — the day's quota ran out partway through that run and
stayed out. Root cause is entirely external (nothing to "fix" about
the quota itself — wait for the daily reset or upgrade the Groq plan),
but the app was swallowing it into an opaque `500 internal_error`
indistinguishable from any other bug, via the generic unhandled-
exception handler in `app/core/exceptions.py`. That handling gap is a
real, fixable problem on its own — the next rate-limit hit (quota,
per-minute, whatever) should be immediately diagnosable from the HTTP
response alone, not require reading server logs.

Changed:
- `backend/app/core/exceptions.py` — added `RateLimitError` (429,
  `error_code: "rate_limited"`).
- `backend/app/llm/base.py` — `GroqLLMClient.generate_reply()` now
  catches `groq.RateLimitError` specifically (-> our `RateLimitError`,
  429) and the broader `groq.APIError` (-> `ServiceUnavailableError`,
  503) around the `chat.completions.create()` call, logging each
  before re-raising. Every caller (classification, query rewriting,
  chat/RAG answer generation, verification — all go through this one
  method) now benefits without needing its own try/except.
- `backend/tests/unit/test_llm_base.py` (new) — constructs real
  `groq.RateLimitError`/`groq.APIError` instances (via `httpx.Response`)
  and asserts they're translated correctly; no real Groq call made.

**Not yet re-verified live** — next step: wait for the Groq daily quota
to reset (or use a different/paid key), re-run `rag_chat_test`, and
confirm failures now return `429 {"error_code":"rate_limited",...}`
instead of the generic 500 if the quota is hit again, and that the
actual RAG answers are otherwise correct once real Groq calls succeed
(chunking/retrieval were never actually implicated by any evidence in
this investigation — the external analysis that first flagged Q9-Q20
as a "retrieval failure" was reasoning from a null-stripped summary
file, not the raw error responses).

### 2026-08-26 — Added rotating file log handler, to diagnose the unresolved `rag_chat_test` 500 errors
**By:** Claude (Sonnet 5), this session.
**Why:** User shared the `rag_chat_test` harness's full run output
(`output/latest.json`) and an external analysis of it claiming a
retrieval/chunking problem (Q9-Q20 "answered null" -> theorized as
later-document chunks not being retrieved). Reading the RAW run JSON
(not the null-stripped `qa_extracted.json` the external analysis was
based on) shows this diagnosis is wrong: Q9-Q20 all failed with a real
`HTTP 500 {"error_code":"internal_error"}` from BOTH `/api/v1/chat` AND
every `/api/v1/rag/ask` rerank-variant replay — a genuine unhandled
server exception, not empty retrieval context. This matches an
already-flagged, still-unresolved earlier log entry: exactly 8 clean
successes then 12 consecutive identical failures that never recover —
something broke partway through the run and stayed broken (rate
limiting, a resource leak, or a crashed client are the leading
suspects). Diagnosis is blocked on the actual traceback, which
`logger.exception(...)` in `app/core/exceptions.py`'s catch-all handler
already logs — but only to stdout (`app/core/logging.py` had no file
handler), so it's gone once the terminal scrollback is gone.

Changed `backend/app/core/logging.py`: `configure_logging()` now
attaches a `RotatingFileHandler` (5MB x 3 backups) writing to
`backend/logs/app.log`, in addition to (not instead of) the existing
stdout handler. Same formatter, same log level, same call sites — no
behavior change beyond "the traceback also survives to disk now."
`backend/logs/` isn't tracked (`.gitignore` already has `*.log`).

**Next step (not yet done):** user will re-run `rag_chat_test` against
the 20 leave-policy questions now that logging is captured to file;
read `backend/logs/app.log` around the first 500 to get the real
traceback and fix the actual cause, rather than the (incorrect)
chunking/retrieval theory from the external analysis.

### 2026-08-26 — Full project overview written here (artifact retracted per user request)
**By:** Claude (Sonnet 5), this session.
**Why:** User asked for a full project overview/summary. It was first
published as a Claude Artifact (a web page), but the user then asked for
it removed and for the summary to live directly in this file instead.
**Note:** there is no tool available to delete an already-published
artifact outright (only individual assets within one) — the page may
still be reachable at its URL unless the user removes it themselves via
`/artifacts` in the Claude Code terminal. It is no longer being treated
as a deliverable; this section is the authoritative overview going
forward. Everything below is a condensed, organized restatement of the
detailed entries elsewhere in this file — nothing here is new work.

#### What this project is

A RAG (Retrieval-Augmented Generation) support chatbot backend:
FastAPI + LangGraph, PostgreSQL for chat/document persistence, Weaviate
for vector search, Groq (`openai/gpt-oss-120b`) as the LLM. A question
is classified as general or document-related; document questions are
rewritten, retrieved (hybrid dense+BM25 search, then reranked), answered
strictly from retrieved context with citations, then checked for
groundedness — an unsupported answer is swapped for a refusal instead of
being shown to the user. Every stage below was built and then proven
against real Postgres/Weaviate/Groq/BGE-M3 calls, not just unit-tested
against fakes, before being marked done — this file's own convention
throughout its history.

#### Request flow through `/api/v1/chat` (`app/graph/workflow.py`)

```
question -> classify -> GENERAL      -> llm reply           -> response + citations (empty)
                      -> RAG_REQUIRED -> rewrite query
                                      -> retrieve (hybrid search + rerank)
                                      -> generate grounded answer
                                      -> verify groundedness -> response + citations
```
A general question triggers exactly 2 Groq calls and zero retrieval; a
RAG question triggers exactly 4 Groq calls (classify/rewrite/generate/
verify) plus the embedding + hybrid-search + rerank steps in between —
both confirmed live via console trace, not just response shape.

#### Pipeline stages — status

- **PDF ingestion** (`app/rag/ingestion/`) — Verified live. Extract
  (`pypdf`) -> whitespace clean -> fixed-size token chunking
  (`tiktoken`, 100 tokens / 20 overlap) -> sha256 content-hash dedup ->
  versioned `Document`/`DocumentVersion` rows.
- **Embeddings + vector write** (`app/rag/embeddings.py`,
  `vector_writer.py`) — Verified live. BGE-M3 (1024-dim) via
  `sentence-transformers`, upserted into Weaviate's `DocumentChunk`
  collection by a deterministic UUID (`document_id:version:chunk_index`)
  — re-embedding is a no-op, never a duplicate.
- **Hybrid search, Stage 1** (`app/rag/retrieval/hybrid_search.py`) —
  Verified live. Dense vector + Weaviate's native BM25, fused via
  `alpha` (0.0 = BM25 only, 1.0 = vector only, default 0.5, untuned —
  no evaluation set exists yet to tune it against). Confirmed genuine
  relevance discrimination on real test documents, not just "returns
  what's in the DB."
- **Reranking, Stage 2** (`app/rag/reranking.py`,
  `retrieval/rerank.py`) — Verified live. `BAAI/bge-reranker-v2-m3`
  cross-encoder narrows the Top-30 candidates to a relevance-ranked
  Top-10 max (not padded to exactly 10).
- **Grounded generation + citations**
  (`app/rag/generation/context_builder.py`, `answer_generator.py`) —
  Verified live. Context built only from retrieved chunks; citations
  are built only from retrieved metadata, never parsed out of the LLM's
  own output (deliberately, to avoid hallucinated page numbers).
  Confirmed live that the model refuses to answer from outside
  knowledge when the retrieved context doesn't cover the question.
- **Classification / rewriting / verification**
  (`app/rag/classification.py`, `query_rewriting.py`,
  `generation/verification.py`) — Verified live, including the
  refusal branch (an answer judged unsupported by its own citations was
  correctly replaced with a refusal message, not shown as-is).
- **Full LangGraph workflow driving `/api/v1/chat`**
  (`app/graph/workflow.py`, `chat_service.py`) — Verified live, as
  described in the flow diagram above.
- **Auto-ingest on startup** — Built, then fully removed. It caught a
  real bug (a document could be archived "done" without ever actually
  being embedded), got fixed, then was removed entirely at the user's
  request in favor of explicit, user-driven control.
- **Documents API** (`app/api/routes/documents.py`,
  `services/document_service.py`) — Verified live. Explicit
  `POST /upload` -> `POST /{id}/ingest` replaced auto-ingest end to end:
  upload/dedup, ingest-with-force, list, and get were all confirmed
  against a real PDF, followed by a real grounded chat answer proving
  the new path feeds retrieval correctly.
- **Delete document / Admin reset endpoints**
  (`DELETE /documents/{id}`, `POST /admin/reset/*`) — Wired and
  unit-tested with fakes only; destructive, deliberately not yet
  exercised for real.
- **OCR fallback for scanned PDFs** — Built (for one real scanned
  119-page manual), then fully deleted at the user's explicit request.
  Scanned/image-only PDFs are simply unsupported now, by design, not by
  omission.

**Deliberately not built yet** (each is a named, later step in
`../../plan.md`, not an oversight): table extraction, header/footer
detection, contextual chunking, parent-child retrieval, tenant/ACL
metadata filtering, conversation-history-aware retrieval, a background
ingestion job queue (`upload_jobs` table exists, unused), a
regenerate-on-failed-verification loop (current behavior is refuse, not
retry), an evaluation framework, tracing/metrics, rate limiting,
auth/authz, MCP, CrewAI, DSPy, a frontend, Redis.

#### API surface

| Endpoint | Status | What it does |
|---|---|---|
| `GET /health` | Live | App + Postgres + Weaviate connectivity. |
| `POST /api/v1/chat` | Verified live | Full LangGraph workflow (see flow diagram above); persists to Postgres. |
| `POST /api/v1/rag/ask` | Verified live | Direct pipeline, no classification; exposes `alpha`/`rerank`/`top_k` for retrieval experiments. |
| `POST /api/v1/documents/upload` | Verified live | Multipart upload; extract/chunk/dedup, returns a `document_id`. |
| `POST /api/v1/documents/{id}/ingest` | Verified live | Embeds a version into Weaviate; skips if already embedded unless `force=true`. |
| `GET /api/v1/documents`, `GET /api/v1/documents/{id}` | Verified live | List / fetch, with live per-version embedded status. |
| `DELETE /api/v1/documents/{id}` | Wired only | Removes Postgres rows, chunk JSON, Weaviate objects — not yet live-verified. |
| `POST /api/v1/admin/reset/{postgres,vector-store,all}` | Wired only | Dev-only destructive resets. No auth — documented as such. |

#### Standalone test harness — `backend/rag_chat_test/`

Built this session, entirely outside `backend/app` — talks to the
running backend only over HTTP, like any other client, and runs only
when invoked by hand (`python test_runner.py`), never automatically.

- Reads an unlimited list of questions from `input/questions.json`,
  sends each to `POST /api/v1/chat` one at a time (never in parallel).
  No forced timeout by default (`config.REQUEST_TIMEOUT_SECONDS =
  None`) — waits for a genuine answer however long it takes; settable
  back to a number of seconds if a hard cutoff is wanted again.
- Every successful answer is scored by an LLM judge (Groq,
  `openai/gpt-oss-120b`) on relevance, groundedness against its own
  citations, completeness, clarity, and hallucination risk.
- Each question is *also* replayed through `POST /api/v1/rag/ask`
  across four retrieval/reranking configurations (`config.
  RERANK_VARIANTS`: no-rerank hybrid, rerank hybrid, rerank
  vector-only, rerank BM25-only) — each independently judged, so every
  question ends up with several differently-retrieved answers to
  compare, plus a computed best variant.
- Writes one detailed JSON report per run to `output/run_<timestamp>
  .json` / `output/latest.json`; `extract_qa.py` strips a report down
  to plain `{question_id, question, answer}` pairs
  (`output/qa_extracted.json`).

**What the last real run (2026-08-26, 20 questions against the leave
policy document) found:**
- 8 of 20 questions succeeded; average overall score 8.5/10 on the
  successful ones; hallucination risk was low on 7, medium on 1.
  Response times on successful questions ranged 2.6s to ~2m6s.
- **Real issue surfaced, not a harness bug:** questions 9 through 20 —
  every question after the 8th — all failed identically with a
  `500 internal_error` from `/api/v1/chat`. 12 consecutive identical
  failures right after 8 clean successes points at the backend
  degrading partway through the run (resource exhaustion, a rate
  limit, or a crashed dependency are the likely candidates) — worth
  checking backend server logs from that time window. **Not yet
  diagnosed or fixed** — flagging it here so it isn't lost.
- Reranked hybrid search scored highest on quality among the four
  variants (8.75/10) but took roughly 8x longer than skipping
  reranking entirely (34.0s vs 4.1s average) for a nearly identical
  score on this small sample — worth a larger run before concluding
  reranking isn't earning its cost here.

### 2026-08-26 — `rag_chat_test`: added multi-variant reranking/retrieval comparison
**By:** Claude (Sonnet 5), this session.
**Why:** User asked for "different types of reranking" and multiple
answer variants for the same single question — `/api/v1/chat` runs one
fixed pipeline with no retrieval knobs exposed, but `/api/v1/rag/ask`
(`app/api/routes/rag.py`, `app/schemas/rag.py::RagAskRequest`) already
exposes `alpha` (BM25-vs-vector weighting), `rerank` (on/off), and
`top_k` — reused that existing endpoint rather than adding anything new
to the backend.

Changed (test-harness-only, no backend app files touched):
- `backend/rag_chat_test/main_script/config.py` — added
  `RAG_ASK_API_URL`, `ENABLE_RERANK_VARIANTS` (default `True`), and
  `RERANK_VARIANTS` (4 default variants: no-rerank hybrid, rerank
  hybrid, rerank vector-only, rerank BM25-only — each with its own
  `alpha`/`rerank`/`limit`/`top_k`, freely editable/extendable).
- `backend/rag_chat_test/main_script/test_runner.py` — refactored the
  HTTP call into a shared `_post_json()` used by both `call_chat_api()`
  and the new `call_rag_ask_api()`; `evaluate_with_llm()` now takes
  `(answer, citations, response_time)` directly instead of assuming the
  `/api/v1/chat` response shape (`reply`/`citations`), so it works for
  `/api/v1/rag/ask`'s shape (`answer`/`citations`) too. For every
  question, after the normal `/api/v1/chat` call+evaluation, the runner
  now also calls `/api/v1/rag/ask` once per configured variant,
  evaluates each with the same Groq judge rubric, and records a
  `best_variant` (highest `overall_score` among the chat answer and all
  rerank variants) on that question. `build_summary()` gained
  `rerank_variant_comparison`: per-variant avg score, avg response time,
  and how many questions each variant "won."
- `backend/rag_chat_test/README.md` — documented the new variant
  comparison, updated the sample output JSON to show
  `rerank_variants`/`best_variant` per question and
  `rerank_variant_comparison` in the summary.

**Not yet run** — extended, syntax-checked (`ast.parse`), not yet
executed against a live backend.

### 2026-08-26 — `rag_chat_test`: removed per-question timeout (waits indefinitely)
**By:** Claude (Sonnet 5), this session.
**Why:** User hit a real `TIMEOUT` result on a genuinely slow question
(multi-part question needing RAG classify/rewrite/retrieve/generate/
verify — several sequential Groq calls) and asked for no timeout at all:
let each question run until the API actually answers, however long that
takes.

Changed:
- `backend/rag_chat_test/main_script/config.py` —
  `REQUEST_TIMEOUT_SECONDS` default changed from `10` to `None`.
  `requests.post(..., timeout=None)` blocks indefinitely instead of
  raising after N seconds; can still be set back to a number of seconds
  if timeout behavior is wanted again later.
- `backend/rag_chat_test/main_script/test_runner.py` — startup log line
  now prints "none (waits indefinitely)" instead of `Nones` when the
  timeout is disabled.

### 2026-08-26 — Added standalone `rag_chat_test` harness (backend/rag_chat_test/)
**By:** Claude (Sonnet 5), this session.
**Why:** User wants a repeatable way to batch-test `/api/v1/chat` with an
arbitrary/unlimited list of questions, measure real response time per
question, and get an LLM-judged quality evaluation (relevance,
groundedness vs. citations, completeness, clarity, hallucination risk)
for each answer — without touching any backend app code.

Added (all new files, nothing existing touched):
- `backend/rag_chat_test/input/questions.json` — user-editable list of
  `{id, question, conversation_id}` entries; unlimited questions
  supported.
- `backend/rag_chat_test/main_script/config.py` — target API URL
  (`http://localhost:8000/api/v1/chat`), 10-second per-question timeout,
  Groq judge model config (`openai/gpt-oss-120b`, key overridable via
  `GROQ_API_KEY` env var).
- `backend/rag_chat_test/main_script/test_runner.py` — runs only when
  invoked manually (`python test_runner.py`), never automatically. Sends
  questions to the chat API strictly one at a time (never in parallel);
  any question that doesn't get a response within 10s is aborted
  (`requests` timeout) and marked `TIMEOUT`, then the runner continues to
  the next question. Every successful answer is separately sent to Groq
  with a scoring rubric prompt and the judge's parsed JSON evaluation is
  attached to that question's record. Writes a full detailed JSON report
  (per-question timing/status/answer/citations/evaluation, plus an
  aggregate summary of averages and counts) to
  `backend/rag_chat_test/output/run_<timestamp>.json` and `output/latest.json`.
- `backend/rag_chat_test/main_script/requirements.txt` (`requests`,
  `groq==0.11.0` — matches the version already pinned in
  `backend/requirements/base.txt`).
- `backend/rag_chat_test/README.md` — usage instructions and an explicit
  "what it can't measure" note: this harness only observes what
  `/api/v1/chat`'s HTTP response actually returns (reply, citations,
  round-trip time) — it has no visibility into internal server-side
  numbers the endpoint doesn't expose (per-stage retrieval/reranking
  scores, individual LLM-call latencies inside the LangGraph workflow),
  since getting those would require instrumenting the backend itself,
  which was explicitly out of scope for this ask.

**Not yet run** — script has been written and syntax-checked
(`ast.parse`) but not executed against a live backend/Groq call yet;
that's the next step whenever the user runs it themselves.

### 2026-08-26 — Chunk size 600->100 tokens; clarified /api/v1/chat needs no document id
**By:** Claude (Sonnet 5), this session.
**Why:** User asked for smaller chunks (100 tokens instead of 600) and,
separately, said they had to "give doc" to get a chat reply and wanted
that removed so any question against any uploaded PDF "just works."
Investigated: `/api/v1/chat` and `/api/v1/rag/ask` never had a
document-id parameter — both already run hybrid search across every
embedded chunk with no document filter. The actual friction was almost
certainly Swagger UI auto-filling `"conversation_id": "string"` in the
example request body for `/api/v1/chat` (an unrelated, always-optional
field) — easy to mistake for a required id given the session's
doc_id-centric upload/ingest flow. `ChatService._parse_session_id()`
already treats an invalid/non-UUID value as "start a new session," so
sending the literal "string" doesn't error, but it's confusing.

Changed:
- `backend/app/core/config.py` — `CHUNK_SIZE_TOKENS` 600 -> 100,
  `CHUNK_OVERLAP_TOKENS` 100 -> 20 (kept proportional; overlap must
  stay below chunk size or `chunk_pages()` raises `ValueError`).
  `backend/README.md` updated to match. No test depends on these
  defaults (`test_ingestion.py` passes explicit values to
  `chunk_pages()` directly).
- `backend/app/schemas/chat.py` — `ChatRequest` gained a
  `json_schema_extra` example showing `conversation_id: null`, and an
  explicit docstring/description stating no document id exists or is
  needed, and that sending the literal placeholder text "string" isn't
  meaningful — purely a Swagger-UI/documentation fix, no behavior
  change (the endpoint always searched everything, with or without
  this change).

**Confirmed live 2026-08-26:** reset both stores via `/api/v1/admin/
reset/all`, re-uploaded `Leave_Policy.pdf` — 25 chunks (vs. 4-7 at the
old 600-token size for similar documents), embedded successfully, and
`/api/v1/chat` correctly answered "7 days of casual leave per year"
with a grounded citation. Smaller chunks did not break retrieval
quality on this question.

### 2026-08-26 — Replaced auto-ingest-on-startup with an explicit Documents API; added admin reset endpoints
**By:** Claude (Sonnet 5), this session.
**Why:** User asked to remove folder-watching auto-ingest entirely (it
just found a real correctness bug — see the entry below — and they
wanted explicit, user-driven control instead: upload a PDF, get its
`document_id` back, then decide when to trigger embedding with that id).
They also asked for endpoints to reset PostgreSQL and/or the Weaviate
vector store, separately and combined, for easier local testing/reset
between runs.

Removed:
- `backend/app/rag/ingestion/auto_ingest.py`, `backend/scripts/
  auto_ingest.py`, `backend/tests/unit/test_auto_ingest.py`.
- `run_auto_ingest()` call from `app/main.py`'s `lifespan`.
- `AUTO_INGEST_ON_STARTUP`, `INGESTED_DOCUMENTS_DIR` from
  `app/core/config.py`. `DOCUMENTS_DIR` kept, repurposed as "where
  uploaded PDFs are saved" (no longer auto-scanned on startup).

Added:
- `backend/app/rag/ingestion/embedder.py` — `read_chunks()`,
  `embed_document()` (moved out of the deleted auto-ingest module, now
  the single implementation shared by the new ingest endpoint and
  `scripts.embed_document`, which no longer duplicates this logic), and
  new `is_already_embedded()` (queries Weaviate directly by
  document_id+version rather than trusting a separately-tracked flag —
  same "check the real source of truth" lesson as the bug fixed below).
- `backend/app/services/document_service.py` (`DocumentService`) +
  `backend/app/api/routes/documents.py` — `POST /api/v1/documents/
  upload` (multipart; extract+chunk+Postgres-write only, returns
  `document_id`), `POST /api/v1/documents/{id}/ingest[?version=&force=]`
  (embeds via `embedder.py`, skips if already embedded unless
  `force=true` — safe either way since embedding upserts by a
  deterministic UUID), `GET /api/v1/documents` (list + live per-version
  embedded status), `GET /api/v1/documents/{id}`, `DELETE /api/v1/
  documents/{id}` (removes Postgres rows, stored chunk JSON, and
  Weaviate chunks for that document via `collection.data.delete_many()`
  filtered by `document_id`).
- `backend/app/repositories/document_repository.py` — `list_all()`,
  `get_by_id_with_versions()` (both eager-load `versions` via
  `selectinload`), `delete_document()`.
- `backend/app/services/admin_service.py` (`AdminService`) +
  `backend/app/api/routes/admin.py` — `POST /api/v1/admin/reset/
  postgres` (deletes `documents`/`chat_sessions`, cascades via existing
  FK `ondelete="CASCADE"`), `POST /api/v1/admin/reset/vector-store`
  (deletes+recreates the `DocumentChunk` collection), `POST /api/v1/
  admin/reset/all` (both). Not gated behind auth — none exists yet;
  documented in the README as dev-only.
- `app/core/exceptions.py` — `BadRequestError` (400), used for upload
  validation (non-PDF, empty file).
- `requirements/base.txt` — `python-multipart==0.0.20` (FastAPI's
  `UploadFile` needs it; wasn't previously installed since nothing used
  file uploads before).
- Tests: `tests/unit/test_document_service.py`,
  `tests/unit/test_admin_service.py`, `tests/unit/test_embedder.py` (all
  fakes/mocks, no real DB/Weaviate).

Updated `backend/README.md`: replaced the "Auto-ingest on startup"
section with "Documents API" and a new "Admin: resetting data" section;
updated the Endpoints list.

**Not yet live-verified** — next step: upload a real PDF via `POST
/api/v1/documents/upload`, call `POST /api/v1/documents/{id}/ingest`,
confirm it lands in Weaviate and `/api/v1/chat`/`/api/v1/rag/ask` can
retrieve it, same as the old auto-ingest path proved out.

### 2026-08-26 — Fixed auto-ingest bug: duplicates could be archived without ever being embedded
**By:** Claude (Sonnet 5), this session.
**Why:** User ingested a real PDF (`India-Leaves and Holiday Policy-042624.pdf`),
auto-ingest's log reported success ("completed -> moved to
completed_ingesting/"), but `/api/v1/chat` and `/api/v1/rag/ask`
afterward both reported the `DocumentChunk` Weaviate collection doesn't
exist at all (`GET /v1/schema` confirmed `classes: {}` live). Diagnosis
ruled out: Weaviate being a duplicate-content match against a stale row
(only one `documents` row existed, matching this exact file); a Docker
restart wiping the volume (`docker compose ps` showed 24h continuous
uptime, no restart); and test fixtures touching the real Weaviate
instance (all Weaviate-touching tests are mocked, confirmed via grep).

Root cause: `run_auto_ingest()` treated `IngestionResult.is_duplicate`
as equivalent to "already embedded in Weaviate," and skipped the
`_embed_document()` call entirely whenever `is_duplicate=True`. But
`is_duplicate` only reflects that the file's *content hash* already has
a `documents` row in Postgres — it says nothing about whether that
prior ingest ever successfully embedded into Weaviate. If an earlier
run ingested the text (writing the Postgres row + chunk JSON) but then
failed or skipped the embed step (e.g. Weaviate transiently unavailable,
or a manual Weaviate collection cleanup happening mid-session, as
happened earlier this session), the file is correctly left in
`data/documents/` for retry per the code's own logic — but the *next*
startup then sees the same content hash, reports `is_duplicate=True`,
and (under the old logic) skipped embedding again and archived it as
done. Net effect: a document could get archived as "completed" while
never having a single vector written to Weaviate.

Fix (`backend/app/rag/ingestion/auto_ingest.py`): always call
`_embed_document()` regardless of `is_duplicate`. This is safe because
`write_chunks_to_weaviate()` already upserts by a deterministic UUID
(`document_id:version:chunk_index`) — re-embedding an
already-embedded document is a no-op, not a duplicate write. This makes
the pipeline self-healing: any residual un-embedded "duplicate" gets
fixed automatically on the next startup instead of being silently
archived as broken forever.

Updated `test_run_auto_ingest_moves_duplicate_without_embedding` (which
asserted the old, incorrect "duplicates never embed" behavior via an
assertion-raising fake) into
`test_run_auto_ingest_reembeds_duplicate`, asserting `_embed_document`
IS called for a duplicate.

**Recovery for the affected document:** since the PDF was already moved
into `completed_ingesting/`, restarting alone won't retry it (auto-ingest
only scans `DOCUMENTS_DIR`) — it needs to be moved back to
`data/documents/` before the next app startup so the fixed logic
re-embeds it.

**Confirmed live:** moved the file back and restarted — startup log now
shows `Wrote 7 chunks to Weaviate for document_id=...` (previously
absent) before the move, `GET /v1/schema` shows the `DocumentChunk`
class exists, and `POST /api/v1/chat` with "What is the leave policy?"
returned a real grounded answer instead of the empty/no-context
response.

### 2026-08-26 — OCR support removed entirely (user request)
**By:** Claude (Sonnet 5), this session.
**Why:** User explicitly asked to remove OCR completely: no OCR-related
PDFs to worry about, and they didn't want the feature sitting in the
project at all (not just unused). Fully reverting the 2026-08-25 OCR
entry rather than leaving it dormant.

Deleted:
- `backend/app/rag/ingestion/ocr.py`, `backend/requirements/ocr.txt`,
  `backend/tests/unit/test_ocr.py`.

Reverted to their pre-OCR state:
- `backend/app/rag/ingestion/pdf_extractor.py` — `extract_pages()` lost
  its `ocr_fallback` param and the `ocr_page_image` import/call.
- `backend/app/rag/ingestion/pipeline.py` — `ingest_pdf()` lost its
  `ocr_fallback` param; `OcrUnavailableError` import/except removed.
- `backend/scripts/ingest_document.py` — `--ocr` flag removed.
- `backend/app/core/config.py` — `OCR_DPI`/`TESSERACT_CMD` settings
  removed.
- `backend/tests/unit/test_ingestion.py` — the two OCR-branching tests
  removed (the blank-PDF-via-`pypdf.PdfWriter` fixture only existed to
  test that branching).
- `backend/README.md` — "Ingest a scanned PDF (OCR fallback)" section
  removed; intro and auto-ingest section's OCR mentions rewritten to
  state plainly that scanned PDFs aren't supported.

Kept as-is (not OCR-specific, still correct without it):
- `backend/app/rag/ingestion/auto_ingest.py`'s zero-chunk safety check
  (the 2026-08-25 bug fix) — still valuable for any PDF that produces 0
  chunks, scanned or otherwise; only its log message's `--ocr` mention
  was removed.

**Verified this session:** `pytest` → **55 passed** (58 - 3 removed OCR
tests, nothing else broken).

**Practical note:** the immediate trigger was a different error —
`python -m scripts.ingest_document ..\data\documents\India-Leaves and
Holiday Policy-042624.pdf` failed with `unrecognized arguments: and
Holiday Policy-042624.pdf` because the path has spaces and wasn't
quoted in PowerShell. Unrelated to OCR removal — the fix is just to
quote the path: `python -m scripts.ingest_document "..\data\documents\
India-Leaves and Holiday Policy-042624.pdf"`.

### 2026-08-26 — Cleanup: cleared all test artifacts before real usage
**By:** Claude (Sonnet 5), this session.
**Why:** User is moving on to uploading a real document and asked to
remove `SQL-Manual.pdf`; a good moment to also clear out everything else
accumulated across the Part A/B test rounds rather than leave stale test
rows sitting in what's about to become real data.

Removed:
- All 4 rows in `documents`/`document_versions` via `DELETE FROM
  documents` (cascades): two `test_leave_policy.pdf` entries (turned out
  to be two *different* content hashes despite looking like the same
  file — the PDF library embeds a generation timestamp, so re-running
  the same `reportlab` snippet twice produces different bytes), the
  empty `SQL-Manual.pdf` row, `test_kitchen_policy.pdf`.
- The `DocumentChunk` Weaviate collection entirely (`client.collections.
  delete(...)`, had 2 objects) — simpler than filtering out specific
  objects, and nothing was in it worth keeping. It's recreated
  automatically (`ensure_chunk_collection()`) the next time anything is
  embedded.
- `backend/test_leave_policy.pdf`, `data/documents/completed_ingesting/
  test_kitchen_policy.pdf`, and all `data/processed/<id>/` folders.
- `SQL-Manual.pdf` itself in `data/documents/` was left alone — user is
  removing that one manually.

Database and vector store are now empty of documents/chunks;
`chat_sessions`/`messages` were untouched (out of scope, real
conversation history).

### 2026-08-26 — Verified live: auto-ingest happy path (ingest + embed + move)
**By:** user, confirmed in this session (Claude Sonnet 5 guiding).
**Why:** Closing out Part B. The previous entry only proved the
*rejection* path (scanned PDF correctly left in place); this proves the
*success* path with a real text-based PDF.

**Setup:** a fresh throwaway PDF (`test_kitchen_policy.pdf`, real text
via `reportlab`) was dropped directly into `data/documents/`, then the
app was started.

**Confirmed via console log, full chain:**
- `ingest_pdf()`: new `Document`/`DocumentVersion` created,
  `pages=1 chunks=1` (real content, unlike the scanned manual).
- `_embed_document()`: `DocumentChunk` collection schema check OK,
  BGE-M3 loaded (already-cached process, ~9s), `"Wrote 1 chunks to
  Weaviate for document_id=..."`.
- `run_auto_ingest()`: `"'test_kitchen_policy.pdf' completed -> moved to
  '...\data\documents\completed_ingesting\test_kitchen_policy.pdf'"` —
  the file was actually moved off disk (`shutil.move`, not just logged
  as if it were).

Also reconfirmed in the same log: `SQL-Manual.pdf` (still sitting in
`data/documents/` from the entry above) was independently, correctly
skipped again on this same startup — auto-ingest handled a genuine mix
of one good and one bad file in the same scan without either one
interfering with the other.

**Status: RESOLVED.** Part B — auto-ingest on startup, including both
the success path and the safety behavior around already-processed/empty
duplicates — is now fully verified live, not just unit-tested. Both
Part A and Part B from the original ask are now complete and confirmed
working end to end.

**Not yet done, still outstanding:** `SQL-Manual.pdf` itself remains
un-ingested in any usable form (still scanned, still 0 chunks) — running
it through OCR for real (`scripts.ingest_document --ocr --force`, ~119
pages, expect it to be slow) has never actually been attempted, only
wired and unit-tested. That's a separate, optional next step, not a
blocker on anything above.

### 2026-08-26 — Verified live: the empty-duplicate auto-ingest fix actually works
**By:** user, confirmed in this session (Claude Sonnet 5 guiding).
**Why:** Starting Part B (auto-ingest) testing. User dropped
`SQL-Manual.pdf` (the scanned manual) back into `data/documents/` and
started the app — this is precisely the "second startup" scenario the
2026-08-25 bug fix targeted (see that entry): the file's content hash
matched the `Document` row created during the very first ingest attempt,
so `ingest_pdf()` correctly reported it as a duplicate.

**Confirmed via console log:** `"Document 'SQL-Manual.pdf' already
ingested... skipping"`, followed by the fix's `_real_chunk_count()` path
re-reading the actual stored version rather than trusting the
duplicate's hardcoded `chunk_count=0` at face value, landing on the same
correct conclusion (genuinely 0 chunks) and logging: `"'SQL-Manual.pdf'
has 0 chunks... leaving in place rather than archiving it as done."`
File was NOT moved to `completed_ingesting/`. This is the second-startup
failure mode the fix was specifically written for, now proven against a
real duplicate, real file, real startup — not just the unit test that
faked `_real_chunk_count`'s return value.

**Not yet tested:** the *successful* auto-ingest path (a real
text-based PDF actually getting ingested, embedded, and moved into
`completed_ingesting/`) — `SQL-Manual.pdf` being scanned means it can
only ever exercise the "correctly reject" branch, never the "correctly
succeed" one. Next: drop a text-based PDF and confirm the full
happy path.

### 2026-08-26 — Verified live: full LangGraph workflow via `/api/v1/chat`
**By:** user, confirmed in this session (Claude Sonnet 5 guiding).
**Why:** Part A (classification/rewriting/verification + the real
LangGraph workflow) was wired and unit-tested with fakes only — this
confirms it actually works against real Groq/BGE-M3/Weaviate.

**Detour first:** several attempts landed on `/api/v1/rag/ask` instead
of `/api/v1/chat` (Swagger UI kept reopening the `rag` tag's last-used
operation), then a raw terminal `curl` attempt failed with
`Cannot bind parameter 'Headers'` — Windows PowerShell aliases `curl` to
`Invoke-WebRequest`, which does not accept real curl's `-H`/`-d` syntax.
Switched to `Invoke-RestMethod` with PowerShell-native parameters, which
worked immediately. **Worth remembering for future PowerShell users on
this project:** use `Invoke-RestMethod -Uri ... -Method Post
-ContentType "application/json" -Body '...'`, not `curl -X POST -H ... -d
...`, unless explicitly invoking `curl.exe` to bypass the alias.

**Verified via real console log, not just response shape:**
- `{"message": "What is 2 + 2?"}` → classified GENERAL. Log shows
  exactly 2 Groq calls (classify, then the direct reply) with *zero*
  Weaviate/embedding activity between them. Response:
  `citations: []` (empty), confirming retrieval was genuinely skipped,
  not just returned empty.
- `{"message": "How many days of annual leave do employees get?"}`
  (asked twice, two different sessions) → classified RAG_REQUIRED both
  times. Log shows exactly 4 Groq calls each time in the right order
  (classify → rewrite → generate → verify), with the Weaviate schema
  check + BGE-M3 query embed + reranker batches sandwiched between
  rewrite and generate — matching the graph's actual structure, not
  just a plausible-looking final answer. Both responses returned the
  correct grounded answer (`"Employees are/receive twenty (20) days of
  paid annual leave per calendar year"`) with the correct citation
  (`test_leave_policy.pdf`, page 1, matching `chunk_id`) — and neither
  was replaced by the unsupported-answer refusal, confirming
  verification judged both SUPPORTED.
- Persistence confirmed for all three requests: `chat_sessions` +
  `messages` (user + assistant) inserted and committed correctly, three
  distinct session UUIDs, matching the pre-existing (already-verified)
  persistence behavior — the new graph wrapping didn't break it.

**Status: RESOLVED.** Part A — query classification, query rewriting,
answer verification, and the full LangGraph workflow driving
`/api/v1/chat` — is now confirmed working end to end with real models,
not just passing unit tests against fakes. This closes the "NOT yet
live-verified" flag from the entry below.

### 2026-08-25 — Bug fix (caught before it bit): auto-ingest would silently archive empty scanned PDFs
**By:** Claude (Sonnet 5), this session.
**Why:** Before running any live test of Part A, checked what was
actually sitting in `data/documents/` — `SQL-Manual.pdf` was still
there (the scanned manual from the earlier OCR-motivating entry;
`auto_ingest.py` runs with `ocr_fallback=False`, matching
`ingest_pdf()`'s own default). Tracing through what would actually
happen on startup: first run, a fresh `Document` gets created with 0
chunks, and — before this fix — auto-ingest treated "didn't raise" as
"succeeded" and would have moved the file to `completed_ingesting/` as
if it had been usefully processed, even though nothing was actually
retrievable. Caught and fixed before the user ever started the app and
hit it.

Changes to `backend/app/rag/ingestion/auto_ingest.py`:
- Added a check: if a freshly-ingested (non-duplicate) file produced 0
  chunks, log a warning and leave it in place rather than moving it —
  it isn't actually done.
- Found a second, subtler version of the same bug while fixing the
  first: `ingest_pdf()`'s duplicate-detection path *always* returns
  `IngestionResult.chunk_count=0`, regardless of the existing
  document's real content (it's a pre-existing shortcut in
  `pipeline.py`, not something changed here) — so on the *second*
  startup, the same empty file would look like an ordinary duplicate
  and get archived anyway, since the fresh-ingest check above only
  looks at non-duplicates. Fixed by adding `_real_chunk_count()`, which
  trusts `IngestionResult.chunk_count` for a fresh ingest but re-reads
  the actual stored chunk JSON for a duplicate rather than trusting the
  hardcoded 0. `_embed_document()` reuses the same new `_read_chunks()`
  helper instead of reading `doc_version.storage_path` inline.
- `backend/tests/unit/test_auto_ingest.py` — updated the existing
  duplicate test (now needs `_real_chunk_count` faked, since the fake
  session can't do a real DB read) and added
  `test_run_auto_ingest_leaves_empty_duplicate_in_place`, which
  reproduces the second bug directly: a duplicate result whose real
  chunk count is faked as 0 must stay in place, not get archived.

**Verified this session:** `pytest` → **58 passed** (1 new).

**Practical note:** `SQL-Manual.pdf` itself was left untouched in
`data/documents/` throughout — no destructive action was taken on it,
this was pure code-reading-ahead before running anything live. It's
still there, still scanned, and will now be correctly skipped (left in
place, warned about) by auto-ingest rather than silently mis-archived.

### 2026-08-25 — Query classification/rewriting/verification, full LangGraph workflow wired into chat, auto-ingest on startup
**By:** Claude (Sonnet 5), this session.
**Why:** User asked for "the complete RAG system" per `plan.md` (they
said they'd test it later — this entry's work is explicitly NOT yet
live-verified, see below) plus a separate, concrete automation request:
run ingestion automatically on app startup for any PDF dropped in
`data/documents/`, moving finished ones to a distinctly-named completed
folder. Two independent asks, both scoped and built this session.

**Part A — the three remaining plan.md pipeline pieces + real graph:**

New files:
- `backend/app/rag/classification.py` — `classify_query()` /
  `QueryClassification` (plan.md section 14).
- `backend/app/rag/query_rewriting.py` — `rewrite_query()` (sections
  15-17).
- `backend/app/rag/generation/verification.py` — `verify_answer()` /
  `AnswerVerification` (section 26).
- `backend/tests/unit/test_classification.py`,
  `test_query_rewriting.py`, `test_verification.py` — each with a
  minimal fake `LLMClient`, no real model.

Rewritten files:
- `backend/app/graph/state.py` — `GraphState` now carries the full
  pipeline's fields (was just `query`/`response`); added
  `initial_state()` so every key is pre-populated regardless of which
  branch a given run takes.
- `backend/app/graph/workflow.py` — `build_graph()` now takes
  `(llm_client, weaviate_client)` and compiles the real
  classify/general/rewrite/retrieve/generate/verify graph described in
  the "Current State" section above, replacing the old single-node
  placeholder. Retrieval reuses `hybrid_search`/`rerank_chunks` directly
  (not via `RagQueryService`, which raises HTTP-flavored exceptions that
  don't belong inside a graph node) — the small amount of orchestration
  glue is duplicated between the two rather than forcing an awkward
  shared abstraction between fundamentally different call shapes (a
  service that raises vs. a graph node that returns partial state).
- `backend/tests/unit/test_graph.py` — the old test (asserted the
  placeholder echoed its input) is gone; replaced with 4 tests covering
  GENERAL short-circuit, RAG with a supported answer, RAG with an
  unsupported answer (refusal returned), and RAG with no Weaviate client
  (graceful "not found," not a crash) — all via a scripted fake
  `LLMClient` that dispatches a canned reply per distinctive prompt
  substring, plus the same fake-Weaviate pattern `test_rag_service.py`
  uses.

Changed files:
- `backend/app/services/chat_service.py` — **rewritten.** `ChatService`
  no longer takes an `llm_client` directly — it takes the compiled
  `graph` instead, runs `graph.ainvoke(initial_state(request.message))`,
  and persists `result["response"]` as the assistant message. This is
  the actual behavior change: `/api/v1/chat` used to call the LLM once;
  it now runs the whole classify/rewrite/retrieve/generate/verify flow.
- `backend/app/schemas/chat.py` — `ChatResponse` gained `citations:
  list[CitationResponse]` (imported from `app.schemas.rag`, reused
  rather than duplicated), empty for a general reply.
- `backend/app/api/dependencies.py` — `get_chat_service()` now builds
  the graph per request via `build_graph(llm_client, weaviate_client)`
  (same `getattr(request.app.state, ...)` guard as
  `get_rag_query_service`) instead of constructing `ChatService` with a
  bare LLM client.
- `backend/README.md` — rewritten intro + new "The full RAG workflow
  (LangGraph)" section + updated `/api/v1/chat` endpoint docs.

**Part B — auto-ingest on startup:**

New files:
- `backend/app/rag/ingestion/auto_ingest.py` — `run_auto_ingest()`.
  Scans `DOCUMENTS_DIR`, calls the existing `ingest_pdf()` +
  (if not a duplicate and Weaviate is reachable) the existing
  `write_chunks_to_weaviate()` for each PDF found, then moves completed
  files into `INGESTED_DOCUMENTS_DIR`
  (`_unique_destination()` avoids clobbering a same-named file already
  there). Never lets one bad file raise out of the loop — logs and
  leaves it in place for the next run.
- `backend/scripts/auto_ingest.py` — `python -m scripts.auto_ingest`,
  same logic standalone, for testing without running the full server.
- `backend/tests/unit/test_auto_ingest.py` — fake settings (redirect
  `DOCUMENTS_DIR`/`INGESTED_DOCUMENTS_DIR` to `tmp_path`), fake
  `ingest_pdf`/`_embed_document` (monkeypatched — **must be async
  functions**, not plain lambdas, since the real code `await`s them;
  caught this exact bug while writing the tests, see below), real file
  moves on disk. Covers: no documents dir (no-op), successful
  ingest+embed+move, Weaviate unavailable (left in place),
  already-a-duplicate (moved without embedding), `IngestionError` (left
  in place).

Changed files:
- `backend/app/core/config.py` — added `AUTO_INGEST_ON_STARTUP` (`true`),
  `DOCUMENTS_DIR` (`../data/documents`), `INGESTED_DOCUMENTS_DIR`
  (`../data/documents/completed_ingesting`).
- `backend/app/main.py` — `lifespan` now calls `run_auto_ingest()` right
  after the Weaviate client is opened (or set to `None`), wrapped in its
  own try/except so a bug in auto-ingest can't prevent the app from
  serving requests.
- `backend/README.md` — new "Auto-ingest on startup" section.

**Bug caught while writing `test_auto_ingest.py`:** the first draft
monkeypatched `ingest_pdf` with plain (non-`async`) lambdas returning
canned `IngestionResult`s. The real `run_auto_ingest()` does `await
ingest_pdf(...)` — awaiting a non-awaitable raises `TypeError`
immediately, which would have made every one of those tests fail for a
reason unrelated to what they were actually testing. Fixed by wrapping
the fakes in real `async def` functions before running anything.

**Verified this session:** `pytest` → **57 passed** (18 new: 3 fake-LLM
tests each for classification/rewriting/verification, 4 rewritten graph
tests, 5 auto-ingest tests, minus the 1 old placeholder graph test
removed — plus the 39 pre-existing from the OCR entry below, unaffected
except `test_graph.py` which was intentionally replaced). Also fixed
in passing: a `DeprecationWarning` from a stray `\-` (backslash-hyphen,
read as an invalid escape sequence) in `workflow.py`'s module docstring
ASCII diagram.

**NOT verified this session — no real LLM/Weaviate/model call has
exercised any of this new code.** Everything above is wired and passes
unit tests against fakes only. Specifically unconfirmed: that
`classify_query`/`rewrite_query`/`verify_answer`'s actual prompts produce
sensible results from real Groq (the fake `LLMClient` in tests returns
whatever canned string the test wrote, proving the *code* branches
correctly, not that the *prompts* work); that `/api/v1/chat` end-to-end
against a real ingested+embedded document produces a sane classify ->
retrieve -> generate -> verify -> response chain; and that dropping a
real PDF into `data/documents/` and starting the app actually results in
it being ingested, embedded, and moved to `completed_ingesting/`. User
said testing this is a later step, not this session's job — but
whoever picks this up next should not assume "unit tests pass" means
"the real thing works," per this project's established practice of
distinguishing wired-with-fakes from live-verified.

### 2026-08-25 — OCR fallback for scanned PDFs (`--ocr` flag)
**By:** Claude (Sonnet 5), this session.
**Why:** User explicitly asked to add OCR after their real 119-page
`SQL-Manual.pdf` turned out to be fully scanned (no text layer at all —
see the entry below). Explained the real cost (new dependency + a
separately-installed system Tesseract binary + slower ingestion + likely
noisier output on technical content) before starting; user confirmed
"yes" to proceed anyway.

New files:
- `backend/app/rag/ingestion/ocr.py` — `ocr_page_image()` /
  `OcrUnavailableError`. Same lazy-import pattern as
  `app/rag/embeddings.py`: `fitz` (PyMuPDF, pip-only, renders a PDF page
  to an image with no external binary needed) and `pytesseract` (a thin
  wrapper around the *separately installed* Tesseract engine — not a pip
  package) are only imported inside the function. Raises
  `OcrUnavailableError` with a clear message either when the Python
  packages are missing, or when `pytesseract.TesseractNotFoundError` is
  raised (packages installed but the actual Tesseract binary isn't found
  — a distinct, real failure mode this project's other lazy-import
  wrappers don't have, since those only depend on pip packages).
- `backend/requirements/ocr.txt` — `pymupdf==1.24.14`,
  `pytesseract==0.3.13`, `Pillow==11.0.0`. Kept out of `base.txt`/
  `dev.txt`, same reasoning as `embeddings.txt` — most ingests don't need
  this.
- `backend/tests/unit/test_ocr.py` — dependency-missing path forced
  deterministically via monkeypatching `builtins.__import__` (same
  technique `test_embeddings.py` uses), regardless of whether `pymupdf`
  actually happens to be installed here.

Changed files:
- `backend/app/rag/ingestion/pdf_extractor.py` — `extract_pages()` gained
  `ocr_fallback: bool = False`. When true, any page whose direct
  extraction comes back empty (`not text.strip()`) is OCR'd instead of
  left blank. Off by default — doesn't change behavior for any existing
  caller.
- `backend/app/rag/ingestion/pipeline.py` — `ingest_pdf()` gained
  `ocr_fallback: bool = False`, passed straight through to
  `extract_pages()`. Also now catches `OcrUnavailableError` alongside
  `PdfExtractionError` and wraps it as the existing `IngestionError`, so
  a missing OCR dependency surfaces as the same clean
  `"Ingestion failed: ..."` CLI message as every other ingestion failure,
  not a raw traceback.
- `backend/scripts/ingest_document.py` — added `--ocr` flag.
- `backend/app/core/config.py` — added `OCR_DPI` (300) and
  `TESSERACT_CMD` (`None` — only needed if the binary isn't on `PATH`,
  common on Windows).
- `backend/tests/unit/test_ingestion.py` — two new tests build a real
  *blank* PDF via `pypdf.PdfWriter` (no new dependency needed — a blank
  page has no text layer, exactly like the real scanned document that
  motivated this) and confirm: without `--ocr` the blank page's text
  stays `""`; with it, a monkeypatched `ocr_page_image` gets called
  exactly once with the right page number and its return value is used.
  This tests `extract_pages()`'s branching logic without needing a real
  Tesseract install.
- `backend/README.md` — new "Ingest a scanned PDF (OCR fallback)"
  section: the `--ocr` flag, the `requirements/ocr.txt` install, and —
  important, since this is not pip-installable — the separate Tesseract
  engine install for Windows/macOS/Linux, plus `TESSERACT_CMD` for when
  it's not on `PATH`.

**Verified this session:** `pytest` → **39 passed** (5 new: 1 OCR
dependency-missing test + 2 new `extract_pages` OCR-branching tests,
plus the 4 `rag_service` tests from the entry above — 33 pre-existing,
unaffected).

**NOT verified this session — no real OCR run has happened yet.**
`requirements/ocr.txt` is not installed in this environment, and neither
is the actual Tesseract engine binary. This is wired and unit-tested with
fakes only, same "wired, not yet exercised" status the embeddings work
had before the user installed `torch` and ran a real BGE-M3 embed.
**Next step for whoever picks this up:** user installs
`requirements/ocr.txt` + the Tesseract binary (Windows: UB-Mannheim's
build, see README), then runs
`python -m scripts.ingest_document ..\data\documents\SQL-Manual.pdf --ocr`
against the real 119-page scanned manual — this is the first real
Tesseract run for this project. Expect it to be slow (119 pages, each
rendered + OCR'd) and possibly noisy (technical content — SQL syntax,
tables, code blocks — is a harder case for OCR than plain prose). If it
works, continue on to `scripts.embed_document` and finally the
still-unverified `/api/v1/rag/ask` live test from the entry below.

### 2026-08-25 — Real-world finding: scanned PDFs produce zero chunks (expected, not a bug)
**By:** Claude (Sonnet 5), this session.
**Why:** User tried ingesting their own real document
(`data/documents/SQL-Manual.pdf`, 119 pages) to verify the new
`/api/v1/rag/ask` endpoint end to end. `scripts.ingest_document` reported
`pages=119 chunks=0` — confirmed via the written JSON (`"chunks": []`)
and directly via `pypdf` (`page.extract_text()` returns `''` for every
page checked). This PDF has no real text layer — it's scanned/image-based
— so there is nothing for the extractor to chunk. **This is the OCR gap
already logged as explicitly deferred** (`plan.md`: "OCR is NOT part of
V1... do not introduce unless explicitly requested"), not a defect in
`pdf_extractor.py`/`chunker.py`. Worth recording because it's the first
time this project has hit that limitation against a real user document
rather than a synthetic text-based test PDF — earlier verifications
(leave-policy/kitchen test PDFs) all had real text layers, so this gap
was never exercised until now.

Cleanup performed (the ingestion had already written a `Document`/
`DocumentVersion` row and an empty processed JSON before the zero-chunks
outcome was noticed): deleted the `Document` row directly via SQL
(cascaded to its `DocumentVersion`), removed the empty
`data/processed/<id>/` folder. Nothing usable was lost — there were zero
chunks to begin with.

**Status: open decision, not yet made.** Two ways forward were offered
to the user and neither has been chosen yet:
1. Use a different, born-digital (real text layer) PDF to verify
   `/api/v1/rag/ask` end to end instead.
2. Add OCR to the ingestion pipeline — a real scope expansion beyond
   plan.md's stated V1 list, not something to add without deciding to.

**Next step for whoever picks this up:** get a decision on the above,
then complete the still-outstanding live verification of
`/api/v1/rag/ask` (see the entry directly below — the endpoint is wired
and unit-tested but has not yet answered a real question end to end over
HTTP).

### 2026-08-25 — HTTP endpoint for the RAG pipeline: `POST /api/v1/rag/ask`
**By:** Claude (Sonnet 5), this session (new account, per the prior
handoff — read this file first as instructed). User reviewed the whole
prior session's work, confirmed everything logged, then chose "HTTP
endpoint first" over "query classification/rewriting/verification" when
asked directly (both had been left as the two open options). Scoped
narrowly: wrap the already-verified CLI pipeline (`scripts.ask`) in an
endpoint, still no classification/rewriting/verification — same scope
`scripts.ask` itself has, just reachable over HTTP now.

New files:
- `backend/app/services/rag_service.py` — `RagQueryService.ask()`.
  Mirrors `scripts/ask.py`'s `_run()` logic exactly (hybrid search →
  optional rerank → `generate_answer()`), but raises the project's
  existing `AppException` subclasses instead of printing to stderr:
  `NotFoundError` (404) if the `DocumentChunk` collection doesn't exist
  yet (nothing ingested/embedded), `ServiceUnavailableError` (503) if the
  Weaviate client is `None` (failed to connect at startup) or if
  `EmbeddingModelUnavailableError` is raised (embedding/reranking deps
  not installed) — reusing `app/core/exceptions.py`'s already-registered
  handler rather than inventing a new error shape.
- `backend/app/schemas/rag.py` — `RagAskRequest`/`RagAskResponse`/
  `CitationResponse`. Field names/defaults mirror `scripts.ask`'s CLI
  flags (`limit`, `alpha`, `rerank`, `top_k`) so the two interfaces stay
  in sync conceptually.
- `backend/app/api/routes/rag.py` — `POST /api/v1/rag/ask`. Route itself
  has no branching logic, just calls `RagQueryService.ask()` and maps the
  dataclass result onto the response schema — consistent with the
  project's "no business logic in routes" rule.
- `backend/tests/unit/test_rag_service.py` — fake Weaviate client (same
  `_FakeCollection`/`_FakeObject` shape as `test_hybrid_search.py`, not
  imported from there — kept self-contained rather than cross-importing
  between test modules), fake embedder/reranker via `monkeypatch.setattr`
  on `app.services.rag_service.get_embedder`/`get_reranker`, fake LLM
  client. Covers: 404 when collection missing, 503 when
  `weaviate_client=None`, a normal answer+citations path, that
  `do_rerank=False` never touches the reranker at all (asserts via a
  monkeypatched function that raises if called), and the zero-chunks
  "unanswerable" path. **Caught and fixed while writing these**: the
  zero-chunks test initially didn't monkeypatch `get_embedder` — but
  `hybrid_search()` calls `embed_query_fn` to build the search request
  regardless of how many results come back, so that test would have
  quietly tried to load the real BGE-M3 model instead of testing the
  intended code path.

Changed files:
- `backend/app/api/dependencies.py` — added `get_rag_query_service()`,
  reading the shared client via `getattr(request.app.state,
  "weaviate_client", None)` (defensive: falls back to `None` — which
  `RagQueryService` already turns into a clean 503 — rather than a raw
  `AttributeError` if some ASGI test harness never runs the lifespan).
- `backend/app/main.py` — `lifespan` now opens one `weaviate.WeaviateClient`
  at startup (`get_weaviate_client()`) and stores it on
  `app.state.weaviate_client`, closing it at shutdown. Wrapped in
  try/except: if Weaviate is unreachable at boot, logs and sets `None`
  instead of crashing the whole app — matches the existing philosophy
  (`/health` already degrades rather than crashes when a dependency is
  down) rather than making the entire API unavailable just because
  Weaviate happens to be down. Registered the new `rag` router.
- `backend/README.md` — documented the new endpoint (request/response
  shape, status codes, the "client opened once at startup" design note).

**Why a shared client instead of the CLI's open-per-call pattern:** the
CLI scripts open a `WeaviateClient`, do one thing, and close it — fine
for a short-lived process. A long-lived FastAPI server handling many
requests would pay a new HTTP+gRPC handshake on every single call if it
did the same thing; opening once at startup and reusing it is the
standard pattern for this kind of client in a server process.

**NOT verified this session — no live HTTP request has been made yet.**
Only unit tests (with fakes) have run. Specifically unverified against
the real stack: that `app.state.weaviate_client` actually gets set
correctly by the real lifespan under real uvicorn startup, that the 404/
503 paths return the right HTTP status when hit for real, and that a
real question against a real ingested+embedded document returns a real
grounded answer with correct citations over HTTP (the equivalent of the
CLI verification already done for `scripts.ask` in an earlier entry, but
via `POST /api/v1/rag/ask` instead of the command line). **Next step for
whoever picks this up:** start the backend, ingest+embed a throwaway test
PDF (same pattern as prior sessions' live verifications — clean up
after), and hit the endpoint for real before considering this "done"
rather than just "wired."

### 2026-08-25 — Session handoff: switching accounts (2nd time)
**Note for the next session/account:** the user is switching Claude
accounts again. Not a blocker — the baseline RAG pipeline is complete
and verified end to end (see the entry directly below and the several
before it). Read this whole file before doing anything else; skip
straight to "Current State" for the authoritative snapshot.

**One-line summary of everything done since the last handoff:** Groq
wired + verified live → PDF ingestion (extract/clean/chunk) verified
live → BGE-M3 embeddings + Weaviate write verified live (real model,
not mocked) → hybrid search (Stage 1) verified live → reranking
(BGE-Reranker-v2-M3, Stage 2) verified live → context construction +
grounded Groq answer generation with citations verified live. Every
one of those was tested against real running Postgres/Weaviate/Groq/
BGE-M3/reranker — not mocked-only — with test artifacts cleaned up
after each.

**Immediate open question, not yet decided:** what to build next. Two
options were offered and neither has been started:
1. Wrap the CLI pipeline (`scripts.ask`) in a real `/api/v1/*` HTTP
   endpoint — there is currently NO HTTP way to run RAG search/answer,
   only the CLI scripts (`scripts.ingest_document`,
   `scripts.embed_document`, `scripts.search`, `scripts.ask`).
2. Query classification / query rewriting / answer verification
   (plan.md sections 14, 15, 26) — still entirely unimplemented.

**Known minor/cosmetic issues, not yet fixed, not urgent:**
- `asyncpg` `ResourceWarning: unclosed connection` shows up at the end
  of several CLI script runs (`embed_document`, `ask`) — the async
  engine/session isn't explicitly disposed before process exit. Cosmetic
  only, doesn't affect correctness.
- Similarly, `GroqLLMClient`'s underlying `httpx` socket triggers an
  `unclosed socket` `ResourceWarning` at the end of `scripts.ask` runs —
  same root cause (no explicit `.close()`/`.aclose()` before exit).
- `weaviate-client` is pinned at 4.9.6 (latest is 4.23.0+) — this was a
  deliberate pin from the original foundation-setup entry, for an
  `httpx` version compatibility reason at the time. Worth revisiting if
  it ever blocks something, not proactively.
- Windows console encoding crashes on non-ASCII LLM output were fixed
  (`scripts/_console.py`, see the entry directly below) — but this fix
  is CLI-scripts-only. If an HTTP endpoint is added next (option 1
  above), FastAPI/uvicorn's own response encoding needs no equivalent
  fix (HTTP responses are UTF-8 over the wire regardless of the
  server's console codepage) — noting this so nobody "fixes" a
  non-problem there by copying the CLI pattern unnecessarily.

See `../../plan.md` for the full long-term spec.

### 2026-08-25 — Grounded answer generation (context + Groq + citations), verified live
**By:** Claude (Sonnet 5), this session.
**Why:** User said "yes" to wiring retrieval → Groq → answer next,
continuing directly from reranking. Scoped to plan.md sections 20/23-25
(context construction, LLM generation, grounded-answer policy,
citations) only — section 14/15 (query classification/rewriting) and
section 26 (answer verification) explicitly stay out, consistent with
the project's phase-by-phase approach; this assumes the caller already
decided RAG applies and already has a final chunk list.

New files:
- `backend/app/rag/generation/context_builder.py` — `build_context()` /
  `build_citations()` / `Citation`. Both take plain `RetrievedChunk`s
  (works whether or not reranking ran upstream — a reranked list is
  just `[r.chunk for r in reranked]`). Citations are built ONLY from
  retrieved metadata, by design — plan.md section 25 explicitly says
  "Do NOT hallucinate page numbers," so nothing here ever parses a
  citation out of LLM output. Dedup by `chunk_id` is a defensive
  backstop, not a real dedup pass (chunks should already be unique
  post-retrieval).
- `backend/app/rag/generation/answer_generator.py` — `generate_answer()`
  / `RAGAnswer`. Composes plan.md section 24's grounded-answer
  instructions + the context block + the question into ONE prompt
  string and sends it through the existing `LLMClient.generate_reply()`
  — **no change to that abstraction was needed**, it already accepts
  an arbitrary message string. Skips the LLM call entirely and returns
  a fixed "not found" `RAGAnswer` when given zero chunks (nothing to
  ground an answer in) — distinct from the LLM itself declining despite
  having context, which is a different, also-observed-live code path
  (see verification below).
- `backend/scripts/ask.py` — CLI: `python -m scripts.ask "question"
  [--rerank] [--top-k 10]`. Runs Stage 1 → optional Stage 2 → context →
  answer, end to end.
- `backend/tests/unit/test_context_builder.py`,
  `backend/tests/unit/test_answer_generator.py` — fake chunks / fake
  LLM client: labeling, content inclusion, dedup, empty input; prompt
  contains context+query, citations match retrieved metadata, empty-
  chunks path never calls the LLM at all.
- `backend/scripts/_console.py` — **new, unrelated-to-RAG fix, see
  below.**

**A real bug found and fixed while verifying live:**
`scripts.ask` crashed with `UnicodeEncodeError: 'charmap' codec can't
encode character '【'` — Groq's real answer contained a Unicode
character (a CJK-style citation bracket, `【`), and this Windows
PowerShell's console defaults to `cp1252`, which can't encode it, so
`print()` itself crashed the script. Root cause is generic (any
non-ASCII LLM output — curly quotes, em-dashes, etc. — could trigger
the same crash in `ingest_document`/`embed_document`/`search` too, not
just `ask`), so fixed once, shared: `scripts/_console.py` →
`fix_windows_console_encoding()` reconfigures stdout/stderr to UTF-8
with `errors="replace"`, called at the top of all four CLI scripts
(`ask.py`, `search.py`, `embed_document.py`, `ingest_document.py`)
before any other import. This was a genuine crash risk in already-
shipped scripts, not just new code — worth remembering if a future
session adds another CLI script that prints LLM/user-supplied text on
Windows.

**Verified this session:**
- `pytest` → **31 passed** (7 new: context builder + answer generator
  tests; 24 pre-existing, unaffected).
- **Live, real Groq, real retrieved context** (throwaway test PDF,
  ingested + embedded + cleaned up after, same pattern as prior
  entries): `scripts.ask "How many days of annual leave do employees
  get?" --rerank --top-k 3` → real answer
  `"Employees receive **twenty days** of paid annual leave per
  calendar year [Source 1]."` with a matching `Sources:` block citing
  the correct document name and page — this is the fix above being
  proven, since the first attempt at this exact call is what crashed.
- **Live, grounded-refusal path**: asked an unrelated question ("What
  is the capital of France?") against the same (irrelevant) retrieved
  context — Groq correctly responded that the information could not
  be found in the provided context, rather than answering from its own
  general knowledge. Confirms plan.md section 24's "do not use external
  knowledge" instruction is actually being honored by the real model,
  not just present in the prompt text.
- Cleaned up afterward: deleted the Weaviate object (confirmed
  collection count back to 0), the `Document` row, the test PDF and
  processed JSON, uninstalled the temporary `reportlab`.

**Status:** the full baseline RAG pipeline (plan.md's "Final V1
Pipeline" diagram, minus query classification/rewriting and answer
verification) now works end to end via `scripts.ask`. Still CLI-only —
no `/api/v1/*` HTTP endpoint wraps any of this yet.

### 2026-08-25 — Reranking (Stage 2, BGE-Reranker-v2-M3), verified live
**By:** Claude (Sonnet 5), this session; final live run executed by the
user (asked to run commands themselves to save tokens partway through —
handed exact commands instead of continuing to run/watch it myself).
**Why:** Continuing directly after Stage 1 hybrid search — reranking is
the very next step in plan.md's pipeline (section 19, Top-30 → Top-10).

New files:
- `backend/app/rag/reranking.py` — `Reranker` / `get_reranker()`. Same
  lazy-import pattern as `embeddings.py`: `sentence_transformers` only
  imported inside `_load_model()`. Uses `CrossEncoder`, not the same
  class as the embedding model — reuses the *same*
  `requirements/embeddings.txt` install (no new optional-deps file
  needed, `sentence-transformers` already covers both).
- `backend/app/rag/retrieval/rerank.py` — `rerank_chunks()` /
  `RerankedChunk`. `score_fn` injected, same testability pattern as
  `hybrid_search.py`. `score_threshold` (optional, unset by default)
  can drop below-threshold results even below `top_k` — deliberately
  NOT forcing exactly `top_k` results, per plan.md section 18's explicit
  instruction.
- `backend/tests/unit/test_rerank.py` — fake score function: sorts
  descending correctly, caps at `top_k`, threshold filtering doesn't
  force the count back up, empty input short-circuits, score/candidate
  count mismatch raises.

Changed files:
- `backend/app/core/config.py` — added `RERANKER_MODEL_NAME` (default
  `BAAI/bge-reranker-v2-m3`), `RERANK_TOP_K` (10), `RERANK_SCORE_THRESHOLD`
  (`None` — no evaluation set yet to derive a real one from).
- `backend/app/rag/embeddings.py` — added `Embedder.embed_query()`, a
  one-line convenience wrapper for the single-query case.
- `backend/scripts/search.py` — added `--rerank` / `--top-k` flags. When
  set, runs Stage 1 then feeds its candidates through Stage 2; prints
  both `hybrid_score` and `rerank_score` per result.
- `backend/README.md` — documented `--rerank`/`--top-k`.

**Verified this session — with real hiccups worth recording:**
- `pytest` → **24 passed** (5 new rerank tests; 19 pre-existing,
  unaffected).
- First live attempt: backgrounded a `scripts.search --rerank` run
  (expecting a multi-GB reranker download) and monitored it. A
  transient `Read timed out` mid-download was misread by my own
  monitor's match pattern as a fatal error (it matched the literal
  word "Error" in "Error while downloading..." — a monitor-script bug,
  not a real failure) and I killed the process to hand off to the user
  — but `kill` on the wrapper left an orphaned child process (a second
  PID) still holding the file, which in turn left a stale
  `huggingface_hub` `.lock` file behind after the user's own
  `scripts.search --rerank` run got stuck on `WeakFileLock` waiting for
  it. Found and killed the actual orphaned child PID, removed the stale
  lock file (`~/.cache/huggingface/hub/.locks/models--BAAI--bge-reranker-v2-m3/*.lock`),
  confirmed no process still held it, then had the user retry.
- **User's retry succeeded fully**: resumed the partial (~1.7GB already
  fetched) `model.safetensors` download to completion (2.27GB total),
  pulled tokenizer files, then printed
  `[1] rerank_score=0.9900 (hybrid_score=1.0000) sample_policy.pdf ...`
  — correct, high-confidence relevance score on the one seeded test
  chunk. This is the **first real BGE-Reranker-v2-M3 run for this
  project**, not a mock.
- Cleaned up afterward: deleted the test document's Weaviate object
  (confirmed collection count back to 0), its `Document` row, the test
  PDF and processed JSON, and uninstalled the temporary `reportlab`.

**Lesson for next time (noted so it doesn't repeat):** when
backgrounding a long model download via Monitor, don't grep for the
bare word "Error" — huggingface_hub's own retry/resume logging contains
that word for conditions it recovers from automatically. And when a
backgrounded process needs to be stopped, verify no child process
survived the kill before treating it as fully stopped, especially
before touching any lock/cache files it might still hold.

**Status: RESOLVED.** Both retrieval stages (hybrid search + reranking)
are now fully verified live, real models, real data, matching plan.md
sections 12 and 19 for the baseline (untuned `alpha`/`score_threshold`
are known-open items, not bugs — no evaluation set exists yet to tune
them against, per plan.md section 39's own instruction to defer that
kind of tuning).

### 2026-08-25 — Hybrid search (Stage 1), verified live with real BGE-M3 + Weaviate
**By:** Claude (Sonnet 5), this session.
**Why:** User asked to continue with retrieval next. Scoped to plan.md
section 18's Stage 1 only (hybrid dense+BM25 search for a high-recall
Top-30 candidate set) — Stage 2 (BGE-Reranker-v2-M3 reranking down to
Top-10) is explicitly a separate later step, not built here, staying
consistent with the project's phase-by-phase approach.

New files:
- `backend/app/rag/retrieval/hybrid_search.py` — `hybrid_search()` +
  `RetrievedChunk`. Embeds the query text via the same `Embedder`
  (BGE-M3) used for ingestion, then calls Weaviate's native `hybrid()`
  query (dense vector + its own BM25, fused server-side) — no BM25
  implemented separately, Weaviate does it. `collection` and
  `embed_query_fn` are injected (not constructed inside), same pattern
  as `vector_writer.py`, so this is unit-testable without a live
  Weaviate/model. `alpha` (default 0.5) is an untuned neutral midpoint
  between BM25-only (0.0) and vector-only (1.0) — plan.md section 39
  lists this as something to A/B once an evaluation set exists, which
  it doesn't yet.
- `backend/scripts/search.py` — CLI: `python -m scripts.search "query"
  [--limit 30] [--alpha 0.5]`. Prints ranked results with scores and
  page-span citation metadata. Errors clearly if the `DocumentChunk`
  collection doesn't exist yet (nothing embedded) rather than a raw
  Weaviate exception.
- `backend/tests/unit/test_hybrid_search.py` — fake collection + fake
  query embedder: results map through in score order, query text/vector/
  limit/alpha are passed to Weaviate correctly, empty/whitespace query
  short-circuits without calling Weaviate at all, no-results case
  returns `[]` cleanly.

Changed files:
- `backend/app/rag/embeddings.py` — added `Embedder.embed_query()`, a
  one-line convenience wrapper around `embed_texts([text])[0]` for the
  single-query case `hybrid_search` needs.

**Verified this session:**
- `pytest` → **19 passed** (4 new hybrid-search tests; 15 pre-existing,
  unaffected).
- **Live, with the real BGE-M3 model and real Weaviate** (both already
  installed/verified in the prior entry): ingested + embedded two
  distinct throwaway test PDFs — a leave-policy doc and an unrelated
  office-kitchen doc (both via `reportlab`, installed temporarily,
  uninstalled after; not added to any requirements file) — then ran
  `scripts.search` with two different queries. Confirmed **actual
  relevance discrimination, not just "returns what's in the DB"**: the
  leave-policy query scored the leave-policy chunk 1.0 and the kitchen
  chunk 0.0, and the coffee-machine query scored the opposite way
  around, each time correctly ranking the relevant document first.
- Cleaned up afterward: deleted both Weaviate objects (confirmed
  collection count back to 0 via `aggregate.over_all()`), both
  `Document` rows, both test PDFs and their processed JSON, and
  uninstalled the temporary `reportlab`. (First cleanup attempt at the
  Weaviate step failed silently-ish with `ModuleNotFoundError: No
  module named 'app'` because the command was run from the repo root
  instead of `backend/` — caught immediately via the aggregate count
  check, redone correctly from the right directory.)

**Not done in this entry, deliberately deferred:** reranking
(BGE-Reranker-v2-M3), query classification (general vs RAG-required),
query rewriting, metadata/ACL filtering (the `filters` param exists on
`hybrid_search()` but nothing populates tenant/ACL properties — no
auth exists yet), context construction for an LLM prompt, citations in
a generated answer, answer verification. This CLI only prints raw
ranked chunks — there is still no endpoint or code path that sends
retrieved context to Groq and generates an answer.

### 2026-08-25 — Verified: real BGE-M3 embed run, end to end
**By:** Claude (Sonnet 5), this session.
**Why:** Closing out the "not verified — no real BGE-M3 run" gap from
the entry below. User hit a Windows path-length limit installing
`requirements/embeddings.txt` (`torch`'s wheel has deeply nested license
file paths that overflowed `MAX_PATH` combined with this project's
already-long path) — gave them the `LongPathsEnabled` registry fix
(admin PowerShell, no reboot needed, just a fresh non-admin terminal
afterward) rather than suggesting they relocate the whole repo. They ran
it and confirmed `torch==2.13.0+cpu` and `sentence-transformers==3.3.1`
installed successfully.

No source changes except one test fix (below). Verification steps:
- Generated a throwaway 2-page test PDF (`reportlab`, installed
  temporarily, uninstalled after — not added to any requirements file).
- `scripts.ingest_document` → 2 pages, 4 chunks, as before.
- `scripts.embed_document <document_id>` (backgrounded and watched via
  a monitor, since a from-scratch BGE-M3 download can take 15-30 min):
  completed with `Wrote 4 chunks to Weaviate collection 'DocumentChunk'`
  — this is the **first real BGE-M3 model load and embed for this
  project**, not a mock.
- Queried Weaviate directly afterward with `include_vector=True`: all 4
  objects present, each with a **1024-dimension** vector (matches
  `EMBEDDING_DIMENSION` in config) and correct `content`/`chunk_index`.
- Cleaned up all test artifacts: deleted the 4 Weaviate objects, the
  test PDF, its processed JSON, the `Document`/`DocumentVersion` rows,
  and uninstalled the temporary `reportlab`.

**Test fix required:** `tests/unit/test_embeddings.py`'s
`test_embed_texts_raises_clear_error_when_dependency_missing` had
assumed `sentence-transformers` would never be installed in the test
environment (true when it was written, no longer true after the
install above) — it started failing for the right reason (the real
library is now present, so the lazy import no longer fails). Rewrote it
to monkeypatch `builtins.__import__` so it forces the "dependency
missing" path deterministically regardless of what's actually installed,
instead of relying on environment state. `pytest` → **15 passed** again
after the fix.

**Status: RESOLVED.** The embeddings + Weaviate write path (previous
entry) is now fully verified live, not just wired-and-mocked. This
closes the last open item from that entry.

### 2026-08-25 — Wire embeddings + Weaviate write (BGE-M3 not yet installed)
**By:** Claude (Sonnet 5), this session (new session — found this file
relocated to `docs/PROJECT_LOG.md` by unlogged repo cleanup; fixed the
now-stale `../plan.md` references to `../../plan.md` in the same pass,
see "Current State" header note).
**Why:** User asked to continue with "wire this pipeline's chunk output
into Weaviate with BGE-M3 embeddings" (the logged next step after PDF
ingestion). Before installing anything, flagged that real BGE-M3 needs
`torch` + `sentence-transformers` and a ~2.3GB download (15-30 min) —
asked the user how to proceed rather than assuming; they chose **"wire
it, don't download yet"** (write + unit-test all the code, skip the
actual model load/live embed for now).

New files:
- `backend/app/rag/embeddings.py` — `Embedder` / `get_embedder()`.
  `sentence_transformers` is imported lazily inside `_load_model()`,
  not at module import time, so this (and everything that imports it)
  works fine with the dependency absent. Raises
  `EmbeddingModelUnavailableError` with a clear message if the library
  isn't installed when actually asked to embed non-empty input.
- `backend/app/rag/ingestion/vector_writer.py` — `write_chunks_to_weaviate()`.
  Takes an injected `collection` and `embed_texts_fn` (not constructed
  internally) specifically so it's unit-testable without a real Weaviate
  connection or a real model. Deterministic per-chunk UUID
  (`document_id:version:chunk_index` via `generate_uuid5`) so re-running
  upserts instead of duplicating.
- `backend/scripts/embed_document.py` — CLI:
  `python -m scripts.embed_document <document_id> [--version N]`. Reads
  the JSON chunks written by `scripts.ingest_document`, embeds them, and
  writes them into Weaviate. Kept as its own script (not folded into
  `scripts.ingest_document`) so the already-verified extract/clean/chunk
  path stays independent of Weaviate/embedding availability.
- `backend/requirements/embeddings.txt` — **new, separate file.**
  `sentence-transformers==3.3.1`, deliberately NOT referenced from
  `base.txt`/`dev.txt` so a routine `pip install -r requirements/dev.txt`
  doesn't silently pull in `torch`. Installed on purpose when someone's
  ready to run real embeddings.
- `backend/tests/unit/test_embeddings.py` — empty-input short-circuits
  without touching the model; non-empty input raises
  `EmbeddingModelUnavailableError` cleanly (proves the lazy-import path
  fails legibly given `sentence-transformers` genuinely isn't installed
  here).
- `backend/tests/unit/test_vector_writer.py` — fake collection + fake
  embed function: correct count written, empty-input no-op, deterministic
  UUID stability across two separate calls (the upsert property),
  vector/chunk count mismatch raises, Weaviate-reported errors raise.

Changed files:
- `backend/app/database/vector_store.py` — added `CHUNK_COLLECTION_NAME`
  and `ensure_chunk_collection()`: creates the `DocumentChunk` collection
  (`vectorizer: none` — vectors are supplied by us) with properties per
  plan.md section 4.1. `section`, `parent_id`, `tenant_id`,
  `document_type`, `access_level` are in the schema for forward
  compatibility but NOT populated by anything yet (no contextual
  chunking/ACL/tenant work exists) — noted in the function's docstring
  so it isn't mistaken for a completed feature later.
- `backend/app/repositories/document_repository.py` — added `get_by_id`,
  `get_version`, `get_active_version` (needed by
  `scripts.embed_document` to find a document's chunk JSON on disk from
  just its id).
- `backend/app/core/config.py` — added `EMBEDDING_MODEL_NAME` (default
  `BAAI/bge-m3`), `EMBEDDING_DIMENSION` (1024).
- `backend/README.md` — documented the embed step, the separate
  `embeddings.txt` install, and exactly what has/hasn't been verified.

**Verified this session:**
- `pytest` → **15 passed** (7 new: embeddings + vector_writer tests;
  8 pre-existing, unaffected).
- **Live, against the real running Weaviate container** (no model
  needed for this part): `ensure_chunk_collection()` actually created
  the `DocumentChunk` collection — confirmed via
  `client.collections.exists()` and reading back its property list
  (all 13 properties present, matching the schema).
- **Live, against the real running Weaviate container**, using a fake
  embedding function (`lambda texts: [[0.01*i]*1024 ...]`) in place of
  BGE-M3: wrote 2 chunks via `write_chunks_to_weaviate()`, re-wrote the
  *same* 2 chunks and confirmed the collection still held exactly 2
  objects (upsert, not duplicate — the deterministic-UUID design works
  against a real server, not just in the mocked unit test), fetched them
  back and checked their content matched, then deleted both to leave the
  collection clean.

**NOT verified — explicitly deferred per the user's choice above:** no
real BGE-M3 model has been loaded or run by anyone yet.
`requirements/embeddings.txt` has not been installed in this
environment. The actual embed step
(`Embedder._load_model()` → `sentence_transformers.SentenceTransformer("BAAI/bge-m3")`)
is completely unexercised — everything verified above used a fake
embedding function standing in for it. **Next session/account, if
picking this up:** `pip install -r requirements/embeddings.txt` (large,
pulls in torch), then `python -m scripts.embed_document <document_id>`
against a document already ingested via `scripts.ingest_document` — this
is the first real BGE-M3 download/run for the project, budget the
15-30 min for it.

### 2026-08-25 — PDF ingestion pipeline (extraction -> cleaning -> chunking)
**By:** Claude (Sonnet 5), this session.
**Why:** User picked this over further Groq work as the next step (the
other of the two logged options after Groq was verified). Read the full
`plan.md` first, per the user's request, before implementing — this
entry implements only the baseline slice explicitly scoped in the prior
handoff note: extraction, cleaning, fixed-size chunking at ~600/100
tokens. Not the full plan.md V1 list (no embeddings/Weaviate/table
extraction/contextual chunking/parent-child yet) — staying phase-by-phase
per the project's stated approach.

New files:
- `backend/app/rag/ingestion/hashing.py` — `hash_file()`, sha256 over
  raw file bytes, for content-based dedup (plan.md section 27).
- `backend/app/rag/ingestion/pdf_extractor.py` — `extract_pages()` via
  `pypdf`, per-page plain text only (no tables — plan.md section 7 lists
  table extraction as V1 but it's out of scope for this slice). Raises
  `PdfExtractionError` on encrypted/corrupt PDFs.
- `backend/app/rag/ingestion/text_cleaner.py` — `clean_text()`,
  whitespace normalization only. Deliberately NOT doing header/footer
  detection (needs cross-page frequency analysis to do well; left as a
  follow-up rather than guessed at).
- `backend/app/rag/ingestion/chunker.py` — `chunk_pages()`, fixed-size
  token windows with overlap via `tiktoken`'s `cl100k_base` encoding
  (chosen as a model-agnostic approximation — it will NOT exactly match
  BGE-M3's own tokenizer once that's wired in later; revisit then).
  Tracks `start_page`/`end_page` per chunk by tagging each token with
  its source page before windowing.
- `backend/app/rag/ingestion/pipeline.py` — `ingest_pdf()` orchestrates
  validate → hash → dedup-check → extract → clean → chunk → write JSON
  → persist `Document`/`DocumentVersion` → commit. Dedup is a no-op by
  default when the exact file content was already ingested;
  `force_reprocess=True` creates a new version instead.
- `backend/app/repositories/document_repository.py` — `DocumentRepository`
  (`get_by_hash`, `create_document`, `next_version_number`,
  `create_version` — the latter deactivates any previously-active
  version before inserting the new one). Documents the schema
  interpretation used (see "Current State" above) since
  `Document`/`DocumentVersion` were created in an earlier session without
  this pipeline to exercise them against.
- `backend/scripts/ingest_document.py` — CLI entry point:
  `python -m scripts.ingest_document path/to/file.pdf [--force]`.
  Terminal/CLI interface, per plan.md section 1 ("initial interface will
  be TERMINAL/CLI only").
- `backend/tests/unit/test_ingestion.py` — unit tests for `clean_text()`
  and `chunk_pages()` (pure functions, no PDF file or DB needed):
  whitespace collapsing, chunk size/overlap correctness, page-span
  tracking across a page boundary, empty input, and the
  overlap-must-be-smaller-than-size guard.

Config additions (`backend/app/core/config.py`): `PROCESSED_DATA_DIR`
(default `../data/processed`), `CHUNK_SIZE_TOKENS` (600),
`CHUNK_OVERLAP_TOKENS` (100).

Dependencies added (`backend/requirements/base.txt`): `pypdf==5.1.0`,
`tiktoken==0.8.0`. Checked for conflicts with existing pins before
installing — none found.

**Verified this session (live, against real Postgres):**
- `pip install -r requirements/dev.txt` clean; `pytest` → **8 passed**
  (5 new ingestion tests + the 3 pre-existing).
- Generated a throwaway 3-page test PDF (via `reportlab`, installed
  temporarily and uninstalled after — NOT added to any requirements
  file) and ran it through the real CLI against the running `postgres`
  container:
  - First run: created a new `Document` + `DocumentVersion` (version 1,
    `is_active=true`), 3 pages → 9 chunks, JSON written to
    `data/processed/<id>/v1.json`. Inspected the JSON — chunk token
    counts, 500-token stride (600-100 overlap), and page-span metadata
    all correct.
  - Second run (same file, no `--force`): correctly detected as a
    duplicate via `content_hash`, logged "already ingested", made no
    new DB writes (confirmed via the SQL echo log — `SELECT` then
    `ROLLBACK`, no `INSERT`).
  - Third run (`--force`): created version 2 (`is_active=true`) and
    confirmed via direct `psql` query that version 1 was flipped to
    `is_active=false` — exactly the intended versioning behavior.
  - Verified error handling: a non-PDF file and a nonexistent path both
    produced a clean `Ingestion failed: ...` message and a non-zero
    exit, not a stack trace.
  - Cleaned up all test artifacts afterward: deleted the generated PDF,
    its processed JSON output, and the test `Document` row (cascade-
    deleted its `DocumentVersion` rows) — nothing from this test run
    was left in the real data/database.

**Not done in this entry, deliberately deferred to later phases per
plan.md's explicit ordering:** embeddings (BGE-M3), Weaviate storage,
table extraction, header/footer detection, contextual chunking,
parent-child structure, background/async ingestion (`upload_jobs` still
unused).

### 2026-08-25 — Verified: live Groq call works end to end
**By:** Claude (Sonnet 5), this session.
**Why:** Closing out the "not verified — no live Groq call" gap from the
previous entry. User added their own `GROQ_API_KEY` to `backend/.env`
(not shared with/typed by the assistant) and asked how to run + test it;
ran the verification directly instead of just giving instructions.

No code changes. Verification steps:
- Confirmed `GROQ_API_KEY` is non-empty in `.env` without printing the
  value (`grep -c` + masked `sed`).
- Docker containers (`postgres`, `weaviate`) already running from
  earlier sessions.
- Started the backend; `GET /health` → `"status":"ok"`, both `database`
  and `vector_store` connected.
- `POST /api/v1/chat` with a real question → got back a genuine
  Groq-generated reply (not the `"Echo: ..."` stub), with a new session
  UUID.
- Confirmed directly in Postgres (`SELECT ... FROM messages`) that both
  the user message and the real Groq reply were persisted correctly,
  UTF-8 intact (smart quotes/em-dash stored fine).

**Noted, not a bug:** the uvicorn log line for that message showed `�`
in place of `’`/`—` — this is a Windows-console-codepage display glitch
in the log *stream* (`app/core/logging.py`'s `StreamHandler`), not data
corruption; the same characters are stored correctly in Postgres. Not
fixed in this entry since it's cosmetic (log readability on Windows
only) and out of scope for this task — worth a `StreamHandler(..., 
encoding="utf-8")`-style fix later if it becomes annoying.

**Also noted, not a bug:** the model (`openai/gpt-oss-120b` via Groq)
answered "what model are you?" by claiming to be "ChatGPT... GPT-4" —
factually wrong (it's an open-weight OSS model served by Groq, not
OpenAI's own hosting), but this is the model hallucinating about its own
identity, not a wiring issue. The API call, auth, response parsing, and
persistence all worked correctly.

**Status: RESOLVED.** `GroqLLMClient` is now fully verified end to end,
not just wired. Independently reconfirmed by the user right after this
entry: their own manual request/log output showed the same shape — new
session created, `POST https://api.groq.com/openai/v1/chat/completions
"200 OK"` in the httpx log, both messages persisted, real reply
(`"Hello! How can I help you today?"`) — matching this session's test.
Next planned option remaining: start the PDF ingestion pipeline (see
earlier entries).

### 2026-08-25 — Groq wired into `LLMClient`
**By:** Claude (Sonnet 5), this session.
**Why:** User picked "wire Groq" over "start PDF ingestion" as the next
step (both were logged as live options after the session-continuity
verification). `GROQ_API_KEY` / `GROQ_MODEL` config placeholders already
existed from the earlier foundation setup; nothing had consumed them yet.

Changes:
- `backend/app/llm/base.py` — added `GroqLLMClient(LLMClient)`, using the
  `groq` SDK's `AsyncGroq` client, `chat.completions.create()` with a
  single user-role message. Logs a warning and returns `""` if Groq
  returns an empty reply (defensive — hasn't actually been observed).
  `StubLLMClient` (echo) untouched, still exists.
- `backend/app/api/dependencies.py` — `get_llm_client()` now returns
  `GroqLLMClient` when `settings.GROQ_API_KEY` is non-empty, else logs a
  warning and falls back to `StubLLMClient`. Chosen over a hard failure
  so the app keeps booting/testable without a key (matches how Weaviate
  connectivity failures already degrade `/health` instead of crashing).
- `backend/requirements/base.txt` — added `groq==0.11.0`. Checked for
  conflicts against the existing `weaviate-client==4.9.6` httpx pin
  (`httpx<=0.27.0`) via `pip install --dry-run` before adding for real —
  none found; `groq` and `weaviate-client`'s own httpx constraints
  coexist fine at the versions currently pinned.
- `backend/README.md` — mentions Groq as the LLM provider, documents the
  automatic stub fallback when no key is set.

**Verified this session:** `pip install -r requirements/dev.txt`
succeeded with no resolver conflicts; `pytest` → **3 passed**, same as
before (no `GROQ_API_KEY` set in this environment, so all three existing
tests exercised the stub-fallback path, confirming that path still works
unchanged).

**NOT verified this session — no live Groq call has been made yet.**
`backend/.env`'s `GROQ_API_KEY` is still empty; user is adding their own
key and testing a live chat request themselves (declined to paste the
key into this session). Until that happens, treat `GroqLLMClient` as
wired-but-unexercised: it compiles and is reachable via DI, but no
request has actually round-tripped through the real Groq API to confirm
the API shape (`response.choices[0].message.content`), model name
(`openai/gpt-oss-120b`), auth, or error handling actually work against
the live service.

### 2026-08-25 — Verified: reusing `conversation_id` appends to the same session
**By:** Claude (Sonnet 5), this session (new account, per the handoff note
below — read PROJECT_LOG.md first as instructed, no code changes needed
before this check).
**Why:** This was the one explicitly unverified item left by the previous
session — `ChatRepository.get_or_create_session()`'s "found an existing
row" branch had never actually been exercised, only "create new" had.

No code changes. Verification steps:
- Confirmed both Docker containers already running (`docker compose ps`):
  `postgres` on host port 5433, `weaviate` on 8080/50051.
- Started the backend (`uvicorn app.main:app`) against the existing
  `backend/.env` (port 5433, as fixed in the entry below — left as is).
- `GET /health` → `{"status":"ok","database":"connected","vector_store":"connected"}`.
- `POST /api/v1/chat` with `{"message":"first message"}` →
  `conversation_id: c189ad72-...`.
- `POST /api/v1/chat` with `{"message":"second message",
  "conversation_id":"c189ad72-..."}` → same `conversation_id` echoed back.
- Queried Postgres directly: exactly **one** `chat_sessions` row for that
  UUID, with all **four** messages (`user`/`first message`,
  `assistant`/`Echo: first message`, `user`/`second message`,
  `assistant`/`Echo: second message`) attached to it in order, and
  `SELECT count(*) FROM chat_sessions` confirms only 2 sessions total
  exist (this one + one from earlier testing) — no duplicate/orphan
  session was created on the second request.

**Status: RESOLVED.** Session-continuity behavior works as designed.
This closes out the last open item from the persistence work; both
remaining options below (Groq, or PDF ingestion) are now unblocked and
still neither has been started.

### 2026-08-25 — PostgreSQL schema: chat persistence + document/ingestion tables
**By:** Claude (Sonnet 5), this session.
**Why:** User approved starting the next phase with the Postgres schema
(over wiring Groq first), since chat/document persistence is the
foundation everything else (ingestion, versioning, evaluation runs) sits
on top of. Before this, `POST /api/v1/chat` did not persist anything —
every request was stateless.

Changes:
- `backend/app/models/base.py` — **new.** Shared `Base`
  (`DeclarativeBase`) + `TimestampMixin` (`created_at`/`updated_at`,
  DB-generated).
- `backend/app/models/chat.py` — **new.** `ChatSession`, `Message`
  (UUID PKs, `messages.session_id` FK with cascade delete). Actively used.
- `backend/app/models/document.py` — **new.** `Document`,
  `DocumentVersion` (unique `(document_id, version)`). **Schema only —
  nothing writes to these yet**, ahead of the ingestion pipeline.
- `backend/app/models/upload_job.py` — **new.** `UploadJob` (status
  tracking for background ingestion). **Schema only, unconsumed for now.**
- `backend/app/models/__init__.py` — imports all models so
  `Base.metadata` is fully populated when the package is imported (needed
  for `create_all` to see every table).
- `backend/scripts/init_db.py` — **new.** One-off `Base.metadata.create_all`
  runner (`python -m scripts.init_db`). No Alembic yet, per the original
  setup constraint — this is the simplest thing that lets tables exist
  without a migration system.
- `backend/app/repositories/chat_repository.py` — **new.**
  `ChatRepository.get_or_create_session()` / `.add_message()`.
- `backend/app/services/chat_service.py` — **rewritten.**
  `ChatService` now takes a `ChatRepository`, persists the user message
  and the assistant reply, and commits the session's transaction.
  `conversation_id` is now parsed as a UUID (`_parse_session_id`); an
  invalid/foreign value is treated as "start a new session" rather than
  raising — this field's validation isn't hardened yet, so silently
  falling back was chosen over a 4xx error.
- `backend/app/api/dependencies.py` — added `get_chat_repository()`;
  `get_chat_service()` now wires the repository in alongside the LLM
  client.
- `backend/app/schemas/chat.py` — updated `conversation_id`'s field
  description to reflect the new UUID-session semantics.
- `backend/README.md` — documented the `init_db` step and the new
  persistence behavior.

**Breaking change to note:** `conversation_id` used to accept any
free-form string (the earlier manual test used `"conversation_id": "1"`,
which is not a valid UUID). That still won't error now — it's just
silently treated as "no id supplied" and a fresh session is created — but
it will no longer round-trip as `"1"` in the response; the response will
contain a real UUID instead.

**Not done in this entry:** no automated test added for the new
persistence behavior — it needs a live Postgres, and there's no test-DB
fixture/conftest yet to isolate that from the dev database. Existing unit
tests (`test_app_imports`, `test_health_endpoint_returns_ok_shape`,
`test_graph_compiles_and_runs`) don't touch this code path and should
still pass without Postgres running.

**Verified (same day):** user ran `python -m scripts.init_db`, started
the backend, and sent a live chat message. SQLAlchemy's own query log
confirmed the full flow: `INSERT INTO chat_sessions`, then two
`INSERT INTO messages` (role `user` with content `"hi"`, role
`assistant` with content `"Echo: hi"`), then `COMMIT`. Persistence works
end to end. Still outstanding: hasn't yet been verified that reusing the
same `conversation_id` on a second request appends to the same session
rather than creating a new one — the code path for that
(`ChatRepository.get_or_create_session` finding an existing row) hasn't
been exercised yet, only the create-new-session path has.

### 2026-08-25 — Session handoff: switching accounts due to token limit
**Note for the next session/account:** the user is switching Claude
accounts here because this session's tokens ran out — not because of any
blocker in the work itself. Foundation phase + chat persistence are both
done and verified (see entries above). Read this whole file before doing
anything else; the "Current State" section above is the authoritative
snapshot, and the entries above this one have all the *why* behind the
current shape of things (especially the Postgres port-5433 fix — don't
"helpfully" revert it back to 5432, there's a real conflict on this
machine that caused hours of debugging).

**Immediate next step, still unverified:** confirm that sending a second
`POST /api/v1/chat` with the `conversation_id` from a previous response
appends a new message pair to the *same* `chat_sessions` row instead of
creating a new session.

**After that, pick one (both were proposed, neither started):**
1. Wire Groq into `LLMClient` (`backend/app/llm/base.py`) using the
   already-present `GROQ_API_KEY` / `GROQ_MODEL` config — replaces
   `StubLLMClient`.
2. Start the PDF ingestion pipeline (extraction → cleaning → fixed-size
   chunking at ~600 tokens / ~100 overlap) as a standalone module, ahead
   of wiring it to Weaviate — the `Document`/`DocumentVersion`/
   `UploadJob` tables already exist for it to write into.

See `../../plan.md` for the full long-term spec either path eventually
feeds into.

### 2026-08-25 — Real root cause: port 5432 conflict with another local Postgres
**By:** Claude (Sonnet 5), this session.
**Why:** Even after a genuinely fresh `postgres` container/volume, with the
Python app confirmed (via `repr()`) to read the exact correct
host/port/user/password/db from `.env`, and the container's own password
confirmed byte-clean via `printenv ... | xxd`, `InvalidPasswordError`
still occurred. `netstat -ano | findstr :5432` revealed **two separate
processes** listening on port 5432 (PIDs `21848` and `7828`), with active
TCP connections actually landing on PID `7828` — i.e., something else on
this Windows machine (a native Postgres service, or another project's
Postgres) was already bound to 5432 and silently absorbing the
connections meant for this project's Docker container, using different
credentials. This user has many other Postgres-backed projects on this
machine (seen in `docker volume ls`: `rag-chatbot`, `system_pg_data`,
`tempmail-server`, `ecommerce-support-bot`, etc.), so a port clash was
plausible from the start — this is a machine-level conflict, not a bug
in this project's code or config.

Changes (moved this project off the contested port instead of touching
any other service on the machine):
- `docker-compose.yml` — `postgres` port mapping `"5432:5432"` →
  `"5433:5432"` (container still listens on 5432 internally; only the
  host-side published port changed).
- `backend/.env`, `backend/.env.example` — `POSTGRES_PORT=5432` →
  `POSTGRES_PORT=5433` to match.

**Status: RESOLVED.** User recreated the `postgres` container on port
5433 and confirmed `GET /health` now returns
`{"status": "ok", "database": "connected", "vector_store": "connected"}`.
Foundation phase is fully verified end to end: install, tests, backend
boot, chat endpoint, Postgres connectivity, and Weaviate connectivity all
confirmed working.

### 2026-08-25 — Debugging: `/health` reports database unavailable
**By:** Claude (Sonnet 5), this session.
**Why:** `GET /health` returned `"database": "unavailable"` while
`"vector_store": "connected"`. `docker compose ps` + `docker compose logs
postgres` show the Postgres container is fully healthy (initialized,
created the `rag_chatbot` DB from `POSTGRES_DB`, listening on
`0.0.0.0:5432`, accepting connections) — so the container itself is not
the problem. But `check_database_connection()` in
`backend/app/database/connection.py` caught the exception and returned
`False` silently — no error was ever visible in the uvicorn logs, so the
real cause (bad credentials? wrong host/port? something else?) couldn't
be diagnosed from the symptom alone.

Changes:
- `backend/app/database/connection.py` — `check_database_connection()`
  now logs the exception (`logger.exception(...)`) before returning
  `False`, instead of swallowing it silently.
- `backend/app/database/vector_store.py` — same fix applied to
  `check_weaviate_connection()` for consistency, even though Weaviate is
  currently working (so the same blind spot doesn't bite later).

**Root cause found** (via the new logging above):
`asyncpg.exceptions.InvalidPasswordError: password authentication failed
for user "postgres"`. The `postgres` container's data volume was almost
certainly initialized (first-ever `initdb`, confirmed by the "creating
subdirectories" log lines) at a moment when `backend/.env` held a
different `POSTGRES_PASSWORD` than it does now — Postgres only applies
`POSTGRES_PASSWORD` on first init of an empty volume, so editing `.env`
afterward doesn't retroactively change the already-initialized cluster's
password. This is an environment/ordering issue, not a code bug.

**Fix given to user** (not run by the assistant — user runs it): reset
the in-container password to match `.env`, using the fact that the
container's `pg_hba.conf` trusts local (socket) connections:
```
docker compose exec postgres psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"
```
Alternative (if that fails): `docker compose down -v` + `docker compose
up -d` to wipe and reinitialize both volumes from current `.env` — safe
here since nothing real is stored yet (just the empty `rag_chatbot` DB
and no Weaviate collections).

**Status: waiting on user to run the fix and confirm `/health` shows
`"database": "connected"`.**

### 2026-08-25 — Verified: Weaviate container reachable from the backend
**By:** user, confirmed in this session.
**Why:** confirming `docker compose up -d` + the new
`check_weaviate_connection()` helper actually work together, not just
that the code compiles.

Result: hitting `GET /health` produced (via the running uvicorn process)
`GET http://localhost:8080/v1/meta` → `200 OK` and
`GET http://localhost:8080/v1/.well-known/ready` → `200 OK`. Weaviate
container is up and the client connects successfully.

Noted (not a bug, just worth knowing): `weaviate-client` itself makes an
outbound call to `https://pypi.org/pypi/weaviate-client/json` on connect
to check for a newer version, which is why that PyPI request and the
"Dep005" deprecation warning (running 4.9.6, latest is 4.23.0) show up in
the logs. This is library behavior, not something added in this project —
harmless, but flagging it because it's an unexpected external network
call on every health check. If that's undesirable later (e.g. offline
dev, egress-restricted environments), it can be disabled via the
client's `additional_config`/env var — not done now since out of scope
for this phase.

Not yet independently confirmed in a logged output: PostgreSQL
connectivity via `check_database_connection()` (asyncpg calls aren't
logged via httpx the way Weaviate's are) — infer it's fine since `/health`
would otherwise report `"database": "unavailable"`, but no explicit log
line was seen to confirm.

### 2026-08-25 — Verified: backend starts and chat endpoint responds
**By:** user, confirmed in this session.
**Why:** confirming the app actually boots and serves a real request, not
just that pytest passes.

Result: `uvicorn app.main:app --reload` started cleanly (
`Application startup complete`), and a `POST /api/v1/chat` request was
handled (`Handling chat message for conversation_id=1` logged) and would
have returned the stub echo reply. No errors.

Not yet verified: `docker compose up -d` (Postgres/Weaviate containers)
hasn't been confirmed running yet in this session — until it is, `/health`
will report `"database": "unavailable"` / `"vector_store": "unavailable"`
even though the app itself is healthy. Next step: run `docker compose up
-d` from the repo root, then hit `GET /health` and confirm both show
`"connected"`.

### 2026-08-25 — Verified: install + test pass
**By:** user, confirmed in this session.
**Why:** confirming the httpx fix actually resolved the install, and that
the foundation (health endpoint, chat stub, LangGraph graph) actually runs.

Result: `pip install -r requirements/dev.txt` succeeded, `pytest` →
**3 passed** (`test_app_imports`, `test_health_endpoint_returns_ok_shape`,
`test_graph_compiles_and_runs`). Two non-blocking warnings only:
- `LangChainPendingDeprecationWarning` from `langgraph.checkpoint` about
  a future default change to `allowed_objects` — safe to ignore for now,
  revisit if it becomes a real deprecation.
- `weaviate-client` 4.9.6 is behind latest (4.23.0) — pinned deliberately
  for the httpx compatibility fix above; revisit the pin if a future
  weaviate-client release relaxes its httpx constraint.

No code changes this entry — pure verification. Postgres/Weaviate Docker
containers not yet started/verified at this point (health endpoint would
report `degraded` until `docker compose up -d` is run).

### 2026-08-25 — Fix pip dependency conflict: httpx pin vs weaviate-client (take 2)
**By:** Claude (Sonnet 5), this session.
**Why:** First fix (`httpx==0.27.2`) still violated weaviate-client 4.9.6's
constraint `httpx<=0.27.0` — 0.27.2 > 0.27.0, so pip still failed. The
upper bound is inclusive of 0.27.0 exactly, not the whole 0.27.x line.

Changes:
- `backend/requirements/dev.txt` — `httpx==0.27.2` → `httpx==0.27.0`
  (the actual upper bound weaviate-client 4.9.6 allows).

Not yet verified: user still needs to re-run `pip install -r
requirements/dev.txt`. If this specific combination keeps causing churn,
consider dropping the hard `==` pin on httpx in dev.txt in favor of a
range (e.g. `httpx>=0.25.0,<=0.27.0`) so future base.txt bumps don't
require hunting the exact patch version again.

### 2026-08-25 — Fix pip dependency conflict: httpx pin vs weaviate-client
**By:** Claude (Sonnet 5), this session.
**Why:** User ran `pip install -r requirements/dev.txt` and hit
`ResolutionImpossible`: `weaviate-client==4.9.6` requires
`httpx>=0.25.0,<=0.27.0`, but `dev.txt` pinned `httpx==0.28.1` (used only
for the test client). The two pins are mutually exclusive.

Changes:
- `backend/requirements/dev.txt` — `httpx==0.28.1` → `httpx==0.27.2`
  (thought this satisfied weaviate-client's upper bound — it didn't, see
  entry above).

Not yet verified: user still needs to re-run `pip install -r
requirements/dev.txt` to confirm this resolves cleanly (no commands were
run by the assistant this turn).

### 2026-08-25 — Foundation setup: LangGraph + Weaviate + env rework
**By:** Claude (Sonnet 5), this session.
**Why:** User's "RAG Chatbot — Initial Project Setup" spec required
Postgres+Weaviate running via Docker (not host-installed), a LangGraph
dependency/foundation, and Weaviate/Groq config — the existing Phase 1
backend only had Postgres wired directly to the host.

Changes:
- `backend/app/core/config.py` — renamed `APP_ENV`→`ENVIRONMENT`,
  `DATABASE_*`→`POSTGRES_*` (to match the spec's exact env var names);
  added `WEAVIATE_HOST/PORT/GRPC_PORT`, `GROQ_API_KEY`, `GROQ_MODEL`.
- `backend/app/main.py` — updated to `settings.ENVIRONMENT`.
- `backend/app/database/vector_store.py` — **new.** Weaviate client +
  `check_weaviate_connection()`. Connection only, no collections/schema
  (matches spec: "do not implement the complete retrieval pipeline yet").
- `backend/app/api/routes/health.py` — now also reports Weaviate
  connectivity (via `asyncio.to_thread` so the sync client call doesn't
  block the event loop).
- `backend/app/schemas/common.py` — `HealthStatus` gained `vector_store`.
- `backend/tests/unit/test_health.py` — asserts the new health fields.
- `backend/app/graph/state.py`, `backend/app/graph/workflow.py` — **new.**
  Minimal `GraphState` + a single-node compiled LangGraph
  (`build_graph()`), proving LangGraph works end to end. NOT yet used by
  the chat endpoint — that's a later phase per spec ("only create the
  clean foundation").
- `backend/tests/unit/test_graph.py` — **new.** Verifies the graph
  compiles and runs.
- `backend/.env.example`, `backend/.env` — rewritten with the new var
  names, added Weaviate + Groq placeholders.
- `backend/requirements/base.txt` — added `langgraph==0.2.60`,
  `weaviate-client==4.9.6`.
- `docker-compose.yml` — **replaced.** Old version ran a `backend`
  container pointing at a host-installed Postgres. New version runs only
  `postgres` (postgres:16-alpine) + `weaviate`
  (semitechnologies/weaviate:1.27.0), both with named volumes, per the
  spec's explicit instruction to containerize the databases and keep the
  backend local for this phase. **Note:** this removes the `backend`
  service that previously existed in compose — flagged to the user, not
  reverted unless they ask.
- `Makefile` — added `docker-up` / `docker-down` targets.
- `README.md`, `backend/README.md` — updated to describe this phase
  accurately (was still saying "Phase 1 complete" with old var names).
- No commands were run (install/docker/tests) — user runs those
  themselves by policy for this session.

### (prior, undated) — Phase 1: backend foundation
**By:** unknown/earlier session (predates this log's creation).
Established: FastAPI app (`main.py`), layered architecture
(routes → services → repositories/LLM), structured logging
(`core/logging.py`), centralized exception handling
(`core/exceptions.py`), async Postgres connection (`database/connection.py`,
host-based at the time), `GET /health`, `POST /api/v1/chat` through
`ChatService` → `StubLLMClient` (echo), Pydantic schemas, base repository
class, pytest setup (`pytest-asyncio`), Dockerfile, `.gitignore`/
`.dockerignore`, empty scaffolding folders for future phases
(`agents/`, `crew/`, `dspy/`, `mcp/`, `rag/`, `graph/`, `models/`,
`observability/`, `mcp-servers/`, `frontend/`, `data/`, `docs/`).
