# Frontend — Autonomous Support AI

Next.js 15 (App Router) + TypeScript + Tailwind. A chat window on the left
(60%) and a live view of the backend's own log on the right (40%), plus
toolbar buttons for the destructive admin operations.

No component library and no icon package — everything is Tailwind and
inline markup, so `npm install` pulls only Next, React, and Tailwind.

## Run it

```powershell
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

Then open <http://localhost:3000>. The backend must be running on
<http://localhost:8000> (change `NEXT_PUBLIC_API_BASE_URL` in
`.env.local` if it's elsewhere).

## What the console panel shows

Real backend logs, not a simulation. The backend already writes to
`backend/logs/app.log` via a RotatingFileHandler; `GET /api/v1/logs/tail`
serves it and this UI polls that once a second with a byte cursor.

Lines marked `ui` in violet are this frontend's own events (requests,
timings, failures) so you can line them up against what the server did.

Three filters:

- **app** (default) — only `app.*` loggers. Without this the panel is
  mostly SQLAlchemy echo and httpx request lines, which drown the handful
  of lines that explain what the pipeline actually decided.
- **all** — includes libraries (uvicorn, sqlalchemy, sentence-transformers).
- **problems** — warnings and errors only.

`follow` auto-scrolls; scrolling up turns it off so reading history isn't
yanked back down by the next line. `pause` stops polling.

## What the chat badges mean

Each answer carries the pipeline's own diagnostics, straight from the
`/api/v1/chat` response:

| badge | meaning |
| --- | --- |
| `from your documents` | `answer_source: rag` — grounded and checked against the retrieved chunks |
| `general knowledge` | `general_fallback` — nothing relevant found, so this is **not** from your documents |
| `small talk` | a greeting; no retrieval ran |
| `off topic` | refused; your message was never sent to the LLM |
| `verified` / `unverified` | the verification step's verdict (hover for its reason) |
| `N chunks` | chunks retrieved before generation |

`details` expands the retrieval rewrite, the English retrieval query (for
non-English questions), the verifier's reasoning, and the source list with
page numbers.

Worth knowing: `N chunks` can be greater than zero on a `general
knowledge` answer. Retrieval finds candidates; generation decides whether
they actually answer the question. That is not a bug.

## Logging in

The backend requires an account for everything except `/health` and the
log tail, so the chat is unusable logged out — the composer still accepts
a message, but the UI answers with "please log in" rather than firing a
request it knows will 401.

Two accounts are seeded by the backend at startup:

| username | password | sees |
| --- | --- | --- |
| `admin` | `admin123` | everything, including the four destructive toolbar buttons |
| `demo` | `demo123` | chat and the console panel; no reset/restart buttons |

**Log in** / **Sign up** are top-right; signing up always creates a plain
user and logs you straight in. The token lives in `localStorage` under
`asai.token` and is restored on page load — a 401 on that restore just
means the session expired and is handled silently.

Sessions last 24h of INACTIVITY, not 24h absolute: every authenticated
request slides the expiry forward.

**Hiding the admin buttons is presentation, not security.** The backend
requires `role == "admin"` on those routes regardless of what this UI
renders; a plain user who called them directly would get a 403.

## Toolbar buttons

Admin only — a plain user does not see these. All four are confirmed with
a dialog that says exactly what is destroyed, because none of them are
reversible.

| button | endpoint | effect |
| --- | --- | --- |
| **New chat** | — | clears the local transcript and starts a new `conversation_id` |
| **Reset PostgreSQL** | `POST /api/v1/admin/reset/postgres` | deletes all documents, chunk text, **stored PDF bytes**, and chat history |
| **Reset Weaviate** | `POST /api/v1/admin/reset/vector-store` | drops and recreates the embedding collection; documents survive, so you can re-ingest without re-uploading |
| **Reset everything** | `POST /api/v1/admin/reset/all` | both of the above |
| **Restart backend** | `POST /api/v1/admin/restart` | SIGTERM to the backend process |

**Restart only works when something supervises the process.** Under
`docker compose` the backend is `restart: unless-stopped`, so exiting *is*
a restart and it returns in a few seconds. Running `uvicorn` directly,
nothing will bring it back — the endpoint tells you which case applies
rather than pretending both work.

**Reset PostgreSQL destroys your source PDFs.** They are stored in
`document_versions.pdf_bytes` and nowhere else. Keep the original files
outside the project.

## Structure

```
frontend/
├── app/
│   ├── layout.tsx        html/body shell; page never scrolls, panels do
│   ├── page.tsx          state, API orchestration, the 60/40 split
│   └── globals.css       Tailwind + .panel/.btn-*/.badge component classes
├── components/
│   ├── AuthModal.tsx     login/signup — one component, two modes
│   ├── ChatPanel.tsx     transcript, composer, suggestions, typing state
│   ├── MessageBubble.tsx one message + its diagnostic badges and details
│   ├── ConsolePanel.tsx  log viewer: filters, follow, pause, clear
│   └── Toolbar.tsx       health + auth area + confirmed admin actions
└── lib/
    ├── api.ts            typed fetch wrappers; token storage + Bearer header
    ├── types.ts          mirrors the backend Pydantic schemas
    └── useConsole.ts     cursor-based log polling + client-side entries
```

## Backend requirements

Two things had to be added to the backend for this to work:

1. **CORS** (`app/main.py`) — a browser blocks a cross-origin call from
   `:3000` to `:8000` outright. `CORS_ALLOW_ORIGINS` in `backend/.env`
   controls it; `docker-compose.yml` sets it for the container.
2. **`GET /api/v1/logs/tail`** (`app/api/routes/logs.py`) — serves
   `logs/app.log` with a byte cursor.

`/api/v1/logs/tail` is deliberately left UNAUTHENTICATED so the console
panel works before login. It serves raw application log lines, which
include other users' questions — fine for a LAN dev tool, wrong for
anything public. The admin router is no longer in that category: it
requires an admin account as of 2026-09-18.

## Troubleshooting

**"Could not reach the backend…"** — either the backend is down or CORS
is blocking. The browser deliberately doesn't distinguish them, so check
both: `curl http://localhost:8000/health`, then check
`CORS_ALLOW_ORIGINS` includes `http://localhost:3000`.

**Console stays empty but chat works** — the log file is only created once
the app logs something, and the first poll starts at the *end* of the file
on purpose (so you see new activity rather than replaying up to 5MB).
Send a message and lines will appear.

**Console shows nothing from the container** — the backend inside Docker
writes to `/app/logs/app.log` in the container, which is what the
container's own endpoint serves, so this works. It is only a problem if
you point the frontend at one backend and read logs from another.

**Restart button says "will NOT come back on its own"** — you're running
uvicorn directly, not under Docker. That message is accurate; start it
yourself.
