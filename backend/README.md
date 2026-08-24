# Autonomous Support AI — Backend

Phase 1: production backend foundation (FastAPI + PostgreSQL). No RAG, agents,
or LLM providers are implemented yet — this phase only establishes the
architecture those features will plug into later.

## Setup

```powershell
cd backend
.venv\Scripts\Activate.ps1
pip install -r requirements/dev.txt
copy .env.example .env   # then edit DATABASE_PASSWORD etc. if needed
```

## Run

```powershell
uvicorn app.main:app --reload
```

## Test

```powershell
pytest
```

## Endpoints

- `GET /health` — service + database status
- `POST /api/v1/chat` — chat request/response through the service → LLM
  abstraction (stub LLM for now)

## Architecture

```
API (routes) -> Service -> Repository / Database
                        -> LLM abstraction -> LLM provider
```

Routes contain no business logic; all logic lives in `app/services`.
