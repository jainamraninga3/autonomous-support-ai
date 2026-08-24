# Autonomous Support AI

A production-oriented autonomous AI support chatbot, built phase by phase.

Planned capabilities (not all implemented yet): FastAPI backend, PostgreSQL,
RAG with hybrid (BM25 + vector) search, MCP, LangGraph, CrewAI, DSPy, and
autonomous issue resolution / ticket creation.

## Current status

**Phase 1 complete:** backend foundation — FastAPI app, configuration,
structured logging, centralized exception handling, async PostgreSQL
connection layer, health endpoint, and a chat endpoint wired through a
service → LLM abstraction (stub LLM, no real provider yet).

See [backend/README.md](backend/README.md) for setup and run instructions.

## Repository layout

- `backend/` — Python/FastAPI application
- `frontend/` — (not yet implemented)
- `mcp-servers/` — MCP server implementations (future phase)
- `data/` — documents, processed data, evaluation sets
- `docker/`, `docker-compose.yml` — containerization
- `docs/` — architecture notes, API docs, decision records
