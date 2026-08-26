# Autonomous Support AI

A production-oriented autonomous AI support chatbot, built phase by phase.

Planned capabilities (not all implemented yet): FastAPI backend, PostgreSQL,
RAG with hybrid (BM25 + vector) search, MCP, LangGraph, CrewAI, DSPy, and
autonomous issue resolution / ticket creation.

## Current status

**Foundation phase complete:** FastAPI app, environment-based configuration,
structured logging, centralized exception handling, async PostgreSQL
connection layer, a Weaviate connection helper, a minimal LangGraph
workflow, a health endpoint, and a chat endpoint wired through a
service → LLM abstraction (stub LLM, no real provider yet).

PostgreSQL and Weaviate run via Docker (`docker-compose.yml`); the backend
runs locally from `backend/.venv` for now.

Not yet implemented (later phases): document ingestion, chunking,
embeddings, hybrid retrieval, reranking, query classification/rewriting,
citations, answer verification, evaluation, auth/RBAC, tenant isolation,
MCP, CrewAI, DSPy.

See [backend/README.md](backend/README.md) for setup and run instructions.

## Repository layout

- `backend/` — Python/FastAPI application
- `frontend/` — (not yet implemented)
- `mcp-servers/` — MCP server implementations (future phase)
- `data/` — documents, processed data, evaluation sets
- `docker/`, `docker-compose.yml` — PostgreSQL + Weaviate containers
- `docs/` — architecture notes, API docs, decision records
