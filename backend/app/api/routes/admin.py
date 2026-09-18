"""Destructive data-reset routes.

**ADMIN ONLY as of 2026-09-18** — every route here requires
`role == "admin"` (`require_admin`). Before that they were completely
unauthenticated, which meant an unauthenticated POST could wipe every
document and chat, or SIGTERM the process.

Still dev-oriented: an admin can destroy all data with one call and there
is no confirmation, no audit trail, and no undo. The router-level
dependency is the boundary now, not the absence of a UI button.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_admin_service, require_admin
from app.schemas.admin import ResetResponse, RestartResponse
from app.services.admin_service import AdminService

# Declared on the ROUTER, not per-route: a new destructive endpoint added
# here is then gated by default, instead of being unprotected until someone
# remembers to add the dependency.
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.post("/reset/postgres", response_model=ResetResponse)
async def reset_postgres(service: AdminService = Depends(get_admin_service)) -> ResetResponse:
    """Deletes all documents and chat history from PostgreSQL. Does not
    touch Weaviate or files on disk — see `/admin/reset/all` for both."""
    details = await service.reset_postgres()
    return ResetResponse(message="PostgreSQL data reset.", details=details)


@router.post("/reset/vector-store", response_model=ResetResponse)
async def reset_vector_store(service: AdminService = Depends(get_admin_service)) -> ResetResponse:
    """Deletes and recreates the Weaviate `DocumentChunk` collection,
    discarding every embedded chunk. Does not touch PostgreSQL."""
    details = service.reset_vector_store()
    return ResetResponse(message="Vector store reset.", details=details)


@router.post("/reset/all", response_model=ResetResponse)
async def reset_all(service: AdminService = Depends(get_admin_service)) -> ResetResponse:
    """Resets both PostgreSQL and the Weaviate vector store."""
    details = await service.reset_postgres()
    details += service.reset_vector_store()
    return ResetResponse(message="PostgreSQL and vector store both reset.", details=details)


@router.post("/restart", response_model=RestartResponse)
async def restart_backend(service: AdminService = Depends(get_admin_service)) -> RestartResponse:
    """Stop this process so its supervisor restarts it.

    Only an actual restart when something is supervising the process —
    `docker compose`'s `restart: unless-stopped` does that. Run directly
    under `uvicorn`, nothing will start it again and this just shuts the
    server down; the response says which case applies rather than
    pretending both work.

    In-flight requests are not drained beyond the short delay before
    SIGTERM. Dev-only, like the resets above.
    """
    will_restart, details = service.request_restart()
    message = (
        "Restarting — the API will be unavailable for a few seconds."
        if will_restart
        else "Shutting down. Nothing is supervising this process, so it will NOT come back on its own."
    )
    return RestartResponse(message=message, will_restart=will_restart, details=details)
