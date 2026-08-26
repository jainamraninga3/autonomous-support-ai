"""Destructive data-reset routes for local development/testing.

Not gated behind auth or an environment check — this project has none
yet (see `docs/PROJECT_LOG.md`). Treat these as dev-only until that
changes; don't expose this router in a deployment that has real users.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_admin_service
from app.schemas.admin import ResetResponse
from app.services.admin_service import AdminService

router = APIRouter(prefix="/admin/reset", tags=["admin"])


@router.post("/postgres", response_model=ResetResponse)
async def reset_postgres(service: AdminService = Depends(get_admin_service)) -> ResetResponse:
    """Deletes all documents and chat history from PostgreSQL. Does not
    touch Weaviate or files on disk — see `/admin/reset/all` for both."""
    details = await service.reset_postgres()
    return ResetResponse(message="PostgreSQL data reset.", details=details)


@router.post("/vector-store", response_model=ResetResponse)
async def reset_vector_store(service: AdminService = Depends(get_admin_service)) -> ResetResponse:
    """Deletes and recreates the Weaviate `DocumentChunk` collection,
    discarding every embedded chunk. Does not touch PostgreSQL."""
    details = service.reset_vector_store()
    return ResetResponse(message="Vector store reset.", details=details)


@router.post("/all", response_model=ResetResponse)
async def reset_all(service: AdminService = Depends(get_admin_service)) -> ResetResponse:
    """Resets both PostgreSQL and the Weaviate vector store."""
    details = await service.reset_postgres()
    details += service.reset_vector_store()
    return ResetResponse(message="PostgreSQL and vector store both reset.", details=details)
