"""Health check route."""

import asyncio

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.database.connection import check_database_connection
from app.database.vector_store import check_weaviate_connection
from app.infrastructure.redis.client import ping_redis
from app.schemas.common import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health_check(request: Request) -> HealthStatus:
    """Report whether the backend (and its dependencies) are up and running."""
    settings = get_settings()
    db_ok = await check_database_connection()
    vector_store_ok = await asyncio.to_thread(check_weaviate_connection)

    redis_client = getattr(request.app.state, "redis_client", None)
    if settings.SESSION_MEMORY_ENABLED and redis_client is not None:
        redis_ok = await ping_redis(redis_client)
        session_memory = "connected" if redis_ok else "unavailable"
    elif not settings.SESSION_MEMORY_ENABLED:
        session_memory = "disabled"
    else:
        session_memory = "unavailable"

    return HealthStatus(
        status="ok" if (db_ok and vector_store_ok) else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        database="connected" if db_ok else "unavailable",
        vector_store="connected" if vector_store_ok else "unavailable",
        session_memory=session_memory,
    )
