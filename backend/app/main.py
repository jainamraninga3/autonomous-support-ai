"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import admin, chat, documents, health, rag
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.database.vector_store import get_weaviate_client

configure_logging()
logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.ENVIRONMENT)

    try:
        app.state.weaviate_client = get_weaviate_client()
    except Exception:
        logger.exception(
            "Failed to connect to Weaviate at startup — /api/v1/rag/ask will be unavailable "
            "until Weaviate is reachable and the app is restarted"
        )
        app.state.weaviate_client = None

    yield

    if app.state.weaviate_client is not None:
        app.state.weaviate_client.close()
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)
app.include_router(rag.router, prefix=settings.API_V1_PREFIX)
app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
app.include_router(admin.router, prefix=settings.API_V1_PREFIX)
