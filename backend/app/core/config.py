"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings.

    All values are sourced from environment variables (or a local .env file
    in development). Nothing here is hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "Autonomous Support AI"
    ENVIRONMENT: str = "development"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- PostgreSQL (application database, runs via Docker) ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "rag_chatbot"

    # --- Weaviate (vector database, runs via Docker) ---
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    WEAVIATE_GRPC_PORT: int = 50051

    # --- Groq (LLM provider) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # --- Document ingestion ---
    # Where extracted/chunked output is written, ahead of Weaviate being
    # wired in. Relative to the backend/ working directory by default.
    PROCESSED_DATA_DIR: str = "../data/processed"
    CHUNK_SIZE_TOKENS: int = 100
    CHUNK_OVERLAP_TOKENS: int = 20

    # --- Embeddings (dense vectors for Weaviate) ---
    # BGE-M3 per plan.md section 11. Loading this model requires `torch`
    # + `sentence-transformers` installed and triggers a ~2.3GB download
    # on first use — deliberately not forced by anything at import time.
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024

    # --- Reranking (plan.md section 19) ---
    # BGE-Reranker-v2-M3, applied to Stage 1's Top-30 to select a
    # relevance-based Top-10-max (not forced to exactly 10 — plan.md
    # section 18 explicitly says the final count may be 3/5/7/10).
    # No RERANK_SCORE_THRESHOLD default is set (None = cap by count
    # only) since there's no evaluation set yet to derive one from.
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K: int = 10
    RERANK_SCORE_THRESHOLD: float | None = None

    # --- Document uploads ---
    # Where PDFs uploaded via `POST /api/v1/documents/upload` are saved
    # to disk (ahead of ingestion — see `app/services/document_service.py`).
    DOCUMENTS_DIR: str = "../data/documents"

    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection URL built from environment variables."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
