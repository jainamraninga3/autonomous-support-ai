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

    # --- CORS ---
    # Required for the Next.js frontend (frontend/): a browser refuses a
    # cross-origin XHR to :8000 from :3000 without these headers, and the
    # failure looks like a network error with no useful detail in the UI.
    # Comma-separated, or "*" to allow any origin. "*" is fine for local
    # development and wrong for anything public — this API has no auth.
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

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
    # GROQ_API_KEY is tried first; GROQ_API_1/_2/_3 are optional extra
    # keys the client automatically falls through to, in order, whenever
    # the current key hits a rate limit (see `app/llm/base.py`). Each is
    # independently optional — set only the ones you have.
    GROQ_API_KEY: str = ""
    GROQ_API_1: str = ""
    GROQ_API_2: str = ""
    GROQ_API_3: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    @property
    def cors_allow_origins(self) -> list[str]:
        """Parsed `CORS_ALLOW_ORIGINS`. `["*"]` allows any origin."""
        raw = self.CORS_ALLOW_ORIGINS.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def groq_api_keys(self) -> list[str]:
        """All configured Groq API keys, in fallback order, blanks
        dropped. Empty if none are set."""
        return [key for key in (self.GROQ_API_KEY, self.GROQ_API_1, self.GROQ_API_2, self.GROQ_API_3) if key]

    # --- Document ingestion ---
    # HISTORY, kept because it names the exact failure mode and predicted
    # this change. Was 100/10, set at the user's explicit direction over
    # the 600/100 that `chunk_pages` defaults to: at ~100 tokens this corpus's
    # policy text splits mid-sentence, and a 10-token overlap is too
    # narrow to guarantee a multi-clause sentence survives whole in
    # either neighbour. That exact failure produced a wrong answer —
    # "Accumulated earned leave may be carried forward, each year, EL
    # beyond 30 days is encashed by the Company in January" was severed,
    # and a query retrieving only the tail told users carry-forward was
    # not in the policy at all. If that class of error reappears, this is
    # the first thing to raise; parent-child chunking (plan.md section
    # 10) is the way to keep small retrieval units without it.
    # 100/10 -> 500/100 on 2026-09-08, from a measured retrieval failure.
    #
    # At 100 tokens the corpus was 481 chunks averaging 97 tokens, and
    # clauses were being cut mid-sentence. The verifier caught the
    # consequence twice on one question: it rejected an answer because
    # "the policy fragment ends with 'A weekly off or declared holiday
    # falling within a'" — the model had completed the sentence from
    # prior knowledge, correctly, but unsupported by the fragment it was
    # given. Small chunks do not just lose context; they invite the model
    # to finish the thought.
    #
    # 500/100 keeps a policy clause whole and, with RERANK_TOP_K=5, gives
    # ~2500 tokens of context instead of ~500.
    #
    # CHANGING THIS REQUIRES RE-INGESTING: chunks are stored, so existing
    # documents keep their old 100-token chunks until re-uploaded.
    CHUNK_SIZE_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 100

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
    # 10 -> 5 on 2026-09-08, from an observed retrieval failure rather
    # than a guess. "what is the leave policy" returned 10 chunks: 7 from
    # Leave_Policy.pdf and 3 from Maternity_Benefit_Policy.pdf and
    # Separation_Policy.pdf. All were technically leave-related, so the
    # verifier passed the answer — but the model dutifully wrote up every
    # one of them and produced a page-long table covering tubectomy leave
    # and notice-period shortfall for someone who asked a one-line
    # question.
    #
    # NOTE THIS APPLIES WITH RERANKING OFF TOO (see RERANK_ENABLED
    # below): `retrieve_node` uses it to slice the RRF-fused candidate
    # list, so it is the context size for every request either way. The
    # name is now misleading and worth renaming when something else
    # touches this area.
    #
    # The trade-off is real: a question whose answer genuinely spans 6+
    # chunks will now lose the tail. Raise it per-request with `top_k` in
    # the chat payload rather than changing this back.
    RERANK_TOP_K: int = 5
    RERANK_SCORE_THRESHOLD: float | None = None

    # Defaults to False: the evaluation set (backend/rag_chat_test) now
    # exists and says reranking does not pay for itself on this corpus.
    # Across both a 16-question and a 2-question run, hybrid-without-
    # rerank matched or beat rerank+hybrid on answer quality (8.00 vs
    # 7.69, then 9.0 vs 9.0) while taking roughly a third of the time
    # (16.1s vs 48.9s, then 31.9s vs 43.5s), and won more questions
    # outright. The reranker model and `rerank_chunks` are kept — this
    # is one env var to flip if a bigger or more heterogeneous corpus
    # changes that answer, and the harness still A/B tests all four
    # variants on every run regardless of this setting.
    RERANK_ENABLED: bool = False

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
