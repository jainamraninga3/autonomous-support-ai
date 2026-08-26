"""Create all tables directly from the ORM models.

No Alembic/migrations yet (deliberately, per the foundation-phase scope).
This is a one-off dev convenience: run it once against a fresh database,
re-run after adding new models. It will not alter existing tables.

Usage (from backend/, with the venv active):
    python -m scripts.init_db
"""

import asyncio

from app.database.connection import engine
from app.models import Base


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(init_db())
