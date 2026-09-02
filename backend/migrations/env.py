"""Alembic environment, wired to the app's own settings and models.

Two deliberate choices:

- The database URL comes from `app.core.config.get_settings()`, not from
  `alembic.ini`. One source of truth, so `alembic upgrade head` can never
  migrate a different database than the one the app talks to.
- Async throughout (`asyncpg`), matching `app/database/connection.py`.
  Alembic's migration steps are synchronous, so the async connection is
  handed to them via `connection.run_sync`.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import get_settings

# Importing the models package registers every table on Base.metadata,
# which is what `--autogenerate` diffs against. A model that isn't
# imported here is invisible to autogenerate.
from app.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (`alembic upgrade --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Both on so autogenerate notices a changed column type or server
        # default, not just added/dropped columns — the failure mode that
        # made the old `create_all` approach quietly wrong.
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
