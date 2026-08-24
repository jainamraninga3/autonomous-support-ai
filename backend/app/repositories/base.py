"""Base repository abstraction for database access.

Concrete repositories (added in later phases as models are introduced)
will extend this base class to keep data access logic out of services
and routes.
"""

from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType]):
    """Generic base repository holding a database session."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
