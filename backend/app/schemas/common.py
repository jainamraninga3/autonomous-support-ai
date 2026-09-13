"""Shared/common Pydantic schemas."""

from pydantic import BaseModel


class HealthStatus(BaseModel):
    """Response schema for the health check endpoint."""

    status: str
    app_name: str
    version: str
    database: str
    vector_store: str
    session_memory: str = "disabled"


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error_code: str
    message: str
