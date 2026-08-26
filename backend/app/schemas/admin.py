"""Pydantic schemas for the admin reset endpoints."""

from pydantic import BaseModel


class ResetResponse(BaseModel):
    message: str
    details: list[str]
