"""Pydantic schemas for the admin reset endpoints."""

from pydantic import BaseModel


class ResetResponse(BaseModel):
    message: str
    details: list[str]


class RestartResponse(BaseModel):
    message: str
    will_restart: bool
    details: list[str]
