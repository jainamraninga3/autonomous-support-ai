"""Pydantic schemas for the auth endpoints."""

from pydantic import BaseModel, Field


class SignupRequest(BaseModel):
    """Public self-signup. Always creates a `role="user"` account — there is
    no field for role here on purpose, so nobody can grant themselves admin
    through this endpoint. Admin accounts are seeded at startup instead
    (see `app/services/auth_service.py::ensure_demo_users`)."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=200)
    full_name: str = Field(..., min_length=1, max_length=200)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    full_name: str
    role: str


class LoginResponse(BaseModel):
    token: str
    user: UserResponse
