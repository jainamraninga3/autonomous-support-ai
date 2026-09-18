"""Signup / login / logout / me.

No business logic here, same as every other router — it all lives in
`AuthService`. These four routes are the only unauthenticated write path in
the application, which is why `SignupRequest` has no `role` field: the
service hardcodes `role="user"`, so this endpoint cannot mint an admin no
matter what is posted to it.
"""

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.dependencies import get_auth_service, get_current_user
from app.models.user import User
from app.schemas.auth import LoginRequest, LoginResponse, SignupRequest, UserResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])

# Logout needs the RAW token to delete that specific session row, not the
# resolved user — `get_current_user` gives the user and throws the token
# away. `auto_error=False` so a logout without a token is a no-op rather
# than an error.
_bearer_scheme = HTTPBearer(auto_error=False)


@router.post("/signup", response_model=LoginResponse)
async def signup(
    request: SignupRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> LoginResponse:
    """Create an account (always a plain user) and log straight in."""
    return await auth_service.signup(request)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> LoginResponse:
    """Exchange credentials for a session token. Send it back as
    `Authorization: Bearer <token>` on every subsequent request."""
    return await auth_service.login(request)


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> dict:
    """End this session. Deliberately succeeds even without a valid token —
    a client clearing its stored token should never be blocked by the
    server disagreeing about whether it was logged in."""
    if credentials and credentials.credentials:
        await auth_service.logout(credentials.credentials)
    return {"message": "Logged out."}


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Who the current token belongs to. The frontend calls this on load to
    restore a session from `localStorage`; a 401 here means "show the
    logged-out UI", not an error worth reporting."""
    return UserResponse(
        id=str(user.id), username=user.username, full_name=user.full_name, role=user.role
    )
