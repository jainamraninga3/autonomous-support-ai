"""Tests that the auth boundary is actually wired to the routes.

These assert on HTTP status codes through the real ASGI app rather than
calling the dependencies directly — the bug worth catching here is not
"does `require_admin` raise 403" (that is a two-line function) but "is it
attached to every route that needs it". A dependency that exists and is
never wired up looks identical to one that works, right until someone
POSTs to `/admin/reset/all` without a token.

No database is touched: `get_current_user`/`require_admin` are overridden
via FastAPI's dependency_overrides, and the authenticated cases never
reach a route body that needs a real session (401/403 are raised before
the handler runs).
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_auth_service, get_current_user, require_admin
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.main import app
from app.models.user import User

# Every route that must reject an anonymous caller. Kept as one list so a
# newly-added protected route is one line to cover.
PROTECTED = [
    ("post", "/api/v1/chat", {"message": "hi"}),
    ("post", "/api/v1/admin/reset/postgres", None),
    ("post", "/api/v1/admin/reset/vector-store", None),
    ("post", "/api/v1/admin/reset/all", None),
    ("post", "/api/v1/admin/restart", None),
    ("get", "/api/v1/documents", None),
]

# Deliberately reachable without logging in. `/health` is a probe, and the
# console panel is meant to work before login — see docs/PROJECT_LOG.md.
PUBLIC = [
    ("get", "/health"),
    ("get", "/api/v1/logs/tail?offset=-1"),
]


def _plain_user() -> User:
    return User(
        id=uuid.uuid4(),
        username="demo",
        password="x",
        full_name="Demo User",
        role="user",
        is_active=True,
    )


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", PROTECTED)
async def test_protected_routes_reject_an_anonymous_request(method, path, body) -> None:
    async with await _client() as client:
        response = await client.request(method.upper(), path, json=body)

    assert response.status_code == 401, f"{method.upper()} {path} was reachable without a token"


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", PROTECTED)
async def test_unknown_token_is_401_on_every_protected_route(method, path, body) -> None:
    """An unknown token must come back as 401, never a 500 — an auth check
    that crashes is an auth check that can be probed.

    `AuthService` is faked here rather than left real: resolving a token
    hits `user_sessions`, and these tests run with no database. What this
    covers is the part that is NOT the service — that
    `UnauthorizedError` raised during dependency resolution reaches the
    registered exception handler and becomes a 401 on each of these routes,
    rather than escaping as an unhandled 500. The service's own
    unknown-token behaviour is covered in test_auth_service.py.
    """

    class RejectingAuthService:
        async def resolve_token(self, token):
            raise UnauthorizedError("Not logged in, or this session is no longer valid.")

    app.dependency_overrides[get_auth_service] = RejectingAuthService
    try:
        async with await _client() as client:
            response = await client.request(
                method.upper(), path, json=body, headers={"Authorization": "Bearer nonsense"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", PUBLIC)
async def test_public_routes_stay_reachable_without_a_token(method, path) -> None:
    async with await _client() as client:
        response = await client.request(method.upper(), path)

    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/admin/reset/postgres",
        "/api/v1/admin/reset/vector-store",
        "/api/v1/admin/reset/all",
        "/api/v1/admin/restart",
    ],
)
async def test_admin_routes_forbid_a_logged_in_plain_user(path) -> None:
    """403, not 401: they ARE logged in, they just aren't allowed. The
    frontend acts differently on each — one opens the login modal."""

    async def _forbid() -> User:
        raise ForbiddenError("This action requires an administrator account.")

    app.dependency_overrides[get_current_user] = _plain_user
    app.dependency_overrides[require_admin] = _forbid
    try:
        async with await _client() as client:
            response = await client.post(path)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_chat_ignores_a_client_supplied_user_id() -> None:
    """The ownership hole this closes: `user_id` is what session ownership
    is enforced against in SQL, so a client-chosen value would let anyone
    read anyone else's conversation. The route must overwrite it with the
    authenticated user's id before the service ever sees it.
    """
    from app.api.dependencies import get_chat_service
    from app.schemas.chat import ChatResponse

    user = _plain_user()
    seen: dict = {}

    class RecordingChatService:
        async def handle_message(self, request):
            seen["user_id"] = request.user_id
            return ChatResponse(reply="ok", conversation_id=str(uuid.uuid4()))

    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_chat_service] = RecordingChatService
    try:
        async with await _client() as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "hi", "user_id": "somebody-elses-id"},
                headers={"Authorization": "Bearer whatever"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert seen["user_id"] == str(user.id)
    assert seen["user_id"] != "somebody-elses-id"
