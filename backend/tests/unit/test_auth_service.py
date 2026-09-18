"""Unit tests for AuthService.

No database: `UserRepository` is faked with an in-memory dict, the same
way the rest of this suite fakes Weaviate, Redis, and the LLM. bcrypt
itself is real — hashing a 4-character password is fast enough, and
faking it would mean the tests never exercise the one thing that must not
silently break (a password that hashes but never verifies).
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.exceptions import BadRequestError, UnauthorizedError
from app.models.user import User, UserSession
from app.schemas.auth import LoginRequest, SignupRequest
from app.services.auth_service import AuthService, _hash_password


class FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class FakeUserRepository:
    """In-memory stand-in for `UserRepository`, with the same method
    signatures the service calls."""

    def __init__(self, users: list[User] | None = None) -> None:
        self.session = FakeSession()
        self.users: dict[str, User] = {u.username: u for u in (users or [])}
        self.sessions: dict[str, UserSession] = {}

    async def get_by_username(self, username: str) -> User | None:
        return self.users.get(username)

    async def create_user(self, username, hashed_password, full_name, role) -> User:
        user = User(
            id=uuid.uuid4(),
            username=username,
            password=hashed_password,
            full_name=full_name,
            role=role,
            is_active=True,
        )
        self.users[username] = user
        return user

    async def create_session(self, user_id, token, ttl_seconds) -> UserSession:
        now = datetime.now(timezone.utc)
        row = UserSession(
            token=token,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self.sessions[token] = row
        return row

    async def get_session_with_user(self, token):
        row = self.sessions.get(token)
        if row is None:
            return None
        user = next(u for u in self.users.values() if u.id == row.user_id)
        return (row, user)

    async def touch_session(self, token, ttl_seconds) -> None:
        row = self.sessions.get(token)
        if row is not None:
            row.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    async def delete_session(self, token) -> None:
        self.sessions.pop(token, None)


def _user(username="alice", password="pw1234", role="user", is_active=True) -> User:
    return User(
        id=uuid.uuid4(),
        username=username,
        password=_hash_password(password),
        full_name="Alice Example",
        role=role,
        is_active=is_active,
    )


@pytest.mark.asyncio
async def test_signup_creates_user_and_returns_token() -> None:
    repo = FakeUserRepository()
    service = AuthService(user_repository=repo)

    result = await service.signup(
        SignupRequest(username="Alice", password="pw1234", full_name="Alice Example")
    )

    assert result.token
    assert result.user.username == "alice"  # normalized to lowercase
    assert result.user.role == "user"
    assert repo.session.commits == 1


@pytest.mark.asyncio
async def test_signup_never_grants_admin_and_stores_a_hash() -> None:
    """The two properties that make public signup safe: the role is fixed,
    and the plaintext password is not what lands in the column."""
    repo = FakeUserRepository()
    service = AuthService(user_repository=repo)

    await service.signup(
        SignupRequest(username="mallory", password="pw1234", full_name="Mallory")
    )

    stored = repo.users["mallory"]
    assert stored.role == "user"
    assert stored.password != "pw1234"
    assert stored.password.startswith("$2b$")


@pytest.mark.asyncio
async def test_signup_rejects_a_taken_username() -> None:
    repo = FakeUserRepository([_user(username="alice")])
    service = AuthService(user_repository=repo)

    with pytest.raises(BadRequestError):
        await service.signup(
            SignupRequest(username="alice", password="pw1234", full_name="Someone Else")
        )


@pytest.mark.asyncio
async def test_login_succeeds_with_the_right_password() -> None:
    repo = FakeUserRepository([_user(password="pw1234")])
    service = AuthService(user_repository=repo)

    result = await service.login(LoginRequest(username="alice", password="pw1234"))

    assert result.user.username == "alice"
    assert result.token in repo.sessions


@pytest.mark.asyncio
async def test_login_rejects_a_wrong_password() -> None:
    repo = FakeUserRepository([_user(password="pw1234")])
    service = AuthService(user_repository=repo)

    with pytest.raises(UnauthorizedError):
        await service.login(LoginRequest(username="alice", password="wrong"))


@pytest.mark.asyncio
async def test_unknown_user_and_wrong_password_give_the_same_error() -> None:
    """Username enumeration: the two failures must be indistinguishable to
    the client, or the login form answers "does this account exist?"."""
    repo = FakeUserRepository([_user(password="pw1234")])
    service = AuthService(user_repository=repo)

    with pytest.raises(UnauthorizedError) as wrong_password:
        await service.login(LoginRequest(username="alice", password="wrong"))
    with pytest.raises(UnauthorizedError) as no_such_user:
        await service.login(LoginRequest(username="nobody", password="pw1234"))

    assert str(wrong_password.value) == str(no_such_user.value)


@pytest.mark.asyncio
async def test_login_rejects_a_deactivated_account() -> None:
    repo = FakeUserRepository([_user(is_active=False)])
    service = AuthService(user_repository=repo)

    with pytest.raises(UnauthorizedError):
        await service.login(LoginRequest(username="alice", password="pw1234"))


@pytest.mark.asyncio
async def test_resolve_token_returns_the_user_and_slides_expiry() -> None:
    repo = FakeUserRepository([_user()])
    service = AuthService(user_repository=repo)
    token = (await service.login(LoginRequest(username="alice", password="pw1234"))).token

    # Wind the expiry close to now, then confirm resolving pushes it back out.
    repo.sessions[token].expires_at = datetime.now(timezone.utc) + timedelta(seconds=5)
    user = await service.resolve_token(token)

    assert user.username == "alice"
    assert repo.sessions[token].expires_at > datetime.now(timezone.utc) + timedelta(seconds=60)


@pytest.mark.asyncio
async def test_resolve_token_rejects_an_unknown_token() -> None:
    service = AuthService(user_repository=FakeUserRepository())

    with pytest.raises(UnauthorizedError):
        await service.resolve_token("not-a-real-token")


@pytest.mark.asyncio
async def test_expired_session_is_rejected_and_deleted() -> None:
    """An expired row must never authenticate. The opportunistic cleanup in
    `create_session` only runs when that same user logs in again, so an
    expired row can sit in the table indefinitely — the check has to be
    here, not left to the cleanup."""
    repo = FakeUserRepository([_user()])
    service = AuthService(user_repository=repo)
    token = (await service.login(LoginRequest(username="alice", password="pw1234"))).token
    repo.sessions[token].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    with pytest.raises(UnauthorizedError):
        await service.resolve_token(token)
    assert token not in repo.sessions


@pytest.mark.asyncio
async def test_naive_expiry_does_not_crash_the_request() -> None:
    """A naive datetime in `expires_at` would raise on the comparison and
    turn a 401 into a 500. It is treated as UTC instead."""
    repo = FakeUserRepository([_user()])
    service = AuthService(user_repository=repo)
    token = (await service.login(LoginRequest(username="alice", password="pw1234"))).token
    repo.sessions[token].expires_at = datetime.utcnow() - timedelta(seconds=1)

    with pytest.raises(UnauthorizedError):
        await service.resolve_token(token)


@pytest.mark.asyncio
async def test_logout_deletes_the_session_and_is_idempotent() -> None:
    repo = FakeUserRepository([_user()])
    service = AuthService(user_repository=repo)
    token = (await service.login(LoginRequest(username="alice", password="pw1234"))).token

    await service.logout(token)
    assert token not in repo.sessions

    # Logging out twice must not raise — a client clearing its token should
    # never be blocked by the server disagreeing about the session.
    await service.logout(token)


@pytest.mark.asyncio
async def test_tokens_are_unique_per_login() -> None:
    repo = FakeUserRepository([_user()])
    service = AuthService(user_repository=repo)

    first = (await service.login(LoginRequest(username="alice", password="pw1234"))).token
    second = (await service.login(LoginRequest(username="alice", password="pw1234"))).token

    assert first != second
    # Both stay valid: logging in on a second device must not sign you out
    # of the first.
    assert await service.resolve_token(first)
    assert await service.resolve_token(second)
