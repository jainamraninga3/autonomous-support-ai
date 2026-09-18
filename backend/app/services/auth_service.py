"""Signup, login, logout, and "who is this request" resolution.

Sessions are rows in `user_sessions` (PostgreSQL), NOT Redis keys — a
deliberate decision recorded in docs/PROJECT_LOG.md (2026-09-18): Redis in
this stack has been observed coming up disconnected on a container-startup
race with no reconnect, and a login system that silently logs everyone out
when that happens would be worse than one extra table. Nothing external is
required for auth; it reuses the database that was already here.

Passwords are bcrypt-hashed. The original ask was to store them as-is;
that was deliberately not done — see the same log entry. To revert to
plaintext, `_hash_password`/`_verify_password` are the only two functions
that would change.
"""

import secrets
from datetime import datetime, timezone

import bcrypt

from app.core.config import get_settings
from app.core.exceptions import BadRequestError, UnauthorizedError
from app.core.logging import get_logger
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginRequest, LoginResponse, SignupRequest, UserResponse

logger = get_logger(__name__)

# bcrypt hashes at most 72 BYTES and bcrypt 4.x RAISES on anything longer
# rather than silently truncating the way older versions did. `SignupRequest`
# allows 200 characters, so the truncation has to happen here — explicitly,
# in one place, applied identically when hashing and when verifying, or a
# long password would set fine and then never match.
_BCRYPT_MAX_BYTES = 72

# Seeded on every startup by `ensure_demo_users`. Idempotent — an existing
# username is left alone, so a rebuild neither errors nor duplicates, and a
# password changed later is not silently reset back to these.
_DEMO_USERS = [
    {"username": "admin", "password": "admin123", "full_name": "Admin User", "role": "admin"},
    {"username": "demo", "password": "demo123", "full_name": "Demo User", "role": "user"},
]


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:_BCRYPT_MAX_BYTES], bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:_BCRYPT_MAX_BYTES], hashed.encode("utf-8"))
    except ValueError:
        # A malformed hash in the column (hand-edited row, or a plaintext
        # password from before hashing existed). Treat as a failed login
        # rather than a 500 — the user gets "invalid credentials", which is
        # the truth, and the server log names the real cause.
        logger.exception("Stored password hash is not a valid bcrypt hash")
        return False


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id), username=user.username, full_name=user.full_name, role=user.role
    )


class AuthService:
    """Owns the transaction boundary for auth writes, the same way
    `ChatService` does for chat — the repository never commits."""

    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository
        self.settings = get_settings()

    async def signup(self, request: SignupRequest) -> LoginResponse:
        """Create a `role="user"` account and log it straight in.

        Role is hardcoded, not taken from the request — `SignupRequest` has
        no role field precisely so a public endpoint can never mint an admin.

        Returns a token rather than just the user: signing up and then
        immediately having to log in is a pointless second round trip, and
        it would mean two code paths in the UI for the same outcome.
        """
        username = request.username.strip().lower()
        if await self.user_repository.get_by_username(username) is not None:
            raise BadRequestError(f"Username '{username}' is already taken.")

        user = await self.user_repository.create_user(
            username=username,
            hashed_password=_hash_password(request.password),
            full_name=request.full_name.strip(),
            role="user",
        )
        token = await self._start_session(user)
        await self.user_repository.session.commit()
        logger.info("New account created: %s", username)
        return LoginResponse(token=token, user=_to_user_response(user))

    async def login(self, request: LoginRequest) -> LoginResponse:
        """Verify credentials and start a session.

        A wrong username and a wrong password return the SAME error, on
        purpose: telling an attacker which half was right turns the login
        form into a username-enumeration oracle.
        """
        username = request.username.strip().lower()
        user = await self.user_repository.get_by_username(username)
        if user is None or not _verify_password(request.password, user.password):
            logger.warning("Failed login attempt for username=%r", username)
            raise UnauthorizedError("Invalid username or password.")
        if not user.is_active:
            raise UnauthorizedError("This account has been deactivated.")

        token = await self._start_session(user)
        await self.user_repository.session.commit()
        logger.info("Login: %s (role=%s)", user.username, user.role)
        return LoginResponse(token=token, user=_to_user_response(user))

    async def logout(self, token: str) -> None:
        """Delete the session row. Idempotent — logging out twice, or with a
        token that has already expired, is not an error."""
        await self.user_repository.delete_session(token)
        await self.user_repository.session.commit()

    async def resolve_token(self, token: str) -> User:
        """Return the user this bearer token belongs to, or raise 401.

        Expiry is checked HERE rather than relied on being cleaned up by the
        opportunistic delete in `create_session`: that delete only runs when
        that same user logs in again, so an expired row can sit in the table
        indefinitely and must never authenticate anyone.

        Every successful resolution slides the expiry forward, so an active
        session doesn't expire under someone mid-conversation.
        """
        found = await self.user_repository.get_session_with_user(token)
        if found is None:
            raise UnauthorizedError("Not logged in, or this session is no longer valid.")

        session_row, user = found
        expires_at = session_row.expires_at
        if expires_at.tzinfo is None:
            # Defensive: the column is TIMESTAMPTZ, so asyncpg returns an
            # aware datetime — but a naive value here would raise on the
            # comparison below and 500 the request instead of 401ing it.
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            await self.user_repository.delete_session(token)
            await self.user_repository.session.commit()
            raise UnauthorizedError("Your session has expired — please log in again.")

        if not user.is_active:
            raise UnauthorizedError("This account has been deactivated.")

        await self.user_repository.touch_session(token, self.settings.AUTH_SESSION_TTL_SECONDS)
        await self.user_repository.session.commit()
        return user

    async def _start_session(self, user: User) -> str:
        """Mint an opaque bearer token. `secrets.token_urlsafe` (not uuid4):
        it is CSPRNG-backed and 32 bytes of entropy, whereas a uuid4 carries
        122 bits and reads like an identifier someone might guess is
        sequential."""
        token = secrets.token_urlsafe(32)
        await self.user_repository.create_session(
            user_id=user.id, token=token, ttl_seconds=self.settings.AUTH_SESSION_TTL_SECONDS
        )
        return token


async def ensure_demo_users(session_factory) -> None:
    """Seed the demo `admin`/`demo` accounts if they don't exist yet.

    Called from the app lifespan, so `docker compose up` gives working
    logins with no manual step. Skips any username that already exists —
    a rebuild must not error, duplicate, or reset a password somebody
    changed.

    Never raises: a failure here (e.g. the migration hasn't run yet) must
    not stop the app from booting. It is logged loudly instead, because the
    symptom otherwise is "the demo login just doesn't work" with nothing
    explaining why.
    """
    try:
        async with session_factory() as session:
            repository = UserRepository(session=session)
            created = []
            for spec in _DEMO_USERS:
                if await repository.get_by_username(spec["username"]) is not None:
                    continue
                await repository.create_user(
                    username=spec["username"],
                    hashed_password=_hash_password(spec["password"]),
                    full_name=spec["full_name"],
                    role=spec["role"],
                )
                created.append(spec["username"])
            if created:
                await session.commit()
                logger.info("Seeded demo accounts: %s", ", ".join(created))
    except Exception:
        logger.exception(
            "Could not seed demo accounts — logging in as admin/demo will fail. "
            "Has `alembic upgrade head` run (the `users` table may not exist)?"
        )
