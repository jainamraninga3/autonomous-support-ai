"""Tests for the health endpoint and application import."""

import pytest
from httpx import ASGITransport, AsyncClient


def test_app_imports() -> None:
    """The FastAPI application must be importable without side effects."""
    from app.main import app

    assert app is not None


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok_shape() -> None:
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "app_name" in body
    assert "version" in body
    assert "database" in body
    assert "vector_store" in body
