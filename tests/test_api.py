"""FastAPI route tests using httpx AsyncClient with mocked app state."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from scholar.api.main import app


def _make_mock_session_factory():
    """Return an async context-manager factory that yields a mock session."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock()

    class _FakeCtx:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    factory = MagicMock(return_value=_FakeCtx())
    return factory, mock_session


@pytest.fixture
def mock_app_state():
    """Patch app.state with minimal mocks so routes don't hit real DB/index."""
    factory, session = _make_mock_session_factory()

    app.state.async_session_factory = factory
    app.state.hybrid_retriever = None
    app.state.dense_retriever = None
    app.state.bm25 = None
    app.state.ltr_model = None

    yield {"factory": factory, "session": session}


@pytest.fixture
def async_client(mock_app_state):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_ok(self, async_client):
        async with async_client as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_no_index_returns_503(self, async_client):
        async with async_client as client:
            resp = await client.post(
                "/search",
                json={"query": "neural networks"},
            )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_search_with_mock_retriever(self, mock_app_state):
        from sqlmodel import select

        from scholar.db.models import Paper

        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [("paper123", 0.9)]

        mock_paper = Paper(
            id="paper123",
            title="Test Paper",
            abstract="Test abstract.",
            authors="Author A",
            categories="cs.LG",
            citation_count=5,
        )

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [mock_paper]
        mock_app_state["session"].execute.return_value = result_mock

        app.state.hybrid_retriever = mock_retriever
        app.state.dense_retriever = None

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/search", json={"query": "test query"})

            assert resp.status_code == 200
            data = resp.json()
            assert "results" in data
        finally:
            app.state.hybrid_retriever = None


class TestUserEndpoint:
    @pytest.mark.asyncio
    async def test_create_user(self, mock_app_state):
        from scholar.db.models import User

        created_user = None

        def capture_add(obj):
            nonlocal created_user
            if isinstance(obj, User):
                created_user = obj

        mock_app_state["session"].add.side_effect = capture_add

        refresh_mock = AsyncMock()
        mock_app_state["session"].refresh = refresh_mock

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/user")

        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data


class TestRecommendEndpoint:
    @pytest.mark.asyncio
    async def test_recommend_no_dense_returns_503(self, async_client):
        async with async_client as client:
            resp = await client.get("/recommend/some-user-id")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_recommend_user_not_found_returns_404(self, mock_app_state):
        mock_dense = MagicMock()
        mock_dense._paper_ids = []
        mock_dense._index = None
        app.state.dense_retriever = mock_dense

        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = None
        mock_app_state["session"].execute.return_value = result_mock

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/recommend/nonexistent-user")
            assert resp.status_code == 404
        finally:
            app.state.dense_retriever = None
