"""Two-stage search_knowledge orchestration, with the vector search and
the Voyage rerank call both mocked -- so this exercises the wiring
(candidate width, reorder, score attachment, toggle) without a DB or any
API call, and runs in CI like every other unit test."""

from unittest.mock import AsyncMock, patch

import pytest

from app.rag import retrieval

CANDIDATES = [
    {"document_slug": "cancellation", "title": "退訂政策", "content": "A", "similarity": 0.52},
    {"document_slug": "force-majeure", "title": "不可抗力政策", "content": "B", "similarity": 0.51},
    {"document_slug": "insurance", "title": "保險政策", "content": "C", "similarity": 0.40},
]


@pytest.fixture
def _settings():
    with patch("app.rag.retrieval.get_settings") as gs:
        gs.return_value.rerank_enabled = True
        gs.return_value.rerank_candidate_count = 20
        yield gs.return_value


async def test_rerank_reorders_and_attaches_score(_settings):
    # rerank promotes force-majeure (candidate index 1) above cancellation.
    with (
        patch("app.rag.retrieval.embed_query", new=AsyncMock(return_value=[0.0])),
        patch("app.rag.retrieval._vector_search", new=AsyncMock(return_value=CANDIDATES)),
        patch(
            "app.rag.retrieval.rerank_documents",
            new=AsyncMock(return_value=[(1, 0.98), (0, 0.42)]),
        ),
    ):
        hits = await retrieval.search_knowledge("因為疫情取消行程,費用會退嗎?", match_count=2)

    assert [h["document_slug"] for h in hits] == ["force-majeure", "cancellation"]
    assert hits[0]["rerank_score"] == 0.98
    # first-stage similarity is preserved alongside the new rerank score
    assert hits[0]["similarity"] == 0.51


async def test_rerank_retrieves_wide_then_trims(_settings):
    _settings.rerank_candidate_count = 20
    vector = AsyncMock(return_value=CANDIDATES)
    with (
        patch("app.rag.retrieval.embed_query", new=AsyncMock(return_value=[0.0])),
        patch("app.rag.retrieval._vector_search", new=vector),
        patch("app.rag.retrieval.rerank_documents", new=AsyncMock(return_value=[(0, 0.9)])),
    ):
        hits = await retrieval.search_knowledge("q", match_count=1)

    # first stage asked for the wide candidate_count, not match_count
    assert vector.await_args is not None
    assert vector.await_args.args[1] == 20
    assert len(hits) == 1


async def test_rerank_disabled_is_single_stage(_settings):
    _settings.rerank_enabled = False
    vector = AsyncMock(return_value=CANDIDATES[:2])
    rerank = AsyncMock()
    with (
        patch("app.rag.retrieval.embed_query", new=AsyncMock(return_value=[0.0])),
        patch("app.rag.retrieval._vector_search", new=vector),
        patch("app.rag.retrieval.rerank_documents", new=rerank),
    ):
        hits = await retrieval.search_knowledge("q", match_count=5)

    rerank.assert_not_awaited()
    # passed match_count straight through, no rerank_score added
    assert vector.await_args is not None
    assert vector.await_args.args[1] == 5
    assert "rerank_score" not in hits[0]


async def test_per_call_rerank_arg_overrides_settings(_settings):
    _settings.rerank_enabled = True
    rerank = AsyncMock()
    with (
        patch("app.rag.retrieval.embed_query", new=AsyncMock(return_value=[0.0])),
        patch("app.rag.retrieval._vector_search", new=AsyncMock(return_value=CANDIDATES)),
        patch("app.rag.retrieval.rerank_documents", new=rerank),
    ):
        await retrieval.search_knowledge("q", match_count=2, rerank=False)

    rerank.assert_not_awaited()


async def test_rerank_on_empty_candidates_returns_empty(_settings):
    rerank = AsyncMock()
    with (
        patch("app.rag.retrieval.embed_query", new=AsyncMock(return_value=[0.0])),
        patch("app.rag.retrieval._vector_search", new=AsyncMock(return_value=[])),
        patch("app.rag.retrieval.rerank_documents", new=rerank),
    ):
        hits = await retrieval.search_knowledge("q", match_count=5)

    assert hits == []
    rerank.assert_not_awaited()
