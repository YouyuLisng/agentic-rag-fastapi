import json

import pytest

from app.agent import tools as tools_module
from app.agent.tools import build_tools, execute_tool

TOOL_NAMES = {"search_knowledge", "search_tours", "get_tour_detail", "check_availability"}


def test_build_tools_without_document_excludes_search_document():
    names = {t["name"] for t in build_tools(document_id=None)}
    assert names == TOOL_NAMES


def test_build_tools_with_document_adds_search_document():
    names = {t["name"] for t in build_tools(document_id="doc-123")}
    assert names == {*TOOL_NAMES, "search_document"}


async def test_execute_tool_unknown_name_is_error():
    content, is_error = await execute_tool("does_not_exist", {})
    assert is_error is True
    assert "Unknown tool" in json.loads(content)["error"]


async def test_execute_tool_search_document_without_upload_is_error():
    content, is_error = await execute_tool("search_document", {"query": "q"}, document_id=None)
    assert is_error is True
    assert "No document uploaded" in json.loads(content)["error"]


async def test_execute_tool_search_document_binds_document_id_server_side(monkeypatch: pytest.MonkeyPatch):
    calls = []

    async def fake_search_document(document_id: str, query: str):
        calls.append((document_id, query))
        return {"chunks": ["match"]}

    monkeypatch.setattr(tools_module, "_search_document", fake_search_document)

    content, is_error = await execute_tool("search_document", {"query": "文件裡說什麼?"}, document_id="doc-abc")

    assert is_error is False
    assert calls == [("doc-abc", "文件裡說什麼?")]
    assert json.loads(content) == {"chunks": ["match"]}


async def test_execute_tool_search_tours_delegates_and_serializes(monkeypatch: pytest.MonkeyPatch):
    async def fake_search_tours(**kwargs):
        assert kwargs == {
            "country": "日本",
            "max_budget_twd": None,
            "min_days": None,
            "max_days": None,
            "suitable_for": None,
        }
        return [{"id": "t1", "title": "東京五日遊"}]

    monkeypatch.setattr(tools_module, "search_tours", fake_search_tours)

    content, is_error = await execute_tool("search_tours", {"country": "日本"})

    assert is_error is False
    assert json.loads(content) == [{"id": "t1", "title": "東京五日遊"}]


async def test_execute_tool_get_tour_detail_not_found_returns_error_payload_but_not_is_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_get_tour_detail(tour_id: str):
        return None

    monkeypatch.setattr(tools_module, "get_tour_detail", fake_get_tour_detail)

    content, is_error = await execute_tool("get_tour_detail", {"tour_id": "missing-id"})

    # A "not found" is reported as a normal result payload (the model
    # sees {"error": ...} and can react), not the tool_result.is_error
    # channel -- distinct from an exception, which does set is_error.
    assert is_error is False
    assert "No tour found" in json.loads(content)["error"]


async def test_execute_tool_swallows_handler_exception_as_error_payload(monkeypatch: pytest.MonkeyPatch):
    async def fake_check_availability(tour_id: str):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(tools_module, "check_availability", fake_check_availability)

    content, is_error = await execute_tool("check_availability", {"tour_id": "t1"})

    assert is_error is True
    assert "db exploded" in json.loads(content)["error"]
