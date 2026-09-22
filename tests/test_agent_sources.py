import json

from app.agent.events import SourceRef
from app.agent.sources import extract_sources, merge_sources


def test_extract_sources_from_search_knowledge_result():
    result = json.dumps(
        [
            {"document_slug": "cancellation", "title": "退訂與退款政策", "content": "退款比例...", "similarity": 0.52},
            {"document_slug": "force-majeure", "title": "不可抗力與天候政策", "content": "颱風...", "similarity": 0.39},
        ]
    )

    sources = extract_sources("search_knowledge", result, is_error=False)

    assert len(sources) == 2
    assert sources[0].document_slug == "cancellation"
    assert sources[1].similarity == 0.39


def test_extract_sources_ignores_non_search_knowledge_tools():
    result = json.dumps([{"id": "t1", "title": "京都賞楓五日遊"}])
    assert extract_sources("search_tours", result, is_error=False) == []


def test_extract_sources_ignores_errored_calls():
    result = json.dumps({"error": "something broke"})
    assert extract_sources("search_knowledge", result, is_error=True) == []


def test_extract_sources_handles_malformed_json_gracefully():
    assert extract_sources("search_knowledge", "not json", is_error=False) == []


def test_merge_sources_dedupes_by_document_slug_and_content():
    existing = [SourceRef(document_slug="cancellation", title="退訂政策", content="A", similarity=0.5)]
    new = [
        SourceRef(document_slug="cancellation", title="退訂政策", content="A", similarity=0.5),  # exact dup
        SourceRef(document_slug="cancellation", title="退訂政策", content="B", similarity=0.4),  # new chunk
    ]

    merged = merge_sources(existing, new)

    assert len(merged) == 2
    assert [s.content for s in merged] == ["A", "B"]


def test_merge_sources_preserves_order_existing_first():
    existing = [SourceRef(document_slug="a", title="A", content="1", similarity=0.9)]
    new = [SourceRef(document_slug="b", title="B", content="2", similarity=0.8)]

    merged = merge_sources(existing, new)

    assert [s.document_slug for s in merged] == ["a", "b"]
