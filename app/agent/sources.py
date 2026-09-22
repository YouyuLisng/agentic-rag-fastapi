"""Citation extraction shared by both agent implementations -- turning
raw tool_result payloads into FinalAnswerEvent.sources.
"""

import json

from app.agent.events import SourceRef


def extract_sources(name: str, result_json: str, is_error: bool) -> list[SourceRef]:
    """Only search_knowledge results become citations -- structured
    tool results (tour data) aren't "sources" in the RAG-citation
    sense, and an errored call has nothing to cite."""
    if name != "search_knowledge" or is_error:
        return []
    try:
        hits = json.loads(result_json)
    except json.JSONDecodeError:
        return []
    return [
        SourceRef(document_slug=h["document_slug"], title=h["title"], content=h["content"], similarity=h["similarity"])
        for h in hits
    ]


def merge_sources(existing: list[SourceRef], new: list[SourceRef]) -> list[SourceRef]:
    """Append-dedupe by (document_slug, content) -- the same chunk can
    come back from more than one search_knowledge call in a multi-tool
    turn, and citations shouldn't list it twice."""
    seen = {(s.document_slug, s.content) for s in existing}
    merged = list(existing)
    for s in new:
        key = (s.document_slug, s.content)
        if key not in seen:
            seen.add(key)
            merged.append(s)
    return merged
