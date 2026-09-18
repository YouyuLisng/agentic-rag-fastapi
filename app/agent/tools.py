import json
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, Field

from app.rag.retrieval import search_knowledge


class SearchKnowledgeInput(BaseModel):
    query: str = Field(
        description="A natural-language question in Traditional Chinese about "
        "policy/informational topics -- cancellation & refunds, insurance "
        "coverage, payment terms, pre-departure preparation, visa/document "
        "requirements, packing advice, force-majeure/weather handling, or "
        "accommodations for elderly/child travelers. NOT for finding tours "
        "by location/budget/days -- that's a different tool."
    )


# Stage 3 adds search_tours / get_tour_detail here -- the loop and
# executor below are written to scale to more tools without changes.
TOOLS: list[ToolParam] = [
    {
        "name": "search_knowledge",
        "description": (
            "Semantic search over the travel agency's policy knowledge base "
            "(cancellation/refund rules, insurance, payment terms, "
            "pre-departure prep, visa requirements, packing advice, "
            "force-majeure/weather policy, accommodations for elderly/child "
            "travelers). Use this for questions about policies, rules, or "
            "general travel-prep advice -- not for finding or comparing "
            "specific tours."
        ),
        "input_schema": SearchKnowledgeInput.model_json_schema(),
    },
]

_HANDLERS: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {}


async def _run_search_knowledge(raw_input: dict[str, Any]) -> Any:
    validated = SearchKnowledgeInput.model_validate(raw_input)
    return await search_knowledge(validated.query)


_HANDLERS["search_knowledge"] = _run_search_knowledge


async def execute_tool(name: str, raw_input: dict[str, Any]) -> tuple[str, bool]:
    """Runs the named tool. Returns (content_json, is_error). Errors are
    caught and returned as an error payload with is_error=True rather than
    raised, so the model sees the failure (per Anthropic's tool_result
    is_error convention) and can decide how to respond -- retry with
    different input, fall back to another tool, or tell the user -- rather
    than the whole turn blowing up."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False), True

    try:
        result = await handler(raw_input)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False), True

    return json.dumps(result, ensure_ascii=False, default=str), False
