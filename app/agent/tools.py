import json
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, Field

from app.rag.retrieval import search_knowledge
from app.tours.queries import check_availability, get_tour_detail, search_tours


class SearchKnowledgeInput(BaseModel):
    query: str = Field(
        description="A natural-language question in Traditional Chinese about "
        "policy/informational topics -- cancellation & refunds, insurance "
        "coverage, payment terms, pre-departure preparation, visa/document "
        "requirements, packing advice, force-majeure/weather handling, or "
        "accommodations for elderly/child travelers. NOT for finding tours "
        "by location/budget/days -- that's a different tool."
    )


class SearchToursInput(BaseModel):
    country: str | None = Field(
        default=None,
        description="Filter by destination country. Must be exactly one of: "
        "日本, 泰國, 紐西蘭, 韓國, 印尼, 台灣. Omit to search every country.",
    )
    max_budget_twd: int | None = Field(
        default=None, description="Maximum budget in TWD. Omit for no budget ceiling."
    )
    min_days: int | None = Field(default=None, description="Minimum trip length in days.")
    max_days: int | None = Field(default=None, description="Maximum trip length in days.")
    suitable_for: str | None = Field(
        default=None,
        description="Filter by traveler type. Must be exactly one of: "
        "一般大眾, 賞楓愛好者, 親子家庭, 長輩, 銀髮族, 學生, 小資族, 自然愛好者, "
        "健行族, 年輕族群, 情侶, 蜜月, 樂齡族. Pick the closest matching tag rather "
        "than inventing a new one -- an unrecognized tag will just return zero results.",
    )


class GetTourDetailInput(BaseModel):
    tour_id: str = Field(
        description="A tour's id (UUID), exactly as returned by a prior "
        "search_tours call -- never guess or invent one."
    )


class CheckAvailabilityInput(BaseModel):
    tour_id: str = Field(
        description="A tour's id (UUID), exactly as returned by a prior "
        "search_tours call -- never guess or invent one."
    )


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
    {
        "name": "search_tours",
        "description": (
            "Structured filter search over the tour catalog by country, "
            "budget, trip length, and/or traveler type. Returns a list of "
            "matching tours (id, title, country, location, days, budget, "
            "suitable_for, summary) -- NOT the full day-by-day itinerary, "
            "use get_tour_detail with a returned id for that. All filters "
            "are optional and combine with AND; omit filters the user didn't "
            "specify rather than guessing a value. Use this for 'find a "
            "tour matching X' questions, not for policy/rule questions."
        ),
        "input_schema": SearchToursInput.model_json_schema(),
    },
    {
        "name": "get_tour_detail",
        "description": (
            "Full detail for one specific tour, including its day-by-day "
            "itinerary. Requires a tour id obtained from a prior "
            "search_tours call in this conversation -- call search_tours "
            "first if you don't already have one."
        ),
        "input_schema": GetTourDetailInput.model_json_schema(),
    },
    {
        "name": "check_availability",
        "description": (
            "Real-time check of whether a specific tour still has open "
            "spots (capacity minus current enrollment) -- e.g. '這團還有位子嗎'. "
            "This is a live fact lookup, not a description -- use "
            "get_tour_detail instead for itinerary/content questions about "
            "a tour. Requires a tour id from a prior search_tours call."
        ),
        "input_schema": CheckAvailabilityInput.model_json_schema(),
    },
]

_HANDLERS: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {}


async def _run_search_knowledge(raw_input: dict[str, Any]) -> Any:
    validated = SearchKnowledgeInput.model_validate(raw_input)
    return await search_knowledge(validated.query)


async def _run_search_tours(raw_input: dict[str, Any]) -> Any:
    validated = SearchToursInput.model_validate(raw_input)
    return await search_tours(
        country=validated.country,
        max_budget_twd=validated.max_budget_twd,
        min_days=validated.min_days,
        max_days=validated.max_days,
        suitable_for=validated.suitable_for,
    )


async def _run_get_tour_detail(raw_input: dict[str, Any]) -> Any:
    validated = GetTourDetailInput.model_validate(raw_input)
    result = await get_tour_detail(validated.tour_id)
    if result is None:
        return {"error": f"No tour found with id {validated.tour_id}"}
    return result


async def _run_check_availability(raw_input: dict[str, Any]) -> Any:
    validated = CheckAvailabilityInput.model_validate(raw_input)
    result = await check_availability(validated.tour_id)
    if result is None:
        return {"error": f"No tour found with id {validated.tour_id}"}
    return result


_HANDLERS["search_knowledge"] = _run_search_knowledge
_HANDLERS["search_tours"] = _run_search_tours
_HANDLERS["get_tour_detail"] = _run_get_tour_detail
_HANDLERS["check_availability"] = _run_check_availability


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
