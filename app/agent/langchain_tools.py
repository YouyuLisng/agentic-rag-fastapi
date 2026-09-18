import json

from langchain_core.tools import tool

from app.rag.retrieval import search_knowledge as _search_knowledge
from app.tours.queries import get_tour_detail as _get_tour_detail
from app.tours.queries import search_tours as _search_tours

# Same three tools, same names, same underlying query functions as the
# hand-rolled version (app/agent/tools.py) -- only the schema/registration
# mechanism differs, so the comparison in the README is apples to apples,
# not "framework version has better tools."


@tool(parse_docstring=True)
async def search_knowledge(query: str) -> str:
    """Semantic search over the travel agency's policy knowledge base
    (cancellation/refund rules, insurance, payment terms, pre-departure
    prep, visa requirements, packing advice, force-majeure/weather
    policy, accommodations for elderly/child travelers). Use this for
    questions about policies, rules, or general travel-prep advice --
    not for finding or comparing specific tours.

    Args:
        query: A natural-language question in Traditional Chinese about
            policy/informational topics. NOT for finding tours by
            location/budget/days -- that's a different tool.
    """
    results = await _search_knowledge(query)
    return json.dumps(results, ensure_ascii=False, default=str)


@tool(parse_docstring=True)
async def search_tours(
    country: str | None = None,
    max_budget_twd: int | None = None,
    min_days: int | None = None,
    max_days: int | None = None,
    suitable_for: str | None = None,
) -> str:
    """Structured filter search over the tour catalog by country,
    budget, trip length, and/or traveler type. Returns a list of
    matching tours (id, title, country, location, days, budget,
    suitable_for, summary) -- NOT the full day-by-day itinerary, use
    get_tour_detail with a returned id for that. All filters are
    optional and combine with AND; omit filters the user didn't
    specify rather than guessing a value. Use this for 'find a tour
    matching X' questions, not for policy/rule questions.

    Args:
        country: Filter by destination country. Must be exactly one
            of: 日本, 泰國, 紐西蘭, 韓國, 印尼, 台灣. Omit to search
            every country.
        max_budget_twd: Maximum budget in TWD. Omit for no ceiling.
        min_days: Minimum trip length in days.
        max_days: Maximum trip length in days.
        suitable_for: Filter by traveler type. Must be exactly one of:
            一般大眾, 賞楓愛好者, 親子家庭, 長輩, 銀髮族, 學生, 小資族,
            自然愛好者, 健行族, 年輕族群, 情侶, 蜜月, 樂齡族. Pick the
            closest matching tag rather than inventing a new one -- an
            unrecognized tag will just return zero results.
    """
    results = await _search_tours(
        country=country,
        max_budget_twd=max_budget_twd,
        min_days=min_days,
        max_days=max_days,
        suitable_for=suitable_for,
    )
    return json.dumps(results, ensure_ascii=False, default=str)


@tool(parse_docstring=True)
async def get_tour_detail(tour_id: str) -> str:
    """Full detail for one specific tour, including its day-by-day
    itinerary. Requires a tour id obtained from a prior search_tours
    call in this conversation -- call search_tours first if you don't
    already have one.

    Args:
        tour_id: A tour's id (UUID), exactly as returned by a prior
            search_tours call -- never guess or invent one.
    """
    result = await _get_tour_detail(tour_id)
    if result is None:
        return json.dumps({"error": f"No tour found with id {tour_id}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)


LANGCHAIN_TOOLS = [search_knowledge, search_tours, get_tour_detail]
