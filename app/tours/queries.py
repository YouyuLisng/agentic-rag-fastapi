from typing import Any

from psycopg import sql

from app.db import get_pool


def _build_search_tours_query(
    country: str | None = None,
    max_budget_twd: int | None = None,
    min_days: int | None = None,
    max_days: int | None = None,
    suitable_for: str | None = None,
) -> tuple[sql.Composed, list[Any]]:
    """Pure query-building, split out from search_tours so the
    filter-combination logic is unit-testable without a live DB."""
    conditions: list[sql.Composable] = []
    params: list[Any] = []

    if country:
        conditions.append(sql.SQL("country = %s"))
        params.append(country)
    if max_budget_twd is not None:
        conditions.append(sql.SQL("budget_twd <= %s"))
        params.append(max_budget_twd)
    if min_days is not None:
        conditions.append(sql.SQL("days >= %s"))
        params.append(min_days)
    if max_days is not None:
        conditions.append(sql.SQL("days <= %s"))
        params.append(max_days)
    if suitable_for:
        conditions.append(sql.SQL("%s = ANY(suitable_for)"))
        params.append(suitable_for)

    where_clause = sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("TRUE")
    query = sql.SQL(
        "select id, title, country, location, days, budget_twd, suitable_for, summary "
        "from tours where {where} order by budget_twd asc"
    ).format(where=where_clause)
    return query, params


async def search_tours(
    country: str | None = None,
    max_budget_twd: int | None = None,
    min_days: int | None = None,
    max_days: int | None = None,
    suitable_for: str | None = None,
) -> list[dict[str, Any]]:
    """Structured filter over the tour catalog -- exact/range matching,
    no embeddings involved. This is the "agentic tool call" half of the
    routing story, contrasting with search_knowledge's semantic RAG."""
    query, params = _build_search_tours_query(
        country=country,
        max_budget_twd=max_budget_twd,
        min_days=min_days,
        max_days=max_days,
        suitable_for=suitable_for,
    )

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query, params)
        rows = await cur.fetchall()
        assert cur.description is not None
        columns = [desc[0] for desc in cur.description]

    return [dict(zip(columns, row, strict=True)) for row in rows]


async def get_tour_detail(tour_id: str) -> dict[str, Any] | None:
    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select id, title, country, location, days, budget_twd, suitable_for, "
            "summary, itinerary from tours where id = %s::uuid",
            (tour_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        assert cur.description is not None
        columns = [desc[0] for desc in cur.description]

    return dict(zip(columns, row, strict=True))


async def check_availability(tour_id: str) -> dict[str, Any] | None:
    """Real-time fact lookup, not a search or a description -- distinct
    from search_tours (filtering) and get_tour_detail (static content):
    this answers "is there still room on this specific tour" the way a
    real booking system would, off enrolled_count/capacity."""
    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select id, title, capacity, enrolled_count from tours where id = %s::uuid",
            (tour_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        tour_id_val, title, capacity, enrolled_count = row

    remaining = capacity - enrolled_count
    return {
        "id": tour_id_val,
        "title": title,
        "capacity": capacity,
        "enrolled_count": enrolled_count,
        "remaining": remaining,
        "is_available": remaining > 0,
    }
