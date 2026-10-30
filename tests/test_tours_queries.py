from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from app.tours.queries import _build_count_tours_query, _build_search_tours_query, search_tours


def test_no_filters_produces_no_where_conditions():
    query, params = _build_search_tours_query()
    assert params == []
    assert "TRUE" in query.as_string(None)


def test_single_filter_produces_matching_param():
    query, params = _build_search_tours_query(country="日本")
    assert params == ["日本"]
    assert "country = %s" in query.as_string(None)


def test_all_filters_combine_with_and_in_declared_order():
    query, params = _build_search_tours_query(
        country="泰國",
        max_budget_twd=20000,
        min_days=3,
        max_days=7,
        suitable_for="長輩",
    )
    assert params == ["泰國", 20000, 3, 7, "長輩"]
    sql_text = query.as_string(None)
    assert "country = %s AND budget_twd <= %s AND days >= %s AND days <= %s AND %s = ANY(suitable_for)" in sql_text


def test_falsy_but_not_none_values_are_still_honored():
    # min_days=0 is a legitimate filter value, not "unset" -- only None
    # should be treated as "omit this filter".
    query, params = _build_search_tours_query(min_days=0)
    assert params == [0]
    assert "days >= %s" in query.as_string(None)


def test_empty_string_filters_are_treated_as_omitted():
    query, params = _build_search_tours_query(country="", suitable_for="")
    assert params == []
    assert "TRUE" in query.as_string(None)


def test_limit_appends_limit_clause_and_param():
    query, params = _build_search_tours_query(country="日本", limit=10)
    assert params == ["日本", 10]
    assert "limit %s" in query.as_string(None)


def test_no_limit_omits_limit_clause():
    query, params = _build_search_tours_query(country="日本")
    assert params == ["日本"]
    assert "limit" not in query.as_string(None)


def test_count_query_counts_with_same_where():
    query, params = _build_count_tours_query(country="泰國", max_budget_twd=20000)
    sql_text = query.as_string(None)
    assert params == ["泰國", 20000]
    assert "count(*)" in sql_text
    assert "country = %s AND budget_twd <= %s" in sql_text


def _mock_pool(total: int, rows: list[tuple]):
    """A psycopg-pool stand-in: count query returns (total,), then the row
    query returns `rows`. cur.description drives the column names."""
    cur = MagicMock()
    cur.execute = AsyncMock()
    cur.fetchone = AsyncMock(return_value=(total,))
    cur.fetchall = AsyncMock(return_value=rows)
    cur.description = [
        ("id",), ("title",), ("country",), ("location",),
        ("days",), ("budget_twd",), ("suitable_for",), ("summary",),
    ]

    @asynccontextmanager
    async def _cursor():
        yield cur

    conn = MagicMock()
    conn.cursor = _cursor

    @asynccontextmanager
    async def _connection():
        yield conn

    pool = MagicMock()
    pool.connection = _connection
    return pool


async def test_search_tours_reports_total_above_returned():
    # 25 total matches, but the row query (capped) returns only 2.
    rows = [
        ("i1", "t1", "日本", "東京", 5, 30000, ["一般大眾"], "s1"),
        ("i2", "t2", "日本", "大阪", 6, 35000, ["一般大眾"], "s2"),
    ]
    with patch("app.tours.queries.get_pool", return_value=_mock_pool(25, rows)):
        result = await search_tours(country="日本", limit=2)

    assert result["total"] == 25
    assert result["returned"] == 2
    assert result["limit"] == 2
    assert len(result["tours"]) == 2
    assert result["tours"][0]["title"] == "t1"
