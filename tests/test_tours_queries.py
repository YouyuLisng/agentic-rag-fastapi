from app.tours.queries import _build_search_tours_query


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
