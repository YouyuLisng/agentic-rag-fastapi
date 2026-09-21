from app.eval.tool_selection_run import score_tool_selection


def test_exact_match_when_sets_equal():
    exact, precision, recall = score_tool_selection({"search_tours"}, {"search_tours"})
    assert exact is True
    assert precision == 1.0
    assert recall == 1.0


def test_no_tools_expected_and_none_called_is_exact_match():
    exact, precision, recall = score_tool_selection(set(), set())
    assert exact is True
    assert precision == 1.0
    assert recall == 1.0


def test_extra_unnecessary_tool_hurts_precision_not_recall():
    exact, precision, recall = score_tool_selection({"search_tours"}, {"search_tours", "search_knowledge"})
    assert exact is False
    assert precision == 0.5
    assert recall == 1.0


def test_missing_expected_tool_hurts_recall_not_precision():
    exact, precision, recall = score_tool_selection({"search_tours", "search_knowledge"}, {"search_tours"})
    assert exact is False
    assert precision == 1.0
    assert recall == 0.5


def test_called_a_tool_when_none_expected_scores_zero():
    exact, precision, recall = score_tool_selection(set(), {"search_tours"})
    assert exact is False
    assert precision == 0.0
    assert recall == 0.0


def test_expected_a_tool_but_called_none_scores_zero():
    exact, precision, recall = score_tool_selection({"search_tours"}, set())
    assert exact is False
    assert precision == 0.0
    assert recall == 0.0


def test_completely_disjoint_tool_sets_scores_zero():
    exact, precision, recall = score_tool_selection({"search_tours"}, {"search_knowledge"})
    assert exact is False
    assert precision == 0.0
    assert recall == 0.0
