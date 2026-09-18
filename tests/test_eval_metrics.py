from app.eval.run import EvalCaseResult, _compute_metrics


def _case(rank: int | None) -> EvalCaseResult:
    return EvalCaseResult(
        query="q",
        expected_slug="slug",
        retrieved_slugs=[],
        rank=rank,
        top_similarity=0.9 if rank else 0.0,
    )


def test_all_top_1_hits_gives_perfect_scores():
    cases = [_case(1), _case(1), _case(1)]
    m = _compute_metrics(cases)
    assert m.accuracy_at_1 == 1.0
    assert m.accuracy_at_3 == 1.0
    assert m.accuracy_at_5 == 1.0
    assert m.mrr == 1.0


def test_rank_outside_top_1_still_counts_toward_at_3_and_at_5():
    cases = [_case(1), _case(3)]
    m = _compute_metrics(cases)
    assert m.accuracy_at_1 == 0.5
    assert m.accuracy_at_3 == 1.0
    assert m.accuracy_at_5 == 1.0
    assert m.mrr == (1 + 1 / 3) / 2


def test_miss_counts_as_zero_everywhere():
    cases = [_case(1), _case(None)]
    m = _compute_metrics(cases)
    assert m.accuracy_at_1 == 0.5
    assert m.accuracy_at_3 == 0.5
    assert m.accuracy_at_5 == 0.5
    assert m.mrr == 0.5


def test_rank_beyond_5_counts_as_miss_for_at_5():
    cases = [_case(6)]
    m = _compute_metrics(cases)
    assert m.accuracy_at_1 == 0.0
    assert m.accuracy_at_3 == 0.0
    assert m.accuracy_at_5 == 0.0
    assert m.mrr == 1 / 6
