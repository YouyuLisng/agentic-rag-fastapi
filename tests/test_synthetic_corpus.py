"""The synthetic benchmark corpus is generated deterministically, so its
structural invariants are unit-testable without a DB: stable size, unique
slugs, and every eval case pointing at a doc that actually exists."""

from app.eval.synthetic_corpus import generate_corpus, generate_eval_cases


def test_corpus_has_unique_slugs():
    docs = generate_corpus()
    slugs = [d.slug for d in docs]
    assert len(slugs) == len(set(slugs))


def test_corpus_is_large_enough_to_challenge_first_stage():
    # The whole point is scale with near-duplicates -- a handful of docs
    # wouldn't stress the first stage. Guard against the families being
    # gutted.
    assert len(generate_corpus()) >= 120


def test_each_family_has_near_duplicate_siblings():
    docs = generate_corpus()
    by_cluster: dict[str, int] = {}
    for d in docs:
        by_cluster[d.cluster] = by_cluster.get(d.cluster, 0) + 1
    # Rerank only earns its keep against siblings; every cluster must have
    # many of them.
    assert all(count >= 10 for count in by_cluster.values())


def test_eval_cases_point_at_real_docs():
    slugs = {d.slug for d in generate_corpus()}
    cases = generate_eval_cases()
    assert cases
    for case in cases:
        assert case.expected_slug in slugs


def test_generation_is_deterministic():
    assert [d.slug for d in generate_corpus()] == [d.slug for d in generate_corpus()]
    assert [c.query for c in generate_eval_cases()] == [c.query for c in generate_eval_cases()]
