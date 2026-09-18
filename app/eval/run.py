from pydantic import BaseModel

from app.eval.dataset import EVAL_CASES
from app.rag.retrieval import search_knowledge

DEFAULT_K = 5


class EvalCaseResult(BaseModel):
    query: str
    expected_slug: str
    retrieved_slugs: list[str]  # in rank order, best match first
    rank: int | None  # 1-indexed rank of the expected doc; None if not in top-k
    top_similarity: float


class EvalMetrics(BaseModel):
    n: int
    accuracy_at_1: float
    accuracy_at_3: float
    accuracy_at_5: float
    mrr: float  # mean reciprocal rank -- rewards ranking the right doc
    # higher even when it's not always #1, unlike plain accuracy@1


class EvalReport(BaseModel):
    metrics: EvalMetrics
    cases: list[EvalCaseResult]


async def run_retrieval_eval(k: int = DEFAULT_K) -> EvalReport:
    cases: list[EvalCaseResult] = []

    for case in EVAL_CASES:
        hits = await search_knowledge(case["query"], match_count=k)
        retrieved_slugs = [hit["document_slug"] for hit in hits]

        rank = None
        if case["expected_slug"] in retrieved_slugs:
            rank = retrieved_slugs.index(case["expected_slug"]) + 1

        cases.append(
            EvalCaseResult(
                query=case["query"],
                expected_slug=case["expected_slug"],
                retrieved_slugs=retrieved_slugs,
                rank=rank,
                top_similarity=hits[0]["similarity"] if hits else 0.0,
            )
        )

    return EvalReport(metrics=_compute_metrics(cases), cases=cases)


def _compute_metrics(cases: list[EvalCaseResult]) -> EvalMetrics:
    n = len(cases)
    return EvalMetrics(
        n=n,
        accuracy_at_1=sum(1 for c in cases if c.rank == 1) / n,
        accuracy_at_3=sum(1 for c in cases if c.rank is not None and c.rank <= 3) / n,
        accuracy_at_5=sum(1 for c in cases if c.rank is not None and c.rank <= 5) / n,
        mrr=sum((1 / c.rank) if c.rank else 0.0 for c in cases) / n,
    )
