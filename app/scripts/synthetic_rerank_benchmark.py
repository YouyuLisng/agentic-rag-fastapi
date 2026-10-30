"""The scale benchmark the 8-document eval set can't be: runs the
synthetic eval set (app/eval/synthetic_corpus.py) against ~160 docs of
near-duplicate distractors, with reranking off vs on, and reports the
before/after. This is where a cross-encoder reranker is supposed to
actually win -- the first stage has real competition here, unlike the
real knowledge base where it already ceilings.

Prereq: seed the synthetic table first --
    uv run python -m app.scripts.seed_synthetic
Then:
    uv run python -m app.scripts.synthetic_rerank_benchmark

Honest note: this measures retrieval on *synthetic* data, so the absolute
numbers describe the benchmark, not the product. What's meaningful is the
*delta* -- how much reranking recovers when the first stage is genuinely
challenged -- and whether it holds up per cluster.
"""

import asyncio
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.eval.synthetic_corpus import generate_eval_cases  # noqa: E402
from app.eval.synthetic_retrieval import search_synthetic  # noqa: E402

K = 5


async def _score(rerank: bool) -> tuple[dict[str, float], dict[str, tuple[int, int]]]:
    cases = generate_eval_cases()
    ranks: list[int | None] = []
    per_cluster: dict[str, list[int | None]] = defaultdict(list)

    for case in cases:
        hits = await search_synthetic(case.query, match_count=K, rerank=rerank)
        slugs = [h["document_slug"] for h in hits]
        rank = slugs.index(case.expected_slug) + 1 if case.expected_slug in slugs else None
        ranks.append(rank)
        per_cluster[case.cluster].append(rank)

    n = len(ranks)
    metrics = {
        "accuracy@1": sum(1 for r in ranks if r == 1) / n,
        "accuracy@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
        "accuracy@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "mrr": sum((1 / r) if r else 0.0 for r in ranks) / n,
    }
    cluster_acc1 = {
        c: (sum(1 for r in rs if r == 1), len(rs)) for c, rs in per_cluster.items()
    }
    return metrics, cluster_acc1


async def main() -> None:
    await init_pool()
    try:
        before, before_clusters = await _score(rerank=False)
        after, after_clusters = await _score(rerank=True)
    finally:
        await close_pool()

    n = len(generate_eval_cases())
    print(f"Synthetic scale benchmark -- {n} queries over ~160 near-duplicate docs\n")
    print(f"{'指標':<14}{'rerank off':>12}{'rerank on':>12}{'Δ':>10}")
    print("-" * 48)
    for key in ["accuracy@1", "accuracy@3", "accuracy@5", "mrr"]:
        b, a = before[key], after[key]
        print(f"{key:<14}{b:>12.3f}{a:>12.3f}{a - b:>+10.3f}")

    print("\naccuracy@1 per cluster (off -> on):")
    for cluster in sorted(before_clusters):
        bh, bn = before_clusters[cluster]
        ah, _an = after_clusters[cluster]
        print(f"  {cluster:<16} {bh}/{bn} -> {ah}/{bn}")


if __name__ == "__main__":
    asyncio.run(main())
