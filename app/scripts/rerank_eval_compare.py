"""Before/after for the two-stage reranker: runs the whole retrieval
eval set (app/eval/dataset.py) once with reranking off (pure vector) and
once with it on (vector -> Voyage cross-encoder), and diffs the metrics
plus every case whose rank changed.

Honest expectation, stated up front: on this 8-document knowledge base
the first stage already ceilings (accuracy@5 = 100%, MRR ~0.97), so the
aggregate barely moves -- there's almost no headroom for a reranker to
win back. The point of this script is not a fake +X%; it's to (a) show
the two-stage pipeline works and is measured, and (b) surface the one
place rerank actually earns its keep here: the 疫情取消 -> force-majeure
near-miss, a semantically-near distractor (force-majeure 0.513 vs
cancellation 0.522 on pure vector) that a cross-encoder reading
query+document together can separate where embedding cosine can't. That
same case is fixed a different way by the metadata pre-filter
(app/scripts/metadata_filter_demo.py); rerank fixes it without the
caller needing to know the topic bucket in advance.

Run: `uv run python -m app.scripts.rerank_eval_compare`
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.eval.run import run_retrieval_eval  # noqa: E402

FORCE_MAJEURE_QUERY = "因為疫情取消行程,費用會退嗎?"


async def main() -> None:
    await init_pool()
    try:
        before = await run_retrieval_eval(rerank=False)
        after = await run_retrieval_eval(rerank=True)
    finally:
        await close_pool()

    bm, am = before.metrics, after.metrics
    print(f"{'指標':<18}{'rerank off':>12}{'rerank on':>12}{'Δ':>10}")
    print("-" * 52)
    for label, b, a in [
        ("Accuracy@1", bm.accuracy_at_1, am.accuracy_at_1),
        ("Accuracy@3", bm.accuracy_at_3, am.accuracy_at_3),
        ("Accuracy@5", bm.accuracy_at_5, am.accuracy_at_5),
        ("MRR", bm.mrr, am.mrr),
    ]:
        print(f"{label:<18}{b:>12.3f}{a:>12.3f}{a - b:>+10.3f}")

    print("\n排名有變化的 case:")
    before_by_query = {c.query: c for c in before.cases}
    any_change = False
    for ac in after.cases:
        bc = before_by_query[ac.query]
        if bc.rank != ac.rank:
            any_change = True
            print(f"  {ac.query[:30]:<32} {bc.rank} -> {ac.rank}  (預期 {ac.expected_slug})")
    if not any_change:
        print("  (無 -- 第一階段在 8 篇文件上已幾乎見頂,符合預期)")

    fm_before = before_by_query.get(FORCE_MAJEURE_QUERY)
    fm_after = next((c for c in after.cases if c.query == FORCE_MAJEURE_QUERY), None)
    if fm_before and fm_after:
        print(
            f"\n焦點 case(疫情取消 -> force-majeure): "
            f"rerank off 排名 {fm_before.rank} / rerank on 排名 {fm_after.rank}"
        )


if __name__ == "__main__":
    asyncio.run(main())
