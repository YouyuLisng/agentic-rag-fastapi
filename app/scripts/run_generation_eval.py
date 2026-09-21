"""Runs the Ragas-style generation eval (Faithfulness / Answer
Relevancy) against the real agent loop, Claude judge, and Voyage
embeddings, then prints a report.

Costs real money -- each of the 8 cases runs a full agent turn plus two
judge calls. Run: `uv run python -m app.scripts.run_generation_eval`
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.eval.generation_run import run_generation_eval  # noqa: E402


async def main() -> None:
    await init_pool()
    try:
        report = await run_generation_eval()
    finally:
        await close_pool()

    for i, case in enumerate(report.cases, start=1):
        print(f"[{i}] {case.query}")
        print(f"    答案: {case.answer[:80]}{'...' if len(case.answer) > 80 else ''}")
        print(f"    Faithfulness: {case.faithfulness:.2f} ({len(case.faithfulness_claims)} claims)")
        relevancy_note = " (noncommittal)" if case.is_noncommittal else ""
        print(f"    Answer Relevancy: {case.answer_relevancy:.2f}{relevancy_note}")
        print()

    m = report.metrics
    print(f"樣本數: {m.n}")
    print(f"平均 Faithfulness: {m.avg_faithfulness:.3f}")
    print(f"平均 Answer Relevancy: {m.avg_answer_relevancy:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
