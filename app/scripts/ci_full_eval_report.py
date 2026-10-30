"""CI reporting (not hard-gating) for the two expensive eval layers --
generation (Faithfulness/Answer Relevancy, Ragas-style LLM-as-a-Judge)
and tool-selection. Each case costs a full live agent turn plus, for
generation, two more judge calls -- real API spend, and the judge's
scores carry real run-to-run variance (same model, not deterministic
like retrieval's embedding similarity) -- see README's own discussion
of the judge's claim-decomposition limitations. A hard per-point
threshold here would flake on noise, not catch real regressions, so
this only fails the job on a floor low enough to mean something is
actually broken (e.g. the agent stopped calling tools at all), and
otherwise just writes the numbers to the job summary for a human to
glance at.

Run: `uv run python -m app.scripts.ci_full_eval_report`
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.eval.generation_run import run_generation_eval  # noqa: E402
from app.eval.tool_selection_run import run_tool_selection_eval  # noqa: E402

FAITHFULNESS_FLOOR = 0.70
EXACT_MATCH_FLOOR = 0.60


async def main() -> int:
    await init_pool()
    try:
        generation, tool_selection = await asyncio.gather(run_generation_eval(), run_tool_selection_eval())
    finally:
        await close_pool()

    gm, tm = generation.metrics, tool_selection.metrics
    table = (
        "| Layer | Metric | Value | Floor |\n"
        "|---|---|---|---|\n"
        f"| Generation | Faithfulness | {gm.avg_faithfulness:.3f} | {FAITHFULNESS_FLOOR:.2f} |\n"
        f"| Generation | Answer Relevancy | {gm.avg_answer_relevancy:.3f} "
        f"(baseline {gm.avg_answer_relevancy_baseline:.3f}) | - |\n"
        f"| Tool selection | Exact-match | {tm.exact_match_accuracy:.1%} | {EXACT_MATCH_FLOOR:.0%} |\n"
        f"| Tool selection | Precision/Recall | {tm.avg_precision:.3f} / {tm.avg_recall:.3f} | - |\n"
    )
    print(table)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Generation + tool-selection eval (report-only)\n\n" + table + "\n")

    failures = []
    if gm.avg_faithfulness < FAITHFULNESS_FLOOR:
        failures.append(f"Faithfulness {gm.avg_faithfulness:.3f} < floor {FAITHFULNESS_FLOOR:.2f}")
    if tm.exact_match_accuracy < EXACT_MATCH_FLOOR:
        failures.append(f"Tool-selection exact-match {tm.exact_match_accuracy:.1%} < floor {EXACT_MATCH_FLOOR:.0%}")

    if failures:
        print("FAILED (drastic regression, not normal judge noise):")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("OK -- no drastic regression")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
