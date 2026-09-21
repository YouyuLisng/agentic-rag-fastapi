"""Hallucination regression tests -- the specific case a reviewer
called out when auditing this portfolio: "問一個不存在的行程細節，Agent
該拒答而非亂編" (ask for details of a tour that doesn't exist; the
agent should honestly say so, not fabricate a plausible-sounding
itinerary).

Distinct from multi_turn_flow_test.py: these are single-turn cases
about content fabrication, not cross-turn context carrying.

Scenario 1 originally asked about a tour by an obviously fictional
name ("火星探索七日遊") and reliably failed -- but not because of
hallucination: search_tours has no name/keyword parameter (only
country/budget/days/suitable_for), so the agent correctly explained it
couldn't search by name and asked for more filters instead of guessing.
That's honest behavior given a real tool-design limitation, not a bug
-- but it meant the scenario wasn't actually exercising "asked about a
tour that doesn't exist, got an empty result, what happens next."
Replaced with a query that filters on a real, searchable combination
guaranteed to return zero rows (a valid country + an impossibly low
budget ceiling) so search_tours is reliably called and empty.

Scenario 2 reuses the Faithfulness judge (app/eval/judge.py) because
RAG has no true "zero results" case the way a SQL filter does -- vector
search always returns *something*. Running it repeatedly surfaced a
genuinely nuanced result worth keeping, not hiding behind a binary
pass/fail: on one run, the model correctly reported that responsibility
for lost/delayed baggage falls on the airline (per the retrieved
insurance.md text) -- but on another run it stated insurance "covers"
baggage loss/delay, which the source actually attributes to the
airline, not insurance. A subtle misattribution between two real,
related policies, not an invented one. Separately, the Faithfulness
judge itself flags reasonable meta-commentary ("sci-fi scenarios
aren't typically covered") as "unsupported claims" alongside genuine
factual claims, since its claim decomposition doesn't distinguish
reasoning from fact -- a real limitation of naive claim-decomposition
scoring, not of the agent. So this scenario's pass criterion checks the
one thing that actually matters (the answer must not claim the agency
offers "外星生物保險"), and reports Faithfulness as diagnostic
information rather than a hard gate on secondary claims.

Costs real money (each scenario is a full agent turn, plus a judge
call for scenario 2). Run:
`uv run python -m app.scripts.hallucination_regression_test`
"""

import asyncio
import json
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

from app.agent.events import (  # noqa: E402
    AgentEvent,
    FinalAnswerEvent,
    MaxTurnsExceededEvent,
    RefusalEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.agent.loop import run_agent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import close_pool, init_pool  # noqa: E402
from app.eval.judge import score_faithfulness  # noqa: E402


async def _collect(events: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [e async for e in events]


def _final_text(events: list[AgentEvent]) -> str:
    for e in reversed(events):
        if isinstance(e, FinalAnswerEvent):
            return e.text
        if isinstance(e, RefusalEvent):
            return e.explanation or "(拒絕回答)"
        if isinstance(e, MaxTurnsExceededEvent):
            return e.text
    return "(沒有產生回答)"


def _tool_calls(events: list[AgentEvent], name: str) -> list[ToolCallEvent]:
    return [e for e in events if isinstance(e, ToolCallEvent) and e.name == name]


def _tool_results(events: list[AgentEvent], name: str) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent) and e.name == name and not e.is_error]


async def scenario_nonexistent_tour() -> tuple[bool, str]:
    """A valid country + an impossibly low budget ceiling guarantees
    search_tours returns zero rows -- no tool call may then fabricate
    an id or itinerary for something that was never returned, and the
    final answer must say so plainly."""
    events = await _collect(run_agent("有沒有預算三千元以下、去日本的行程?請介紹詳細的每日行程"))

    search_results = _tool_results(events, "search_tours")
    if not search_results:
        return False, "沒有呼叫 search_tours 就直接回答"

    tours = json.loads(search_results[0].result)
    if tours:
        return False, f"search_tours 竟然查到符合條件的行程,測試假設有誤: {tours}"

    detail_calls = _tool_calls(events, "get_tour_detail")
    if detail_calls:
        return False, f"沒有真實 id 卻呼叫了 get_tour_detail: {detail_calls[0].input}"

    final_text = _final_text(events)
    honest_markers = ["找不到", "沒有這個", "查無", "不存在", "沒有找到", "目前沒有", "並沒有"]
    if not any(m in final_text for m in honest_markers):
        return False, f"回答沒有明確表示查無符合條件的行程,可能在硬湊答案: {final_text[:150]}"

    return True, "search_tours 正確查無結果,沒有呼叫 get_tour_detail 編造細節,且誠實告知使用者"


async def scenario_out_of_scope_policy(client: AsyncAnthropic, judge_model: str) -> tuple[bool, str]:
    """search_knowledge always returns its top-k regardless of
    relevance -- for a topic genuinely outside the knowledge base, the
    one thing that must never happen is claiming the product exists.
    Faithfulness is reported as diagnostic detail, not a hard gate --
    see the module docstring for why a lower score here isn't
    necessarily the agent's fault."""
    events = await _collect(run_agent("你們有『外星生物保險』這個保單嗎?"))
    context = [r.result for r in _tool_results(events, "search_knowledge")]
    answer = _final_text(events)

    false_offer_markers = ["有的,我們有外星", "是的,我們有外星", "我們確實提供外星"]
    if any(m in answer for m in false_offer_markers):
        return False, f"回答疑似聲稱真的有提供『外星生物保險』: {answer[:150]}"

    if not context:
        return True, "這次沒有呼叫 search_knowledge,答案本身也沒有聲稱提供這個保單(無 Faithfulness 可算)"

    score, claims = await score_faithfulness(client, judge_model, answer, context)
    unsupported = [c.claim for c in claims if not c.supported]
    detail = f"Faithfulness {score:.2f}"
    if unsupported:
        detail += f"(診斷用,非硬性門檻 -- 未受檢索資料直接支持的陳述: {unsupported})"
    return True, f"{detail};沒有聲稱提供『外星生物保險』這個產品"


async def main() -> None:
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    await init_pool()
    try:
        out_of_scope = await scenario_out_of_scope_policy(client, settings.claude_model_fast)
        results = [
            ("nonexistent_tour_no_fabrication", *await scenario_nonexistent_tour()),
            ("out_of_scope_policy_stays_grounded", *out_of_scope),
        ]
        all_passed = True
        for name, passed, detail in results:
            all_passed = all_passed and passed
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        print()
        print("全部通過" if all_passed else "有測試失敗,見上方 FAIL 項目")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
