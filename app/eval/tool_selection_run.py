from pydantic import BaseModel

from app.agent.events import ToolCallEvent
from app.agent.loop import run_agent
from app.eval.tool_selection_dataset import TOOL_SELECTION_CASES


class ToolSelectionCaseResult(BaseModel):
    query: str
    expected_tools: list[str]
    actual_tools: list[str]
    exact_match: bool
    precision: float
    recall: float


class ToolSelectionMetrics(BaseModel):
    n: int
    exact_match_accuracy: float
    avg_precision: float
    avg_recall: float


class ToolSelectionReport(BaseModel):
    metrics: ToolSelectionMetrics
    cases: list[ToolSelectionCaseResult]


def score_tool_selection(expected: set[str], actual: set[str]) -> tuple[bool, float, float]:
    """Pure so the scoring math is unit-testable without a live agent
    run. Precision/recall give partial credit beyond binary exact-match
    -- e.g. calling one extra unnecessary tool still shows up as
    "recall was fine, precision took a hit" rather than just "wrong."
    """
    exact_match = expected == actual
    if not actual:
        precision = 1.0 if not expected else 0.0
    else:
        precision = len(expected & actual) / len(actual)
    if not expected:
        recall = 1.0 if not actual else 0.0
    else:
        recall = len(expected & actual) / len(expected)
    return exact_match, precision, recall


async def _run_case(case: dict[str, object]) -> ToolSelectionCaseResult:
    query = str(case["query"])
    expected = set(case["expected_tools"])  # type: ignore[arg-type]

    actual: set[str] = set()
    async for event in run_agent(query):
        if isinstance(event, ToolCallEvent):
            actual.add(event.name)

    exact_match, precision, recall = score_tool_selection(expected, actual)
    return ToolSelectionCaseResult(
        query=query,
        expected_tools=sorted(expected),
        actual_tools=sorted(actual),
        exact_match=exact_match,
        precision=precision,
        recall=recall,
    )


async def run_tool_selection_eval() -> ToolSelectionReport:
    cases = [await _run_case(case) for case in TOOL_SELECTION_CASES]

    n = len(cases)
    metrics = ToolSelectionMetrics(
        n=n,
        exact_match_accuracy=sum(1 for c in cases if c.exact_match) / n,
        avg_precision=sum(c.precision for c in cases) / n,
        avg_recall=sum(c.recall for c in cases) / n,
    )
    return ToolSelectionReport(metrics=metrics, cases=cases)
