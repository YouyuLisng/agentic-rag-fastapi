import pytest

from app.agent.events import FinalAnswerEvent, MaxTurnsExceededEvent, RefusalEvent, ToolCallEvent, ToolResultEvent
from app.eval import generation_run as generation_run_module
from app.eval.generation_run import GenerationEvalCase, _collect_answer_and_context, _compute_relevancy_baseline


def _case(query: str, questions: list[str]) -> GenerationEvalCase:
    return GenerationEvalCase(
        query=query,
        answer="",
        context=[],
        faithfulness=1.0,
        faithfulness_claims=[],
        answer_relevancy=1.0,
        relevancy_questions=questions,
        is_noncommittal=False,
    )


def test_collects_final_answer_and_non_error_tool_results():
    events = [
        ToolCallEvent(turn=0, tool_use_id="t1", name="search_knowledge", input={}, model="m"),
        ToolResultEvent(turn=0, tool_use_id="t1", name="search_knowledge", result="chunk-a", is_error=False),
        ToolResultEvent(turn=0, tool_use_id="t2", name="search_knowledge", result="broken", is_error=True),
        FinalAnswerEvent(text="這是最終回答", model="m"),
    ]

    answer, context = _collect_answer_and_context(events)

    assert answer == "這是最終回答"
    assert context == ["chunk-a"]


def test_refusal_uses_explanation_as_answer():
    events = [RefusalEvent(category="policy", explanation="無法協助這個請求")]

    answer, context = _collect_answer_and_context(events)

    assert answer == "無法協助這個請求"
    assert context == []


def test_refusal_without_explanation_falls_back_to_placeholder():
    events = [RefusalEvent(category=None, explanation=None)]

    answer, _ = _collect_answer_and_context(events)

    assert answer == "(拒絕回答)"


def test_max_turns_exceeded_uses_its_text():
    events = [MaxTurnsExceededEvent(text="查詢步驟過多")]

    answer, _ = _collect_answer_and_context(events)

    assert answer == "查詢步驟過多"


def test_no_terminal_event_falls_back_to_placeholder():
    events = [ToolResultEvent(turn=0, tool_use_id="t1", name="x", result="r", is_error=False)]

    answer, context = _collect_answer_and_context(events)

    assert answer == "(沒有產生回答)"
    assert context == ["r"]


async def test_relevancy_baseline_pairs_each_case_with_the_next_ones_query(monkeypatch: pytest.MonkeyPatch):
    # Circular shift by one: case A's generated question is scored
    # against case B's original query (not its own), and vice versa.
    cases = [_case("qA", ["genA"]), _case("qB", ["genB"])]

    embeddings = {
        "qA": [1.0, 0.0],
        "qB": [0.0, 1.0],
        "genA": [1.0, 0.0],  # paired against qB -> orthogonal -> 0.0
        "genB": [1.0, 0.0],  # paired against qA -> identical -> 1.0
    }

    async def fake_embed(text: str):
        return embeddings[text]

    monkeypatch.setattr(generation_run_module, "embed_query", fake_embed)

    baseline = await _compute_relevancy_baseline(cases)

    assert baseline == pytest.approx(0.5)


async def test_relevancy_baseline_is_zero_with_fewer_than_two_scorable_cases():
    assert await _compute_relevancy_baseline([_case("qA", [])]) == 0.0
