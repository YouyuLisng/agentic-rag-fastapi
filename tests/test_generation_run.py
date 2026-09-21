from app.agent.events import FinalAnswerEvent, MaxTurnsExceededEvent, RefusalEvent, ToolCallEvent, ToolResultEvent
from app.eval.generation_run import _collect_answer_and_context


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
