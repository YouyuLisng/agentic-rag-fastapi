from langchain_core.messages import AIMessage, HumanMessage

from app.agent.history import HistoryMessage
from app.agent.langchain_loop import _history_to_base_messages
from app.agent.loop import _history_to_messages


def test_history_to_messages_empty_when_none():
    assert _history_to_messages(None) == []


def test_history_to_messages_empty_when_no_prior_turns():
    assert _history_to_messages([]) == []


def test_history_to_messages_preserves_role_and_order():
    history = [
        HistoryMessage(role="user", text="有沒有峇里島的團?"),
        HistoryMessage(role="assistant", text="有的,峇里島蜜月五日遊..."),
    ]
    assert _history_to_messages(history) == [
        {"role": "user", "content": "有沒有峇里島的團?"},
        {"role": "assistant", "content": "有的,峇里島蜜月五日遊..."},
    ]


def test_history_to_base_messages_maps_user_to_human_and_assistant_to_ai():
    history = [
        HistoryMessage(role="user", text="這團還有位子嗎?"),
        HistoryMessage(role="assistant", text="還有 2 個名額。"),
    ]
    result = _history_to_base_messages(history)

    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "這團還有位子嗎?"
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "還有 2 個名額。"


def test_history_to_base_messages_empty_when_none():
    assert _history_to_base_messages(None) == []
