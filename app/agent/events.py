from typing import Any, Literal

from pydantic import BaseModel


class ToolCallEvent(BaseModel):
    """Model decided to call a tool -- the moment worth visualizing."""

    type: Literal["tool_call"] = "tool_call"
    turn: int
    tool_use_id: str
    name: str
    input: dict[str, Any]


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    turn: int
    tool_use_id: str
    name: str
    result: str  # JSON string
    is_error: bool


class FinalAnswerEvent(BaseModel):
    type: Literal["final_answer"] = "final_answer"
    text: str


class RefusalEvent(BaseModel):
    type: Literal["refusal"] = "refusal"
    category: str | None
    explanation: str | None


class MaxTurnsExceededEvent(BaseModel):
    type: Literal["max_turns_exceeded"] = "max_turns_exceeded"
    text: str


class ErrorEvent(BaseModel):
    """Transport/infra failure (e.g. the Claude API call itself errored) --
    distinct from RefusalEvent, which is Claude successfully responding
    with a policy decline."""

    type: Literal["error"] = "error"
    message: str


AgentEvent = (
    ToolCallEvent | ToolResultEvent | FinalAnswerEvent | RefusalEvent | MaxTurnsExceededEvent | ErrorEvent
)
