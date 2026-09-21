from typing import Literal

from pydantic import BaseModel


class HistoryMessage(BaseModel):
    """One prior turn's final text. The backend is stateless -- no
    session storage, no conversation_id -- so the client resends its
    own turn list on every request and this is how the agent gets
    conversational context back. Deliberately just role+text, not the
    full tool_call/tool_result trail: continuity only needs "what was
    said", not a replay of how the previous answer was produced."""

    role: Literal["user", "assistant"]
    text: str
