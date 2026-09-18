from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.events import AgentEvent, FinalAnswerEvent, ToolCallEvent, ToolResultEvent
from app.agent.langchain_tools import LANGCHAIN_TOOLS
from app.agent.loop import MAX_TOKENS, SYSTEM_PROMPT
from app.config import get_settings

_agent: Any = None


def _get_agent() -> Any:
    """Built once and reused -- same reasoning as the Anthropic client in
    the hand-rolled loop, this holds no per-conversation state itself
    (LangGraph threads that through the invoke/astream call args, which
    we don't use here since each request is a single-turn conversation)."""
    global _agent
    if _agent is None:
        settings = get_settings()
        # pyright synthesizes ChatAnthropic's __init__ from its pydantic
        # fields' aliases and, even with every field defaulted, reports
        # timeout/stop as missing -- a known pydantic-plugin/pyright
        # friction point with alias-heavy models, not a real issue (this
        # constructs and runs correctly; verified against the live API).
        model: BaseChatModel = ChatAnthropic(  # pyright: ignore[reportCallIssue]
            model_name=settings.claude_model,
            api_key=settings.anthropic_api_key,
            max_tokens_to_sample=MAX_TOKENS,
        )
        _agent = create_agent(model, tools=LANGCHAIN_TOOLS, system_prompt=SYSTEM_PROMPT)
    return _agent


def _extract_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


async def run_agent_langchain(user_message: str) -> AsyncIterator[AgentEvent]:
    """Same three tools, same system prompt, same underlying query
    functions as run_agent() in loop.py -- the only thing this
    reimplements is the loop mechanics themselves, using LangGraph's
    prebuilt tool-calling agent (create_agent) instead of a hand-rolled
    while loop. Streams the same AgentEvent union so the frontend needs
    no changes to support either backend."""
    agent = _get_agent()
    turn = -1

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": user_message}]},
        version="v2",
    ):
        match event["event"]:
            case "on_chat_model_start":
                turn += 1
            case "on_tool_start":
                yield ToolCallEvent(
                    turn=turn,
                    tool_use_id=str(event["run_id"]),
                    name=event["name"],
                    input=event["data"].get("input", {}),
                )
            case "on_tool_end":
                tool_message: ToolMessage = event["data"]["output"]
                yield ToolResultEvent(
                    turn=turn,
                    tool_use_id=str(event["run_id"]),
                    name=event["name"],
                    result=str(tool_message.content),
                    is_error=getattr(tool_message, "status", "success") == "error",
                )
            case "on_chat_model_end":
                ai_message: AIMessage = event["data"]["output"]
                if not ai_message.tool_calls:
                    yield FinalAnswerEvent(text=_extract_text(ai_message))
