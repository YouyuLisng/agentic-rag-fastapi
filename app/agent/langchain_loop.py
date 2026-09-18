from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.events import AgentEvent, FinalAnswerEvent, ToolCallEvent, ToolResultEvent
from app.agent.langchain_tools import LANGCHAIN_TOOLS
from app.agent.loop import FINAL_ANSWER_PROMPT_SUFFIX, MAX_TOKENS, SYSTEM_PROMPT
from app.config import get_settings

_agent: Any = None
_smart_model: BaseChatModel | None = None


def _build_model(model_name: str) -> BaseChatModel:
    settings = get_settings()
    # pyright synthesizes ChatAnthropic's __init__ from its pydantic
    # fields' aliases and, even with every field defaulted, reports
    # timeout/stop as missing -- a known pydantic-plugin/pyright
    # friction point with alias-heavy models, not a real issue (this
    # constructs and runs correctly; verified against the live API).
    return ChatAnthropic(  # pyright: ignore[reportCallIssue]
        model_name=model_name,
        api_key=settings.anthropic_api_key,
        max_tokens_to_sample=MAX_TOKENS,
    )


def _get_agent() -> Any:
    """Built once and reused -- same reasoning as the Anthropic client in
    the hand-rolled loop, this holds no per-conversation state itself
    (LangGraph threads that through the invoke/astream call args, which
    we don't use here since each request is a single-turn conversation).

    Bound to the fast model only -- every tool-routing decision goes
    through it, same as the hand-rolled loop. The smart model never
    runs inside this graph; see run_agent_langchain's escalation call."""
    global _agent
    if _agent is None:
        settings = get_settings()
        model = _build_model(settings.claude_model_fast)
        _agent = create_agent(model, tools=LANGCHAIN_TOOLS, system_prompt=SYSTEM_PROMPT)
    return _agent


def _get_smart_model() -> BaseChatModel:
    global _smart_model
    if _smart_model is None:
        settings = get_settings()
        _smart_model = _build_model(settings.claude_model_smart)
    return _smart_model


def _extract_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


async def _escalate_final_answer(history: list[BaseMessage]) -> str:
    """Mirrors loop.py's _escalate_final_answer: one call to the smart
    model, no tools bound, given the accumulated history -- its only
    job is to synthesize a final answer from tool results already
    gathered by the fast model."""
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT + FINAL_ANSWER_PROMPT_SUFFIX), *history]
    response = await _get_smart_model().ainvoke(messages)
    assert isinstance(response, AIMessage)
    return _extract_text(response)


async def run_agent_langchain(user_message: str) -> AsyncIterator[AgentEvent]:
    """Same tools, same system prompt, same underlying query functions,
    same fast/smart model routing policy as run_agent() in loop.py --
    the only thing this reimplements is the loop mechanics themselves,
    using LangGraph's prebuilt tool-calling agent (create_agent)
    instead of a hand-rolled while loop. Streams the same AgentEvent
    union so the frontend needs no changes to support either backend.

    The message history is tracked manually (mirroring what the
    hand-rolled loop already does) rather than read back from
    LangGraph state, specifically so the escalation call below can
    reuse it outside the graph -- create_agent's compiled graph is
    bound to one fixed model, so switching to the smart model for the
    final answer means stepping outside it for that one call."""
    settings = get_settings()
    agent = _get_agent()
    turn = -1
    used_tool = False
    history: list[BaseMessage] = [HumanMessage(content=user_message)]

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
                    model=settings.claude_model_fast,
                )
            case "on_tool_end":
                tool_message: ToolMessage = event["data"]["output"]
                history.append(tool_message)
                yield ToolResultEvent(
                    turn=turn,
                    tool_use_id=str(event["run_id"]),
                    name=event["name"],
                    result=str(tool_message.content),
                    is_error=getattr(tool_message, "status", "success") == "error",
                )
            case "on_chat_model_end":
                ai_message: AIMessage = event["data"]["output"]
                if ai_message.tool_calls:
                    used_tool = True
                    history.append(ai_message)
                elif not used_tool:
                    yield FinalAnswerEvent(model=settings.claude_model_fast, text=_extract_text(ai_message))
                else:
                    final_text = await _escalate_final_answer(history)
                    yield FinalAnswerEvent(model=settings.claude_model_smart, text=final_text)
