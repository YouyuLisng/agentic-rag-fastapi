"""Compares open-weight Ollama models against Claude Haiku on the same
generation task -- given identical retrieved context (from the real
search_knowledge, unchanged), generate a Traditional Chinese answer,
then score every candidate with the same fixed judge (Claude Haiku)
for a fair, apples-to-apples comparison. Also measures wall-clock
latency per model.

Deliberately does NOT test agentic tool-calling through Ollama --
open-model function-calling reliability varies a lot model to model
and is a different, larger evaluation than what's being measured here
(繁中回答品質、推論速度、RAG 效果, per the JD language this was written
against). Retrieval stays identical across every candidate; only the
generation step swaps models, so any quality/speed difference is
attributable to the model, not to different context.

Models picked for an 8GB-RAM Apple M1 (this machine): small enough to
run without swapping, spanning two families (Qwen vs Llama) and two
sizes within the Qwen family, for a comparison with some actual
variance in it. A machine with more RAM could reasonably run larger
(7B+) models instead.

Requires a running local Ollama server (`ollama serve`) with the
models below pulled:
    ollama pull qwen2.5:3b
    ollama pull qwen2.5:1.5b
    ollama pull llama3.2:3b

Run: `uv run python -m app.scripts.model_comparison`
"""

import asyncio
import time
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.db import close_pool, init_pool  # noqa: E402
from app.eval.generation_dataset import GENERATION_EVAL_QUERIES  # noqa: E402
from app.eval.judge import score_answer_relevancy, score_faithfulness  # noqa: E402
from app.rag.retrieval import search_knowledge  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODELS = ["qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b"]
CLAUDE_BASELINE = "claude-haiku (baseline)"

# Curated set of common Simplified-only character forms (their
# Traditional counterpart is a different codepoint) -- not an exhaustive
# converter, just enough everyday/travel-policy vocabulary (insurance,
# visa, passport, refund terms included, since that's what this
# project's answers are actually about) to catch the thing manual
# spot-checking surfaced: some open models answer in Simplified even
# when explicitly told 用繁體中文回答. Faithfulness/Answer Relevancy
# don't see this at all (a fluent Simplified answer scores fine on
# both), so it's tracked as its own metric.
_SIMPLIFIED_ONLY_CHARS = set(
    "国学后时间应还问现开关经长义点电语让务备变单参发号团门会车儿买卖说认边龙"
    "华灵归图圆园卫双达础导为个来这对样种从见医药险费员证签护检疗缓优购汇银"
    "账询议择补贴与网络线联"
)


def _simplified_char_count(text: str) -> int:
    return sum(1 for ch in text if ch in _SIMPLIFIED_ONLY_CHARS)

ANSWER_PROMPT = """你是旅行社的客服助理,請根據以下檢索到的政策文件內容,用繁體中文回答使用者的問題。
只根據提供的資料回答,不要編造資料中沒有的內容;如果資料不足以回答,請誠實說明。

參考資料:
{context}

使用者問題:{query}"""


async def _generate_ollama(model: str, prompt: str) -> tuple[str, float]:
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False})
        resp.raise_for_status()
        data = resp.json()
    return data["response"], time.monotonic() - start


async def _generate_claude(client: AsyncAnthropic, model: str, prompt: str) -> tuple[str, float]:
    start = time.monotonic()
    response = await client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, time.monotonic() - start


async def main() -> None:
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    judge_model = settings.claude_model_fast  # fixed judge across every candidate for a fair comparison

    await init_pool()
    try:
        candidates = [CLAUDE_BASELINE, *OLLAMA_MODELS]
        results: dict[str, list[dict[str, Any]]] = {c: [] for c in candidates}

        for query in GENERATION_EVAL_QUERIES:
            hits = await search_knowledge(query, match_count=5)
            context = [h["content"] for h in hits]
            prompt = ANSWER_PROMPT.format(context="\n\n---\n\n".join(context), query=query)

            for candidate in candidates:
                if candidate == CLAUDE_BASELINE:
                    answer, elapsed = await _generate_claude(client, settings.claude_model_fast, prompt)
                else:
                    answer, elapsed = await _generate_ollama(candidate, prompt)

                faithfulness, _ = await score_faithfulness(client, judge_model, answer, context)
                relevancy, _, _ = await score_answer_relevancy(client, judge_model, query, answer)
                simplified_chars = _simplified_char_count(answer)

                results[candidate].append(
                    {
                        "latency_s": elapsed,
                        "faithfulness": faithfulness,
                        "answer_relevancy": relevancy,
                        "simplified_chars": simplified_chars,
                    }
                )
                flag = f" ⚠️ {simplified_chars} 個簡體字" if simplified_chars > 0 else ""
                print(
                    f"[{candidate}] {query[:24]}... {elapsed:5.1f}s  F={faithfulness:.2f}  R={relevancy:.2f}{flag}"
                )

        print()
        header = f"{'模型':<26}{'平均延遲(秒)':<16}{'平均 Faithfulness':<20}{'平均 Relevancy':<16}{'簡體字案例數'}"
        print(header)
        print("-" * 96)
        for candidate, cases in results.items():
            n = len(cases)
            avg_latency = sum(c["latency_s"] for c in cases) / n
            avg_faith = sum(c["faithfulness"] for c in cases) / n
            avg_rel = sum(c["answer_relevancy"] for c in cases) / n
            simplified_cases = sum(1 for c in cases if c["simplified_chars"] > 0)
            print(
                f"{candidate:<26}{avg_latency:<16.2f}{avg_faith:<20.3f}{avg_rel:<16.3f}{simplified_cases}/{n}"
            )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
