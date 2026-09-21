# Agentic RAG Travel Assistant

A multi-step agentic loop (Claude tool use) that routes between
semantic retrieval (RAG over a small travel-policy knowledge base) and
structured tool calls (querying a tour catalog), deciding autonomously
which path -- or how many steps -- a question needs. Implemented
twice: once hand-rolled (no framework, to demonstrate the mechanics),
once with `langchain`/`langgraph` (to match what the target job
postings actually ask for) -- same tools, same prompt, same underlying
queries, so the two are a direct, fair comparison rather than one
being a strawman for the other.

## Stack

FastAPI, Claude (Anthropic API), Voyage embeddings, Postgres + pgvector
(Supabase), Next.js frontend (separate project) consuming this API over
SSE for streamed decision visualization. Both implementations sit
behind the same `POST /chat` endpoint, selected per-request via an
`impl` field.

## Architecture

The model decides per-turn whether it needs a tool at all, and if so,
which one -- routing is driven entirely by the tool descriptions and
system prompt, not hardcoded branching. Policy questions go through
semantic RAG (embedding + pgvector similarity, since policy text has
no fixed structure); tour questions go through structured SQL
(`tours` table has real columns to filter on, so no embedding is
needed). Both paths are tool calls the model chooses between in the
same loop, and it can call several -- sequentially or in parallel --
before producing a final answer.

```mermaid
flowchart TD
    A[使用者問題] --> B[Agent Loop: 呼叫 Claude]
    B --> C[模型理解語意]
    C --> D{需要呼叫工具嗎?}
    D -->|不需要,例如打招呼| ZFinal[直接輸出回答]

    D -->|語意/政策類問題| E[呼叫 search_knowledge]
    E --> E1[Voyage embed 把問題轉成向量]
    E1 --> E2[pgvector 餘弦相似度檢索]
    E2 --> E3[回傳 top-k 政策片段]
    E3 --> Merge[工具結果塞回 messages]

    D -->|尋找/篩選行程| F[呼叫 search_tours]
    F --> F1[模型從問題萃取條件<br/>地區/預算/天數/適合對象]
    F1 --> F2["SQL 查詢 tours 表<br/>(目前:8 筆靜態種子資料)"]
    F2 --> F3[回傳符合條件的行程列表]
    F3 --> Merge

    D -->|已知某行程,要細節| G[呼叫 get_tour_detail]
    G --> G1[依 tour_id 查 tours 表]
    G1 --> G2[回傳完整行程含每日細節]
    G2 --> Merge

    D -->|問還有沒有位子/最新報價| H[呼叫 check_availability]
    H --> H1["即時 SQL 查詢 tours 表<br/>capacity - enrolled_count"]
    H1 --> H2[回傳剩餘名額 + 目前報價]
    H2 --> Merge

    Merge --> B
    B --> C2{這輪 stop_reason}
    C2 -->|還要再查| D
    C2 -->|end_turn| ZFinal2[輸出最終答案]
```

Every tour-path tool (`search_tours`/`get_tour_detail`/`check_availability`)
queries a real Postgres table with real SQL, live, per request -- there
is no cached/precomputed answer anywhere in this path. `check_availability`
specifically exists to demonstrate that this isn't a static FAQ bot:
"這團還有位子嗎" always re-reads `capacity - enrolled_count` from the
database at the moment it's asked, not a number baked into the prompt
or a snapshot from ingestion time. The table happens to be seeded from
8 hand-written rows (`data/tours.json`) rather than a production
inventory system, but that's a data-volume difference, not an
architectural one -- swapping in a real product database wouldn't
change this diagram at all, only what's behind `tours`.

### Data isolation (internal cost price never reaches the model)

`tours.cost_price_twd` (internal floor/cost price, as opposed to
`budget_twd`, the customer-facing sell price) is deliberately never
selected by `search_tours`/`get_tour_detail`/`check_availability`, and
never appears in any router's response model. This is enforced at the
data layer, not the prompt layer: the column simply isn't in any SELECT
the agent's tools can run, so there's no channel for the model to leak
it through regardless of how a user phrases the question -- a system
prompt clause telling the model not to disclose cost data (in
`app/agent/loop.py`'s `SYSTEM_PROMPT`) is a second line of defense, not
the primary one, since prompts alone can be argued or injected around.

Verified live, not just by code review: `app/scripts/security_probe.py`
sends both implementations a handful of adversarial prompts (direct
asks, "estimate from the sell price" framing, and a prompt-injection
attempt telling the model to ignore its instructions and act as an
internal finance system) and greps every answer for the actual known
`cost_price_twd` values from the seed data -- an exact-number match, not
a keyword heuristic. Run it yourself: `uv run python -m
app.scripts.security_probe` (costs real API calls, one full agent turn
per prompt per implementation).

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, DATABASE_URL
```

Run `db/schema.sql` once against the Supabase project (SQL Editor, or
`psql "$DATABASE_URL" -f db/schema.sql`) to create the `tours` and
`policy_chunks` tables plus the `match_policy_chunks` retrieval function,
then seed it:

```bash
uv run python -m app.scripts.seed
uv run uvicorn app.main:app --reload
```

`GET /health` checks the app is up and the DB pool can reach Postgres.
`POST /chat` (`{"message": "...", "impl": "handrolled" | "langchain"}`)
streams the agent's run as SSE -- see the Next.js frontend project for
the UI that consumes it, or drive it directly from the terminal:

```bash
uv run python -m app.scripts.chat_cli "退訂政策是什麼?"
uv run python -m app.scripts.chat_cli_langchain "退訂政策是什麼?"
```

## Project layout

```
app/
  main.py           FastAPI app, lifespan-managed DB pool, CORS
  config.py         pydantic-settings (env vars)
  db.py             psycopg async connection pool
  routers/
    chat.py         POST /chat -- SSE stream, picks handrolled/langchain by impl
  agent/
    loop.py          hand-rolled agentic loop (no framework)
    tools.py         tool registry + dispatch for the hand-rolled loop
    events.py        the AgentEvent union both implementations stream
    langchain_loop.py    same agent, built with langchain.agents.create_agent
    langchain_tools.py   same 3 tools, as @tool-decorated functions
  rag/               embedding (Voyage) + chunking + pgvector retrieval
  tours/
    queries.py       structured SQL search_tours/get_tour_detail
  scripts/
    seed.py                  seeds tours.json + policies/*.md
    chat_cli.py               terminal harness, hand-rolled loop
    chat_cli_langchain.py     terminal harness, LangChain loop
db/
  schema.sql        pgvector schema + match_policy_chunks function
data/
  tours.json        8 seed tours (structured fields + itinerary)
  policies/         8 hand-written policy documents (source for RAG)
tests/
```

## Testing

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Evaluation (three layers, not just "it seemed to work")

Unit tests (above) verify pure logic with everything mocked -- they
can't tell you whether the *model* is actually good at this job. That
needs live evaluation against the real Claude/Voyage APIs, split into
three layers that each catch a different failure mode:

| Layer | Endpoint / script | What it catches | Live result |
|---|---|---|---|
| **Retrieval** | `GET /eval/retrieval`, `app/scripts/run_eval.py` | Wrong document ranked first even though the right one exists | Accuracy@1 93.8%, @3/@5 100%, MRR 0.969 (16 cases) |
| **Generation** | `GET /eval/generation`, `app/scripts/run_generation_eval.py` | Hallucination (Faithfulness) and answering the wrong question (Answer Relevancy) -- Ragas-style LLM-as-a-Judge, Haiku as judge | Faithfulness 1.00, Answer Relevancy 0.47-0.78 (8 cases) |
| **Tool selection** | `GET /eval/tool-selection`, dataset in `app/eval/tool_selection_dataset.py` | Routing mistakes: wrong tool, missing tool, or an unnecessary tool call the first two layers can't see | Exact-match 19/19, precision/recall 1.0 (19 cases) |

Plus two live regression suites that target specific failure modes by name rather than a general score:

- `app/scripts/multi_turn_flow_test.py` -- does context survive across
  separate turns (tour_id resolution, filter pivots after an empty
  result, RBAC holding on a follow-up)
- `app/scripts/hallucination_regression_test.py` -- given a query with
  no real answer (a nonexistent tour, a fictional insurance product),
  does the agent say so honestly instead of fabricating one

Retrieval eval auto-runs on the `/eval` page load (cheap: pure
embedding calls). Generation, tool-selection, and both regression
suites all cost a full agent turn (or several) per case, so they're
manually triggered, not run on every page load or every commit.

### Debugging stories (found by testing, not by inspection)

**The tour_id hallucination bug.** Building `multi_turn_flow_test.py`
surfaced a 100%-reproducible bug: on a follow-up turn ("這團還有位子
嗎?"), the LangChain implementation invented a fake tour_id
(`"bali-honeymoon-5d"`) instead of re-querying `search_tours`, because
conversation history only replays prior turns' text, not tool state --
neither implementation actually has the real id on a later turn unless
it re-queries for it. The hand-rolled loop already did this correctly;
a `SYSTEM_PROMPT` clause fixed the LangChain side (`app/agent/loop.py`),
verified by rerunning the exact scenario: 3/3 failing before, 3/3
passing after.

**A test that failed for the wrong reason, twice.** The first version
of the "empty result, then pivot" scenario asked "有沒有非洲的行程"
and failed on *both* implementations -- for two different, both-
legitimate reasons. Hand-rolled filtered by `country="非洲"` (an
invalid enum value) and correctly got zero rows. LangChain instead
fetched all 8 tours unfiltered and reasoned "none of these are Africa"
from the full list -- a non-empty tool result, but still an honest
final answer. Neither was a bug; the test's assertion (search_tours
must return `[]`) was too strict about *how* the agent reaches an
honest answer. Replaced with a budget-ceiling case where no filtering
strategy can produce a false match.

**A hallucination test that also failed for the wrong reason.** The
original nonexistent-tour scenario asked about a tour by an obviously
fictional name ("火星探索七日遊") and reliably "failed" -- but
`search_tours` has no name/keyword parameter (only country/budget/
days/suitable_for), so the agent correctly explained it couldn't
search by name instead of guessing. Honest behavior given a real tool
design limitation, not hallucination. Replaced with a query on a real,
searchable filter combination guaranteed to return zero rows.

**A genuinely nuanced hallucination finding, kept rather than hidden.**
Asking about a fictional insurance product across repeated runs: once,
the model correctly said lost/delayed baggage is the airline's
responsibility per the retrieved policy text; another time, it stated
insurance "covers" baggage loss/delay -- misattributing one real
policy's content to another real, related policy, not inventing
anything. Separately, the Faithfulness judge's own claim decomposition
doesn't distinguish factual claims from reasonable meta-commentary
("sci-fi scenarios aren't typically covered"), so it flags both as
"unsupported" -- a real limitation of naive claim-decomposition
scoring, not of the agent. The regression test's pass criterion checks
the one thing that actually matters (never claim to offer a fictional
product) and reports Faithfulness as diagnostic detail, not a hard
gate on secondary claims -- see the docstring in
`app/scripts/hallucination_regression_test.py` for the full writeup.

## Open-source model comparison (Ollama)

`app/scripts/model_comparison.py` reruns the same live eval framework
above against local Ollama models instead of Claude, to compare
Traditional Chinese answer quality and inference speed against
open-weight alternatives on consumer hardware. See that script for the
current models and results.
