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
