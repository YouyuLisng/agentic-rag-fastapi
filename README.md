# Agentic RAG Travel Assistant

Hand-rolled multi-step agentic loop (Claude tool use, no framework) that
routes between semantic retrieval (RAG over a small travel-policy
knowledge base) and structured tool calls (querying a tour catalog),
deciding autonomously which path -- or how many steps -- a question
needs. A LangChain-based reimplementation of the same system follows
once the hand-rolled version is complete, as a direct comparison.

## Stack

FastAPI, Claude (Anthropic API), Voyage embeddings, Postgres + pgvector
(Supabase), Next.js frontend (separate project) consuming this API over
SSE for streamed decision visualization.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, DATABASE_URL
```

Run `db/schema.sql` once against the Supabase project (SQL Editor, or
`psql "$DATABASE_URL" -f db/schema.sql`) to create the `tours` and
`policy_chunks` tables plus the `match_policy_chunks` retrieval function.

```bash
uv run uvicorn app.main:app --reload
```

`GET /health` checks the app is up and the DB pool can reach Postgres.

## Project layout

```
app/
  main.py       FastAPI app, lifespan-managed DB pool, CORS
  config.py     pydantic-settings (env vars)
  db.py         psycopg async connection pool
  routers/      API route handlers (chat/SSE endpoint, etc.)
  agent/        the hand-rolled agentic loop + tool definitions
  rag/          embedding + vector retrieval
  models/       pydantic schemas
  scripts/      one-off scripts (data seeding)
db/
  schema.sql    pgvector schema + retrieval function
data/
  policies/     source markdown for the policy knowledge base
tests/
```

## Testing

```bash
uv run pytest
uv run ruff check .
uv run pyright
```
