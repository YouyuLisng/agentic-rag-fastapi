-- Run once against the Supabase project's Postgres database (SQL
-- Editor, or `psql "$DATABASE_URL" -f db/schema.sql`).

create extension if not exists vector;
create extension if not exists pgcrypto;  -- gen_random_uuid()

-- Structured tour data. Queried directly (search_tours / get_tour_detail
-- tools) -- not embedded, exact/filtered lookups only.
create table if not exists tours (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    country text not null,
    location text not null,
    days int not null,
    budget_twd int not null,
    suitable_for text[] not null default '{}',
    summary text not null,
    -- [{day, title, description, breakfast, lunch, dinner, hotel}, ...]
    itinerary jsonb not null default '[]',
    capacity int not null default 20,
    enrolled_count int not null default 0,
    created_at timestamptz not null default now()
);

-- `create table if not exists` above is a no-op against an already-
-- existing tours table (this schema shipped once already, before
-- these columns existed) -- these keep schema.sql safely re-runnable
-- against a live DB instead of requiring a drop/recreate.
alter table tours add column if not exists capacity int not null default 20;
alter table tours add column if not exists enrolled_count int not null default 0;

-- Policy knowledge base. Embedded, queried by search_knowledge via
-- cosine similarity.
create table if not exists policy_chunks (
    id uuid primary key default gen_random_uuid(),
    document_slug text not null,
    title text not null,
    content text not null,
    chunk_index int not null,
    embedding vector(1024),
    created_at timestamptz not null default now()
);

create index if not exists policy_chunks_embedding_idx
    on policy_chunks using hnsw (embedding vector_cosine_ops);

create or replace function match_policy_chunks(
    query_embedding vector(1024),
    match_count int default 5
)
returns table (
    id uuid,
    document_slug text,
    title text,
    content text,
    similarity float
)
language sql stable
as $$
    select
        id, document_slug, title, content,
        1 - (embedding <=> query_embedding) as similarity
    from policy_chunks
    order by embedding <=> query_embedding
    limit match_count;
$$;

-- Mode B: per-upload document chunks. document_id scopes retrieval to
-- one specific upload -- never mixed with the policy knowledge base or
-- another upload, since there's no auth/session model beyond that id.
create table if not exists document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null,
    filename text not null,
    chunk_index int not null,
    content text not null,
    embedding vector(1024),
    created_at timestamptz not null default now()
);

create index if not exists document_chunks_embedding_idx
    on document_chunks using hnsw (embedding vector_cosine_ops);

create index if not exists document_chunks_document_id_idx
    on document_chunks (document_id);

create or replace function match_document_chunks(
    target_document_id uuid,
    query_embedding vector(1024),
    match_count int default 5
)
returns table (
    id uuid,
    chunk_index int,
    content text,
    similarity float
)
language sql stable
as $$
    select
        id, chunk_index, content,
        1 - (embedding <=> query_embedding) as similarity
    from document_chunks
    where document_id = target_document_id
    order by embedding <=> query_embedding
    limit match_count;
$$;
